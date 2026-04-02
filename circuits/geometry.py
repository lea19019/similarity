"""
Cross-lingual geometry comparison metrics.

Measures how similar the representations and weight-level circuits are
across languages using CKA, SVCCA, RSA, Procrustes, and cosine similarity.

Also collects per-layer residual stream activations for the comparison.

Usage:
    uv run python -m circuits.geometry --langs en es tr sw --model gemma-2b
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from tqdm import tqdm

from circuits.config import ALL_LANGS, DATA_DIR, RESULTS_DIR


# ── Geometry metrics ───────────────────────────────────────────────────────


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Centered Kernel Alignment with linear kernel.

    Measures global similarity between two representation matrices,
    invariant to orthogonal transformations and isotropic scaling.

    Args:
        X: (n_examples, d1) — activation matrix for language A
        Y: (n_examples, d2) — activation matrix for language B

    Returns: CKA similarity in [0, 1]. 1 = identical geometry.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def svcca(X: np.ndarray, Y: np.ndarray, threshold: float = 0.99) -> float:
    """
    Singular Vector Canonical Correlation Analysis.

    1. SVD on X and Y, keep enough components for `threshold` variance
    2. CCA on the reduced representations
    3. Return mean canonical correlation

    Args:
        X: (n_examples, d1)
        Y: (n_examples, d2)
        threshold: cumulative variance threshold for SVD truncation

    Returns: mean CCA correlation in [0, 1]
    """
    def _svd_reduce(M, thresh):
        U, S, _ = np.linalg.svd(M - M.mean(axis=0), full_matrices=False)
        var = np.cumsum(S ** 2) / np.sum(S ** 2)
        k = max(1, int(np.searchsorted(var, thresh) + 1))
        return U[:, :k] * S[:k]

    X_red = _svd_reduce(X, threshold)
    Y_red = _svd_reduce(Y, threshold)

    # CCA: find canonical correlations between reduced representations
    k = min(X_red.shape[1], Y_red.shape[1])
    if k == 0:
        return 0.0

    # Cross-covariance via SVD
    Qx, _ = np.linalg.qr(X_red)
    Qy, _ = np.linalg.qr(Y_red)
    _, S_cca, _ = np.linalg.svd(Qx.T @ Qy, full_matrices=False)

    # Canonical correlations are clipped to [0, 1]
    correlations = np.clip(S_cca[:k], 0.0, 1.0)
    return float(np.mean(correlations))


def rsa(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Representational Similarity Analysis.

    Compares representational geometries via second-order similarity:
    correlation of pairwise distance matrices.

    Args:
        X: (n_examples, d1)
        Y: (n_examples, d2)

    Returns: Pearson correlation of pairwise distance vectors in [-1, 1]
    """
    if X.shape[0] < 3:
        return 0.0
    dx = pdist(X, metric="correlation")
    dy = pdist(Y, metric="correlation")
    # Handle constant distance vectors
    if np.std(dx) < 1e-10 or np.std(dy) < 1e-10:
        return 0.0
    return float(pearsonr(dx, dy)[0])


def procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Orthogonal Procrustes alignment distance.

    Finds optimal rotation R minimizing ||X - Y @ R||_F after
    centering and normalizing. Lower = more similar.

    Returns: Procrustes distance (0 = identical up to rotation/scaling)
    """
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    norm_x = np.linalg.norm(X_c)
    norm_y = np.linalg.norm(Y_c)
    if norm_x < 1e-10 or norm_y < 1e-10:
        return float("inf")

    X_c = X_c / norm_x
    Y_c = Y_c / norm_y

    # Pad to same dimension if needed
    d = max(X_c.shape[1], Y_c.shape[1])
    if X_c.shape[1] < d:
        X_c = np.pad(X_c, ((0, 0), (0, d - X_c.shape[1])))
    if Y_c.shape[1] < d:
        Y_c = np.pad(Y_c, ((0, 0), (0, d - Y_c.shape[1])))

    R, _ = orthogonal_procrustes(Y_c, X_c)
    return float(np.linalg.norm(X_c - Y_c @ R))


def cosine_task_projection_similarity(
    task_weights_a: np.ndarray,
    task_weights_b: np.ndarray,
) -> np.ndarray:
    """
    Per-head cosine similarity of task projection vectors between two languages.

    Args:
        task_weights_a: (n_layers, n_heads, d_model) from circuit_map
        task_weights_b: (n_layers, n_heads, d_model) from circuit_map

    Returns: (n_layers, n_heads) cosine similarities in [-1, 1]
    """
    norm_a = np.linalg.norm(task_weights_a, axis=-1, keepdims=True) + 1e-10
    norm_b = np.linalg.norm(task_weights_b, axis=-1, keepdims=True) + 1e-10
    a_hat = task_weights_a / norm_a
    b_hat = task_weights_b / norm_b
    return (a_hat * b_hat).sum(axis=-1)


# ── Activation collection ─────────────────────────────────────────────────


def collect_layer_activations(
    model,
    examples: list,
    device: str,
) -> np.ndarray:
    """
    Collect residual stream activations at the final token for each layer.

    Runs forward passes and caches hook_resid_post at every layer.

    Args:
        model: HookedTransformer
        examples: list of SVA examples
        device: "cuda" or "cpu"

    Returns: (n_layers, n_examples, d_model) numpy array
    """
    import torch
    from circuits.model import tokenize_pair, is_multi_token_lang

    n_layers = model.cfg.n_layers
    all_acts = []

    for ex in tqdm(examples, desc="Collecting activations"):
        mt = is_multi_token_lang(ex.get("lang", "en"))
        tokens, _, _ = tokenize_pair(model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt)

        hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        layer_acts = []
        for layer in range(n_layers):
            act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].detach().cpu().numpy()
            layer_acts.append(act)
        all_acts.append(np.stack(layer_acts))  # (n_layers, d_model)

    return np.stack(all_acts, axis=1)  # (n_layers, n_examples, d_model)


# ── Full comparison pipeline ──────────────────────────────────────────────


def compute_pairwise_geometry(
    activations: dict,
    task_weights: dict,
    langs: list,
) -> dict:
    """
    Compute all cross-lingual geometry metrics for all language pairs.

    Args:
        activations: {lang: (n_layers, n_examples, d_model)}
        task_weights: {lang: (n_layers, n_heads, d_model)} from circuit_map
        langs: list of language codes to compare

    Returns dict with:
        "cka_per_layer": (n_layers, n_pairs)
        "svcca_per_layer": (n_layers, n_pairs)
        "rsa_per_layer": (n_layers, n_pairs)
        "procrustes_per_layer": (n_layers, n_pairs)
        "task_cosine": (n_pairs, n_layers, n_heads) if task_weights provided
        "pair_labels": list of "lang_a-lang_b" strings
        "convergence": (n_layers,) mean CKA across all pairs per layer
    """
    pairs = list(combinations(langs, 2))
    pair_labels = [f"{a}-{b}" for a, b in pairs]

    # Get n_layers from first available activation
    first_lang = langs[0]
    n_layers = activations[first_lang].shape[0]

    cka_arr = np.zeros((n_layers, len(pairs)))
    svcca_arr = np.zeros((n_layers, len(pairs)))
    rsa_arr = np.zeros((n_layers, len(pairs)))
    proc_arr = np.zeros((n_layers, len(pairs)))

    for pi, (la, lb) in enumerate(pairs):
        # Use the minimum number of examples between the two languages
        n_ex = min(activations[la].shape[1], activations[lb].shape[1])
        for layer in range(n_layers):
            X = activations[la][layer, :n_ex, :]
            Y = activations[lb][layer, :n_ex, :]
            cka_arr[layer, pi] = linear_cka(X, Y)
            svcca_arr[layer, pi] = svcca(X, Y)
            rsa_arr[layer, pi] = rsa(X, Y)
            proc_arr[layer, pi] = procrustes_distance(X, Y)

    results = {
        "cka_per_layer": cka_arr,
        "svcca_per_layer": svcca_arr,
        "rsa_per_layer": rsa_arr,
        "procrustes_per_layer": proc_arr,
        "pair_labels": np.array(pair_labels),
        "convergence": cka_arr.mean(axis=1),
    }

    # Task projection cosine similarity (if circuit_map results available)
    if task_weights:
        n_heads = task_weights[first_lang].shape[1]
        task_cos = np.zeros((len(pairs), n_layers, n_heads))
        for pi, (la, lb) in enumerate(pairs):
            task_cos[pi] = cosine_task_projection_similarity(
                task_weights[la], task_weights[lb]
            )
        results["task_cosine"] = task_cos

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-lingual geometry comparison"
    )
    parser.add_argument("--langs", nargs="+", default=ALL_LANGS)
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=128,
                        help="Max examples per language for activation collection")
    args = parser.parse_args()

    from circuits.model import load_model
    from circuits.data import load_sva_dataset

    model = load_model(args.model, device=args.device)

    # Collect activations for each language
    activations = {}
    for lang in args.langs:
        data_path = f"{args.data_dir}/{lang}_sva.jsonl"
        dataset = load_sva_dataset(data_path)
        if args.max_examples and len(dataset) > args.max_examples:
            dataset = dataset[:args.max_examples]
        print(f"Collecting {lang.upper()} activations ({len(dataset)} examples)...")
        acts = collect_layer_activations(model, dataset, device=args.device)
        activations[lang] = acts

        # Save per-language activations
        act_path = Path(args.out_dir) / f"activations_{lang}.npz"
        np.savez_compressed(act_path, activations=acts)
        print(f"Saved activations → {act_path}")

    # Load task weights from circuit_map results (if available)
    task_weights = {}
    for lang in args.langs:
        cm_path = Path(args.results_dir) / f"circuit_map_{lang}.npz"
        if cm_path.exists():
            data = np.load(cm_path)
            task_weights[lang] = data["head_task_weights"]

    # Compute all pairwise metrics
    print("Computing geometry metrics...")
    results = compute_pairwise_geometry(activations, task_weights, args.langs)

    out_path = Path(args.out_dir) / "geometry.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **results)
    print(f"Saved geometry results → {out_path}")

    # Print summary
    pair_labels = results["pair_labels"]
    convergence = results["convergence"]
    print(f"\nPer-layer mean CKA convergence:")
    for layer, score in enumerate(convergence):
        bar = "█" * int(score * 40)
        print(f"  Layer {layer:2d}: {score:.3f} {bar}")

    print(f"\nPair-wise CKA at final layer:")
    for pi, label in enumerate(pair_labels):
        score = results["cka_per_layer"][-1, pi]
        print(f"  {label}: {score:.3f}")


if __name__ == "__main__":
    main()
