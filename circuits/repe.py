"""
Representation Engineering (RepE) layer scanning.

Extracts a "reading vector" at every layer using contrastive difference vectors.
Two methods are supported:
  - PCA: first principal component of difference vectors (Zou et al. 2023 default)
  - Mean difference: normalized mean of difference vectors (supervised alternative)

This traces where the subject-number signal emerges, peaks, and fades
across the network — the signal emergence profile.

Inspired by Zou et al. (2023) "Representation Engineering: A Top-Down Approach
to AI Transparency".

Usage:
    uv run python -m circuits.repe --model gemma-2b --langs en es --method both
    uv run python -m circuits.repe --model bloom-3b --langs en es --method mean_diff
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from circuits.config import ALL_LANGS, DATA_DIR, RESULTS_DIR
from circuits.model import load_model, tokenize_pair, is_multi_token_lang
from circuits.data import load_sva_dataset


def collect_contrastive_residuals(
    model,
    examples: list,
    device: str,
) -> tuple:
    """
    Collect residual stream activations for both clean and corrupted inputs
    at every layer, returning per-example difference vectors.

    Returns:
        diffs: (n_layers, n_examples, d_model) — clean minus corrupted residuals
        clean_acts: (n_layers, n_examples, d_model)
        corrupted_acts: (n_layers, n_examples, d_model)
        labels: (n_examples,) — 0 if clean=singular, 1 if clean=plural
    """
    n_layers = model.cfg.n_layers
    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]

    clean_all, corrupted_all, labels = [], [], []

    for ex in tqdm(examples, desc="Collecting contrastive residuals"):
        mt = is_multi_token_lang(ex.get("lang", "en"))

        # Clean forward pass
        tokens_c, _, _ = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )
        with torch.no_grad():
            _, cache_c = model.run_with_cache(tokens_c, names_filter=hook_names)

        # Corrupted forward pass
        tokens_x, _, _ = tokenize_pair(
            model, ex["corrupted"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )
        with torch.no_grad():
            _, cache_x = model.run_with_cache(tokens_x, names_filter=hook_names)

        clean_layers = []
        corrupted_layers = []
        for layer in range(n_layers):
            c_act = cache_c[f"blocks.{layer}.hook_resid_post"][0, -1, :].detach().cpu().numpy()
            x_act = cache_x[f"blocks.{layer}.hook_resid_post"][0, -1, :].detach().cpu().numpy()
            clean_layers.append(c_act)
            corrupted_layers.append(x_act)

        clean_all.append(np.stack(clean_layers))       # (n_layers, d_model)
        corrupted_all.append(np.stack(corrupted_layers))
        labels.append(0)  # clean = correct = singular-subject convention

    clean_acts = np.stack(clean_all, axis=1)       # (n_layers, n_examples, d_model)
    corrupted_acts = np.stack(corrupted_all, axis=1)
    diffs = clean_acts - corrupted_acts             # the number signal at each layer
    labels = np.array(labels)

    return diffs, clean_acts, corrupted_acts, labels


def compute_reading_vectors(diffs: np.ndarray, method: str = "pca") -> tuple:
    """
    At each layer, extract the reading vector using the specified method.

    Methods:
        "pca": First principal component of difference vectors.
               This is the default in Zou et al. (2023). Unsupervised —
               finds the direction of maximum variance in the diffs.
        "mean_diff": Normalized mean of difference vectors.
                     Supervised alternative — uses the average direction
                     of change. More robust when n_examples << d_model.

    Args:
        diffs: (n_layers, n_examples, d_model)
        method: "pca" or "mean_diff"

    Returns:
        reading_vectors: (n_layers, d_model) — reading vector at each layer
        explained_variance: (n_layers,) — for PCA: fraction of variance
            captured by PC1; for mean_diff: cosine consistency (mean cosine
            between each diff and the mean direction)
    """
    n_layers = diffs.shape[0]
    d_model = diffs.shape[2]
    reading_vectors = np.zeros((n_layers, d_model))
    explained_variance = np.zeros(n_layers)

    for layer in range(n_layers):
        layer_diffs = diffs[layer]  # (n_examples, d_model)

        if method == "pca":
            if layer_diffs.shape[0] < 2:
                continue
            n_components = min(5, layer_diffs.shape[0], layer_diffs.shape[1])
            pca = PCA(n_components=n_components)
            pca.fit(layer_diffs)
            reading_vectors[layer] = pca.components_[0]
            explained_variance[layer] = pca.explained_variance_ratio_[0]

        elif method == "mean_diff":
            mean_dir = layer_diffs.mean(axis=0)  # (d_model,)
            norm = np.linalg.norm(mean_dir)
            if norm < 1e-10:
                continue
            reading_vectors[layer] = mean_dir / norm

            # Cosine consistency: how aligned are individual diffs with the mean?
            diff_norms = np.linalg.norm(layer_diffs, axis=1, keepdims=True)
            diff_norms = np.maximum(diff_norms, 1e-10)
            cosines = (layer_diffs / diff_norms) @ (mean_dir / norm)
            explained_variance[layer] = np.mean(np.abs(cosines))

    return reading_vectors, explained_variance


def compute_signal_profile(
    diffs: np.ndarray,
    reading_vectors: np.ndarray,
) -> dict:
    """
    Compute the signal emergence profile: at each layer, how strongly does
    the reading vector separate the contrastive pairs?

    Metrics per layer:
        - signal_magnitude: mean absolute projection onto reading vector
        - signal_std: std of projections (spread)
        - signal_snr: signal-to-noise ratio (magnitude / std of noise)

    Args:
        diffs: (n_layers, n_examples, d_model)
        reading_vectors: (n_layers, d_model)

    Returns:
        dict with per-layer metrics
    """
    n_layers = diffs.shape[0]
    signal_magnitude = np.zeros(n_layers)
    signal_std = np.zeros(n_layers)
    signal_snr = np.zeros(n_layers)
    diff_norms = np.zeros(n_layers)

    for layer in range(n_layers):
        rv = reading_vectors[layer]
        rv_norm = np.linalg.norm(rv)
        if rv_norm < 1e-10:
            continue

        rv_unit = rv / rv_norm

        # Project each example's diff onto the reading vector
        projections = diffs[layer] @ rv_unit  # (n_examples,)

        signal_magnitude[layer] = np.mean(np.abs(projections))
        signal_std[layer] = np.std(projections)

        # SNR: how consistent is the signal direction across examples?
        mean_proj = np.mean(projections)
        residuals = projections - mean_proj
        noise = np.std(residuals)
        signal_snr[layer] = abs(mean_proj) / noise if noise > 1e-10 else 0.0

        # Also track raw diff norms (total amount of change at each layer)
        diff_norms[layer] = np.mean(np.linalg.norm(diffs[layer], axis=1))

    return {
        "signal_magnitude": signal_magnitude,
        "signal_std": signal_std,
        "signal_snr": signal_snr,
        "diff_norms": diff_norms,
    }


def compute_reading_vector_accuracy(
    clean_acts: np.ndarray,
    corrupted_acts: np.ndarray,
    reading_vectors: np.ndarray,
) -> np.ndarray:
    """
    At each layer, test if the reading vector can classify clean vs corrupted
    above chance. This measures whether the concept direction actually encodes
    the number distinction in the full activations (not just the diffs).

    Uses a simple threshold classifier: project clean and corrupted activations
    onto the reading vector, classify based on which side of the midpoint they
    fall on.

    Returns:
        accuracy: (n_layers,) — classification accuracy per layer
    """
    n_layers = clean_acts.shape[0]
    n_examples = clean_acts.shape[1]
    accuracy = np.zeros(n_layers)

    for layer in range(n_layers):
        rv = reading_vectors[layer]
        rv_norm = np.linalg.norm(rv)
        if rv_norm < 1e-10:
            accuracy[layer] = 0.5
            continue

        rv_unit = rv / rv_norm

        # Project clean and corrupted onto reading vector
        clean_proj = clean_acts[layer] @ rv_unit     # (n_examples,)
        corrupted_proj = corrupted_acts[layer] @ rv_unit

        # Classification: clean should project higher (or lower) than corrupted
        # Try both conventions, take the better one
        correct_pos = np.sum(clean_proj > corrupted_proj)
        correct_neg = np.sum(clean_proj < corrupted_proj)
        accuracy[layer] = max(correct_pos, correct_neg) / n_examples

    return accuracy


def _run_method(method, diffs, clean_acts, corrupted_acts):
    """Run a single reading vector method and return all results."""
    reading_vectors, explained_variance = compute_reading_vectors(diffs, method=method)
    profile = compute_signal_profile(diffs, reading_vectors)
    accuracy = compute_reading_vector_accuracy(clean_acts, corrupted_acts, reading_vectors)
    return reading_vectors, explained_variance, profile, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="RepE-style layer scanning for subject-number signal"
    )
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--langs", nargs="+", default=["en"])
    parser.add_argument("--method", choices=["pca", "mean_diff", "both"], default="both",
                        help="Reading vector method: pca, mean_diff, or both")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=128,
                        help="Max examples per language")
    args = parser.parse_args()

    model = load_model(args.model, device=args.device)
    model_name = args.model.replace("-", "_")

    methods = ["pca", "mean_diff"] if args.method == "both" else [args.method]

    for lang in args.langs:
        data_path = f"{args.data_dir}/{lang}_sva.jsonl"
        dataset = load_sva_dataset(data_path)
        if args.max_examples and len(dataset) > args.max_examples:
            dataset = dataset[:args.max_examples]

        print(f"\n{'='*60}")
        print(f"RepE scanning: {args.model} / {lang.upper()} ({len(dataset)} examples)")
        print(f"{'='*60}")

        # Step 1: Collect contrastive residuals (shared across methods)
        diffs, clean_acts, corrupted_acts, labels = collect_contrastive_residuals(
            model, dataset, device=args.device
        )
        print(f"Diffs shape: {diffs.shape}")

        for method in methods:
            print(f"\n--- Method: {method} ---")
            reading_vectors, explained_variance, profile, accuracy = _run_method(
                method, diffs, clean_acts, corrupted_acts
            )

            # Save results with method suffix
            suffix = f"_{method}" if len(methods) > 1 else ""
            out_path = Path(args.out_dir) / f"repe_{model_name}_{lang}{suffix}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_path,
                reading_vectors=reading_vectors,
                explained_variance=explained_variance,
                accuracy=accuracy,
                method=method,
                **profile,
            )
            print(f"Saved → {out_path}")

            # Print summary
            ev_label = "ExpVar" if method == "pca" else "Consistency"
            print(f"\nSignal emergence profile ({lang.upper()}, {method}):")
            print(f"{'Layer':>6} {'Accuracy':>9} {'Magnitude':>10} {ev_label:>12} {'DiffNorm':>9}")
            for layer in range(diffs.shape[0]):
                bar = "█" * int(accuracy[layer] * 20)
                print(
                    f"  {layer:4d}  {accuracy[layer]:8.3f}  "
                    f"{profile['signal_magnitude'][layer]:9.3f}  "
                    f"{explained_variance[layer]:11.3f}  "
                    f"{profile['diff_norms'][layer]:8.3f}  {bar}"
                )


if __name__ == "__main__":
    main()
