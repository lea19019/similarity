"""
Cross-model comparison of information flow topology.

Compares how two models (e.g. Gemma 2B vs BLOOM 3B) route the subject-number
signal through their layers, using RepE signal profiles and CKA.

The key insight: models have different architectures (different layer counts,
head counts, dimensions), but we can compare them by normalizing to relative
depth (0-100%) and asking whether the "shape" of information flow is similar.

Usage:
    uv run python -m circuits.cross_model \
        --model-a gemma-2b --model-b bloom-3b \
        --langs en es fr
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import spearmanr

from circuits.config import ALL_LANGS, DATA_DIR, RESULTS_DIR
from circuits.geometry import linear_cka, svcca, procrustes_distance


def load_repe_profile(results_dir: str, model_key: str, lang: str) -> dict:
    """Load a RepE signal profile for a given model and language."""
    model_name = model_key.replace("-", "_")
    path = Path(results_dir) / f"repe_{model_name}_{lang}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"RepE profile not found: {path}\n"
            f"Run: uv run python -m circuits.repe --model {model_key} --langs {lang}"
        )
    return dict(np.load(path))


def normalize_profile(values: np.ndarray) -> tuple:
    """
    Normalize a per-layer signal profile to relative depth [0, 1].

    Returns:
        relative_depth: (n_layers,) values in [0, 1]
        normalized_values: (n_layers,) values scaled to [0, 1]
    """
    n_layers = len(values)
    relative_depth = np.linspace(0, 1, n_layers)
    # Normalize values to [0, 1] for shape comparison
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10:
        normalized = np.zeros_like(values)
    else:
        normalized = (values - vmin) / (vmax - vmin)
    return relative_depth, normalized


def interpolate_to_common_grid(
    depth_a: np.ndarray,
    values_a: np.ndarray,
    depth_b: np.ndarray,
    values_b: np.ndarray,
    n_points: int = 100,
) -> tuple:
    """
    Interpolate two profiles with different layer counts onto a common
    relative-depth grid for direct comparison.

    Returns:
        common_depth: (n_points,)
        interp_a: (n_points,)
        interp_b: (n_points,)
    """
    common_depth = np.linspace(0, 1, n_points)

    f_a = interp1d(depth_a, values_a, kind="linear", fill_value="extrapolate")
    f_b = interp1d(depth_b, values_b, kind="linear", fill_value="extrapolate")

    return common_depth, f_a(common_depth), f_b(common_depth)


def compare_profiles(profile_a: np.ndarray, profile_b: np.ndarray) -> dict:
    """
    Compare two signal profiles (already on the same grid) using multiple
    similarity metrics.

    Returns dict with:
        - pearson: Pearson correlation of the two curves
        - spearman: Spearman rank correlation
        - peak_shift: absolute difference in peak positions (0 = same relative depth)
        - l2_distance: L2 distance between normalized curves
        - cosine: cosine similarity of the two curves as vectors
    """
    from scipy.stats import pearsonr

    # Pearson and Spearman correlation
    r_pearson, p_pearson = pearsonr(profile_a, profile_b)
    r_spearman, p_spearman = spearmanr(profile_a, profile_b)

    # Peak position comparison
    peak_a = np.argmax(profile_a) / len(profile_a)
    peak_b = np.argmax(profile_b) / len(profile_b)
    peak_shift = abs(peak_a - peak_b)

    # L2 distance
    l2 = np.sqrt(np.mean((profile_a - profile_b) ** 2))

    # Cosine similarity
    norm_a = np.linalg.norm(profile_a)
    norm_b = np.linalg.norm(profile_b)
    cosine = float(profile_a @ profile_b / (norm_a * norm_b)) if norm_a > 1e-10 and norm_b > 1e-10 else 0.0

    return {
        "pearson": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman": float(r_spearman),
        "spearman_p": float(p_spearman),
        "peak_shift": float(peak_shift),
        "peak_a": float(peak_a),
        "peak_b": float(peak_b),
        "l2_distance": float(l2),
        "cosine": float(cosine),
    }


def cross_model_cka(
    results_dir: str,
    model_a: str,
    model_b: str,
    lang: str,
) -> np.ndarray:
    """
    Compute CKA between every layer of model A and every layer of model B
    for the same language. Uses the reading vectors as compact representations
    of what each layer encodes about the number signal.

    This produces a (n_layers_a, n_layers_b) alignment heatmap showing where
    the two models develop similar representations.

    Returns:
        cka_matrix: (n_layers_a, n_layers_b)
    """
    profile_a = load_repe_profile(results_dir, model_a, lang)
    profile_b = load_repe_profile(results_dir, model_b, lang)

    rv_a = profile_a["reading_vectors"]  # (n_layers_a, d_model_a)
    rv_b = profile_b["reading_vectors"]  # (n_layers_b, d_model_b)

    n_a, n_b = rv_a.shape[0], rv_b.shape[0]
    cka_matrix = np.zeros((n_a, n_b))

    for i in range(n_a):
        for j in range(n_b):
            # CKA needs (n_examples, d) matrices — here each "example" is
            # a dimension of the reading vector, reshaped for comparison
            # Instead, use the raw vectors directly: treat each reading vector
            # as a 1D signal and compute cosine similarity as a proxy,
            # or use the full diff matrices if available.
            # For reading vectors alone, cosine is the natural metric.
            pass

    # Since reading vectors have different dimensions (d_model differs),
    # CKA on the reading vectors alone isn't meaningful. Instead, we compare
    # the signal profiles (which are scalar per layer) using profile comparison.
    # Full CKA requires same-example activations from both models, which
    # needs the cross_model_cka_from_activations function below.
    return cka_matrix


def _collect_residuals_raw(model, examples: list, device: str) -> np.ndarray:
    """
    Collect residual stream activations at the final token position for each
    layer, tokenizing each sentence with the model's own tokenizer.

    Unlike geometry.collect_layer_activations, this doesn't require valid
    verb token IDs — it just tokenizes the clean prompt and reads residuals.
    This makes it safe for cross-model use where the same sentence may
    tokenize differently.

    Returns: (n_layers, n_examples, d_model)
    """
    import torch
    from tqdm import tqdm

    n_layers = model.cfg.n_layers
    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    all_acts = []

    for ex in tqdm(examples, desc="Collecting residuals"):
        tokens = model.to_tokens(ex["clean"])
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        layer_acts = []
        for layer in range(n_layers):
            act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].detach().cpu().numpy()
            layer_acts.append(act)
        all_acts.append(np.stack(layer_acts))

    return np.stack(all_acts, axis=1)  # (n_layers, n_examples, d_model)


def cross_model_cka_from_activations(
    model_a,
    model_b,
    examples: list,
    device: str,
) -> np.ndarray:
    """
    Compute CKA between every layer of model A and every layer of model B
    using actual activations from the same examples.

    This is the gold-standard cross-model comparison: feed identical sentences
    through both models and compare the representation geometry at each layer
    pair. Each model tokenizes the sentence with its own tokenizer.

    Args:
        model_a, model_b: HookedTransformer instances
        examples: list of SVA examples (same for both models)
        device: "cuda" or "cpu"

    Returns:
        cka_matrix: (n_layers_a, n_layers_b)
    """
    print("Collecting activations from model A...")
    acts_a = _collect_residuals_raw(model_a, examples, device)  # (n_layers_a, n_ex, d_a)
    print("Collecting activations from model B...")
    acts_b = _collect_residuals_raw(model_b, examples, device)  # (n_layers_b, n_ex, d_b)

    n_a = acts_a.shape[0]
    n_b = acts_b.shape[0]
    n_ex = min(acts_a.shape[1], acts_b.shape[1])

    cka_matrix = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            cka_matrix[i, j] = linear_cka(acts_a[i, :n_ex], acts_b[j, :n_ex])

    return cka_matrix


def compare_flow_topology(
    results_dir: str,
    model_a: str,
    model_b: str,
    langs: list,
    n_grid: int = 100,
) -> dict:
    """
    Full cross-model flow topology comparison.

    For each language, loads RepE profiles from both models, normalizes
    to relative depth, and compares the signal emergence curves.

    Returns:
        dict with per-language and aggregate comparison metrics
    """
    results = {
        "model_a": model_a,
        "model_b": model_b,
        "langs": langs,
        "per_lang": {},
        "metrics": ["accuracy", "signal_magnitude", "explained_variance", "diff_norms"],
    }

    all_comparisons = {m: [] for m in results["metrics"]}

    for lang in langs:
        try:
            pa = load_repe_profile(results_dir, model_a, lang)
            pb = load_repe_profile(results_dir, model_b, lang)
        except FileNotFoundError as e:
            print(f"Skipping {lang}: {e}")
            continue

        lang_results = {}

        for metric in results["metrics"]:
            vals_a = pa[metric]
            vals_b = pb[metric]

            depth_a, norm_a = normalize_profile(vals_a)
            depth_b, norm_b = normalize_profile(vals_b)

            common_depth, interp_a, interp_b = interpolate_to_common_grid(
                depth_a, norm_a, depth_b, norm_b, n_grid
            )

            comparison = compare_profiles(interp_a, interp_b)
            comparison["raw_a"] = vals_a.tolist()
            comparison["raw_b"] = vals_b.tolist()
            comparison["norm_a"] = interp_a.tolist()
            comparison["norm_b"] = interp_b.tolist()
            comparison["common_depth"] = common_depth.tolist()

            lang_results[metric] = comparison
            all_comparisons[metric].append(comparison)

        results["per_lang"][lang] = lang_results

    # Aggregate: mean correlation across languages for each metric
    agg = {}
    for metric in results["metrics"]:
        if all_comparisons[metric]:
            agg[metric] = {
                "mean_pearson": np.mean([c["pearson"] for c in all_comparisons[metric]]),
                "mean_spearman": np.mean([c["spearman"] for c in all_comparisons[metric]]),
                "mean_peak_shift": np.mean([c["peak_shift"] for c in all_comparisons[metric]]),
                "mean_cosine": np.mean([c["cosine"] for c in all_comparisons[metric]]),
            }
    results["aggregate"] = agg

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model comparison of information flow topology"
    )
    parser.add_argument("--model-a", default="gemma-2b")
    parser.add_argument("--model-b", default="bloom-3b")
    parser.add_argument("--langs", nargs="+", default=["en", "es"])
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=64,
                        help="Max examples per language for CKA computation")
    parser.add_argument("--skip-cka", action="store_true",
                        help="Skip CKA (requires loading both models)")
    args = parser.parse_args()

    # Phase 1: Compare RepE signal profiles
    print("=" * 60)
    print("Phase 1: Flow Topology Comparison (RepE profiles)")
    print("=" * 60)

    results = compare_flow_topology(
        args.results_dir, args.model_a, args.model_b, args.langs
    )

    # Print summary
    print(f"\nFlow comparison: {args.model_a} vs {args.model_b}")
    print(f"{'Language':>8} {'Metric':>20} {'Pearson r':>10} {'Peak Δ':>8} {'Cosine':>8}")
    for lang in args.langs:
        if lang not in results.get("per_lang", {}):
            continue
        for metric in results["metrics"]:
            c = results["per_lang"][lang][metric]
            print(
                f"  {lang.upper():>6} {metric:>20}  "
                f"{c['pearson']:>9.3f}  {c['peak_shift']:>7.3f}  {c['cosine']:>7.3f}"
            )

    print(f"\nAggregate (mean across languages):")
    for metric, agg in results.get("aggregate", {}).items():
        print(f"  {metric:>20}: r={agg['mean_pearson']:.3f}, "
              f"peak_shift={agg['mean_peak_shift']:.3f}, "
              f"cosine={agg['mean_cosine']:.3f}")

    # Phase 2: Cross-model CKA (requires both models loaded)
    if not args.skip_cka:
        print(f"\n{'='*60}")
        print("Phase 2: Cross-Model CKA")
        print("=" * 60)

        from circuits.model import load_model
        from circuits.data import load_sva_dataset

        model_a = load_model(args.model_a, device=args.device)
        model_b = load_model(args.model_b, device=args.device)

        for lang in args.langs:
            data_path = f"{args.data_dir}/{lang}_sva.jsonl"
            dataset = load_sva_dataset(data_path)
            if args.max_examples and len(dataset) > args.max_examples:
                dataset = dataset[:args.max_examples]

            print(f"\nCKA: {args.model_a} vs {args.model_b} on {lang.upper()} "
                  f"({len(dataset)} examples)")

            cka_matrix = cross_model_cka_from_activations(
                model_a, model_b, dataset, device=args.device
            )

            # Save CKA matrix
            out_path = (Path(args.out_dir) /
                        f"cross_cka_{args.model_a}_{args.model_b}_{lang}.npz")
            np.savez(out_path, cka_matrix=cka_matrix)
            print(f"Saved CKA matrix → {out_path}")

            # Print diagonal CKA (matched relative depth)
            n_a, n_b = cka_matrix.shape
            print(f"  Diagonal CKA (matched relative depth):")
            for frac in [0.0, 0.25, 0.5, 0.72, 0.85, 1.0]:
                i = min(int(frac * (n_a - 1)), n_a - 1)
                j = min(int(frac * (n_b - 1)), n_b - 1)
                print(f"    depth={frac:.0%}: L{i}(A) vs L{j}(B) = {cka_matrix[i,j]:.3f}")

            # Find best-matching layer for each layer of model A
            best_match = np.argmax(cka_matrix, axis=1)
            print(f"  Best layer alignment:")
            for i in range(0, n_a, max(1, n_a // 6)):
                j = best_match[i]
                print(f"    {args.model_a} L{i} → {args.model_b} L{j} "
                      f"(CKA={cka_matrix[i,j]:.3f})")

    # Save full comparison results
    import json
    out_path = (Path(args.out_dir) /
                f"cross_model_{args.model_a}_{args.model_b}.json")

    # Convert numpy types for JSON serialization
    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)
    print(f"\nSaved full comparison → {out_path}")


if __name__ == "__main__":
    main()
