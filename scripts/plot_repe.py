"""
Generate RepE-style figures: LAT scan plots comparing PCA vs mean_diff
reading vectors across layers, similar to Zou et al. (2023) Figure 3.

Usage:
    uv run python scripts/plot_repe.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "es", "fr", "sw"]
LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "sw": "Swahili"}
LANG_COLORS = {"en": "#2ecc71", "es": "#e74c3c", "fr": "#3498db", "sw": "#f39c12"}
MODEL_STYLES = {"gemma_2b": "-", "bloom_3b": "--"}
MODEL_NAMES = {"gemma_2b": "Gemma 2B", "bloom_3b": "BLOOM 3B"}


def load_repe(model, lang, method):
    """Load RepE results, trying method-suffixed name first, then plain."""
    path = RESULTS_DIR / f"repe_{model}_{lang}_{method}.npz"
    if not path.exists():
        # Fall back to unsuffixed (old format, PCA only)
        path = RESULTS_DIR / f"repe_{model}_{lang}.npz"
    if not path.exists():
        return None
    return dict(np.load(path))


def fig1_lat_scan_accuracy():
    """
    RepE-style LAT scan: accuracy per layer for each language.
    One panel per model, PCA vs mean_diff side by side.
    Similar to Zou et al. (2023) Figure 3.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, method in enumerate(["pca", "mean_diff"]):
        for row, model in enumerate(["gemma_2b", "bloom_3b"]):
            ax = axes[row, col]

            for lang in LANGS:
                d = load_repe(model, lang, method)
                if d is None:
                    continue
                acc = d["accuracy"]
                layers = np.arange(len(acc))
                ax.plot(layers, acc, "o-", color=LANG_COLORS[lang],
                        linewidth=2, markersize=4, label=LANG_NAMES[lang])

            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Chance")
            ax.set_ylim(0.4, 1.05)
            ax.set_xlabel("Layer", fontsize=11)
            ax.set_ylabel("Classification Accuracy", fontsize=11)
            method_label = "PCA (PC1)" if method == "pca" else "Mean Difference"
            ax.set_title(f"{MODEL_NAMES[model]} — {method_label}",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(alpha=0.15)

    plt.suptitle(
        "LAT Scan: Reading Vector Accuracy Across Layers\n"
        "(Higher = stronger number signal at that layer)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_repe_lat_scan.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig_repe_lat_scan.png")


def fig2_method_comparison():
    """
    Direct comparison: PCA vs mean_diff accuracy for the same model/language.
    One panel per language, both models overlaid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, lang in zip(axes.flat, LANGS):
        for model in ["gemma_2b", "bloom_3b"]:
            for method, ls, alpha in [("pca", "--", 0.5), ("mean_diff", "-", 1.0)]:
                d = load_repe(model, lang, method)
                if d is None:
                    continue
                acc = d["accuracy"]
                rel_depth = np.linspace(0, 1, len(acc))
                method_label = "PCA" if method == "pca" else "Mean"
                ax.plot(rel_depth, acc, f"o{ls}",
                        color="#2ecc71" if "gemma" in model else "#9b59b6",
                        linewidth=2, markersize=3, alpha=alpha,
                        label=f"{MODEL_NAMES[model]} ({method_label})")

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.set_title(f"{LANG_NAMES[lang]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Relative Depth", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.15)

    plt.suptitle(
        "PCA vs Mean Difference: Reading Vector Accuracy\n"
        "(Solid = Mean Diff, Dashed = PCA)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_repe_pca_vs_mean.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig_repe_pca_vs_mean.png")


def fig3_signal_profile_full():
    """
    Full signal profile: accuracy, magnitude, explained variance, diff norms.
    Four metrics × one model. Similar to a comprehensive LAT scan.
    """
    metrics = [
        ("accuracy", "Classification Accuracy", (0.4, 1.05)),
        ("signal_magnitude", "Signal Magnitude", None),
        ("explained_variance", "Explained Var. / Consistency", (0, 1.05)),
        ("diff_norms", "Diff Norms", None),
    ]

    for model in ["gemma_2b", "bloom_3b"]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Use mean_diff method if available, fall back to pca
        method = "mean_diff"

        for ax, (metric, ylabel, ylim) in zip(axes.flat, metrics):
            for lang in LANGS:
                d = load_repe(model, lang, method)
                if d is None:
                    d = load_repe(model, lang, "pca")
                if d is None:
                    continue
                values = d[metric]
                layers = np.arange(len(values))
                ax.plot(layers, values, "o-", color=LANG_COLORS[lang],
                        linewidth=2, markersize=4, label=LANG_NAMES[lang])

            ax.set_xlabel("Layer", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(ylabel, fontsize=11, fontweight="bold")
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.15)

        method_label = "Mean Diff" if method == "mean_diff" else "PCA"
        plt.suptitle(
            f"{MODEL_NAMES[model]} — Full Signal Profile ({method_label})",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        fname = f"fig_repe_profile_{model}.png"
        plt.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def fig4_cross_model_accuracy_overlay():
    """
    The key cross-model figure: accuracy curves for both models on the same
    plot, normalized to relative depth. One panel per language.
    Uses mean_diff (the better method).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    method = "mean_diff"

    for ax, lang in zip(axes.flat, LANGS):
        for model, color in [("gemma_2b", "#2ecc71"), ("bloom_3b", "#9b59b6")]:
            d = load_repe(model, lang, method)
            if d is None:
                d = load_repe(model, lang, "pca")  # fallback
            if d is None:
                continue
            acc = d["accuracy"]
            rel_depth = np.linspace(0, 1, len(acc))
            n_layers = len(acc)
            ax.plot(rel_depth, acc, "o-", color=color, linewidth=2.5,
                    markersize=4, label=f"{MODEL_NAMES[model]} ({n_layers}L)")

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Chance")
        ax.set_title(f"{LANG_NAMES[lang]}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Relative Depth (0=input, 1=output)", fontsize=10)
        ax.set_ylabel("Reading Vector Accuracy", fontsize=10)
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.15)

    plt.suptitle(
        "Cross-Model Signal Emergence: Gemma 2B vs BLOOM 3B\n"
        "Reading vector accuracy across layers (mean difference method)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_repe_cross_model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig_repe_cross_model_accuracy.png")


if __name__ == "__main__":
    fig1_lat_scan_accuracy()
    fig2_method_comparison()
    fig3_signal_profile_full()
    fig4_cross_model_accuracy_overlay()
    print("\nAll RepE figures generated.")
