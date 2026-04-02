"""
Generate all paper figures from saved result files.

Usage:
    uv run python -m circuits.plotting --results-dir results --out-dir results/figures
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from circuits.config import ALL_LANGS, LANG_CONFIGS, RESULTS_DIR, FIGURES_DIR


def plot_head_patching(npz_path: str, out_path: str, title: str = "") -> None:
    data = np.load(npz_path)
    head_out = data["head_out"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        head_out,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-0.2,
        vmax=1.0,
        xticklabels=range(head_out.shape[1]),
        yticklabels=range(head_out.shape[0]),
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title or f"Activation Patching — {npz_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_dla(npz_path: str, out_path: str, top_k: int = 15, title: str = "") -> None:
    data = np.load(npz_path)
    head_dla = data["head_dla"]

    flat = head_dla.flatten()
    top_idx = np.argsort(np.abs(flat))[::-1][:top_k]
    labels = [f"L{i // head_dla.shape[1]}H{i % head_dla.shape[1]}" for i in top_idx]
    values = flat[top_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["steelblue" if v > 0 else "tomato" for v in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("DLA")
    ax.set_title(title or f"Direct Logit Attribution — top {top_k} heads")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_pca_scatter(npz_path: str, out_path: str) -> None:
    data = np.load(npz_path, allow_pickle=True)
    projections = data["projections"]
    labels = data["labels"]
    langs = data["langs"]

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"en": "o", "es": "^"}
    colors = {0: "steelblue", 1: "tomato"}

    for lang in ["en", "es"]:
        for label, name in [(0, "singular"), (1, "plural")]:
            mask = (langs == lang) & (labels == label)
            ax.scatter(
                np.where(mask)[0],
                projections[mask],
                marker=markers[lang],
                color=colors[label],
                alpha=0.5,
                label=f"{lang} {name}",
                s=20,
            )

    ax.set_ylabel("PC1 projection")
    ax.set_xlabel("Example index")
    ax.set_title("Subject-number direction (PC1 of L13H7 output)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_steering(npz_path: str, out_path: str) -> None:
    data = np.load(npz_path)
    alphas = data["alphas"]
    flip_pos = data["flip_rate_pos"]
    flip_neg = data["flip_rate_neg"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, flip_pos, "o-", label="+ α·PC1 (push plural)")
    ax.plot(alphas, flip_neg, "s-", label="- α·PC1 (push singular)")
    ax.set_xlabel("α (steering magnitude)")
    ax.set_ylabel("Flip rate")
    ax.set_title("Cross-lingual steering (EN direction → ES predictions)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_weight_importance_heatmap(npz_path: str, out_path: str, title: str = "") -> None:
    """Heatmap of head_importance from circuit_map results."""
    data = np.load(npz_path)
    head_imp = data["head_importance"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        head_imp, ax=ax, cmap="YlOrRd",
        xticklabels=range(head_imp.shape[1]),
        yticklabels=range(head_imp.shape[0]),
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title or f"Weight Importance — {npz_path}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_svd_spectrum(npz_path: str, out_path: str, layer: int = 13, head: int = 7) -> None:
    """Bar chart of top-k singular values for a given head's OV matrix."""
    data = np.load(npz_path)
    spectra = data["svd_spectra"]
    n_layers = spectra.shape[0]
    layer = min(layer, n_layers - 1)
    head = min(head, spectra.shape[1] - 1)
    sv = spectra[layer, head]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(sv)), sv, color="steelblue")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"SVD spectrum — L{layer}H{head} OV matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_convergence_curve(geometry_npz_path: str, out_path: str) -> None:
    """Line plot: X=layer, Y=mean cross-lingual CKA similarity."""
    data = np.load(geometry_npz_path, allow_pickle=True)
    convergence = data["convergence"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(convergence)), convergence, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean CKA")
    ax.set_title("Cross-lingual convergence across layers")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_logit_lens(npz_path: str, out_path: str, title: str = "") -> None:
    """Line plot of mean logit diff and P(correct) across layers."""
    data = np.load(npz_path)
    mean_ld = data["mean_logit_diff"]
    mean_prob = data["mean_correct_prob"]
    layers = list(range(len(mean_ld)))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(layers, mean_ld, "o-", color="steelblue", label="Logit diff")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean logit diff", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(layers, mean_prob, "s-", color="tomato", label="P(correct)")
    ax2.set_ylabel("P(correct verb)", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")

    ax1.set_title(title or "Logit Lens — prediction formation by layer")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_attention_subject(npz_path: str, out_path: str, title: str = "") -> None:
    """Heatmap of subject attention scores per head."""
    data = np.load(npz_path)
    subject_attn = data["subject_attention"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(subject_attn, ax=ax, cmap="YlOrRd",
                xticklabels=range(subject_attn.shape[1]),
                yticklabels=range(subject_attn.shape[0]))
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title or "Subject attention from verb position")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_knockout_summary(knockout_data: dict, out_path: str) -> None:
    """Bar chart comparing baseline, necessity, and sufficiency across languages."""
    langs = sorted(knockout_data.keys())
    baseline = [knockout_data[l]["baseline"] for l in langs]
    necessity = [knockout_data[l]["necessity"] for l in langs]
    sufficiency = [knockout_data[l]["sufficiency"] for l in langs]

    x = np.arange(len(langs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, baseline, width, label="Baseline", color="steelblue")
    ax.bar(x, necessity, width, label="Ablate circuit", color="tomato")
    ax.bar(x + width, sufficiency, width, label="Ablate complement", color="seagreen")

    ax.set_ylabel("Accuracy")
    ax.set_title("Circuit Knockout — Necessity and Sufficiency")
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs])
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_eap_comparison(npz_paths: dict, out_path: str, top_k: int = 15) -> None:
    """Side-by-side bar chart of top EAP scores for multiple languages."""
    fig, axes = plt.subplots(1, len(npz_paths), figsize=(6 * len(npz_paths), 5), sharey=True)
    if len(npz_paths) == 1:
        axes = [axes]

    for ax, (lang, path) in zip(axes, npz_paths.items()):
        data = np.load(path, allow_pickle=True)
        scores = data["node_scores"]
        labels = data["component_labels"]
        top_idx = np.argsort(scores)[::-1][:top_k]

        name = LANG_CONFIGS.get(lang, {}).get("name", lang.upper())
        ax.barh(
            [str(labels[i]) for i in top_idx][::-1],
            scores[top_idx][::-1],
            color="steelblue",
        )
        ax.set_title(f"EAP — {name}")
        ax.set_xlabel("Score")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    r = Path(args.results_dir)
    o = Path(args.out_dir)
    o.mkdir(parents=True, exist_ok=True)

    # Original figures
    if (r / "patching_en.npz").exists():
        plot_head_patching(r / "patching_en.npz", o / "fig1_patching_en.png",
                           title="Activation Patching — English SVA, per head")
    if (r / "patching_es.npz").exists():
        plot_head_patching(r / "patching_es.npz", o / "fig2_patching_es.png",
                           title="Activation Patching — Spanish SVA, per head")
    if (r / "dla_en.npz").exists():
        plot_dla(r / "dla_en.npz", o / "fig3_dla_en.png",
                 title="DLA — English SVA")
    if (r / "dla_es.npz").exists():
        plot_dla(r / "dla_es.npz", o / "fig4_dla_es.png",
                 title="DLA — Spanish SVA")
    if (r / "pca_L13H7.npz").exists():
        plot_pca_scatter(r / "pca_L13H7.npz", o / "fig5_pca_scatter.png")
    if (r / "steering.npz").exists():
        plot_steering(r / "steering.npz", o / "fig6_steering.png")

    # All languages patching/DLA
    for lang in ALL_LANGS:
        if (r / f"patching_{lang}.npz").exists():
            name = LANG_CONFIGS.get(lang, {}).get("name", lang.upper())
            plot_head_patching(r / f"patching_{lang}.npz",
                               o / f"fig_patching_{lang}.png",
                               title=f"Activation Patching — {name} SVA")
        if (r / f"dla_{lang}.npz").exists():
            name = LANG_CONFIGS.get(lang, {}).get("name", lang.upper())
            plot_dla(r / f"dla_{lang}.npz", o / f"fig_dla_{lang}.png",
                     title=f"DLA — {name} SVA")

    # Steering for all target languages
    for lang in ALL_LANGS:
        if (r / f"steering_{lang}.npz").exists():
            name = LANG_CONFIGS.get(lang, {}).get("name", lang.upper())
            plot_steering(str(r / f"steering_{lang}.npz"),
                          str(o / f"fig_steering_{lang}.png"))

    # Weight importance heatmaps
    for lang in ALL_LANGS:
        cm_path = r / f"circuit_map_{lang}.npz"
        if cm_path.exists():
            name = LANG_CONFIGS.get(lang, {}).get("name", lang.upper())
            plot_weight_importance_heatmap(
                str(cm_path), str(o / f"fig_weight_importance_{lang}.png"),
                title=f"Weight Importance — {name}",
            )
            plot_svd_spectrum(
                str(cm_path), str(o / f"fig_svd_spectrum_{lang}.png"),
            )

    # EAP comparison
    eap_paths = {}
    for lang in ALL_LANGS:
        ep_path = r / f"edge_patching_{lang}.npz"
        if ep_path.exists():
            eap_paths[lang] = str(ep_path)
    if eap_paths:
        plot_eap_comparison(eap_paths, str(o / "fig_eap_comparison.png"))

    # Convergence curve
    if (r / "geometry.npz").exists():
        plot_convergence_curve(str(r / "geometry.npz"), str(o / "fig_convergence.png"))

    # Logit lens per language
    for lang in ALL_LANGS:
        ll_path = r / f"logit_lens_{lang}.npz"
        if ll_path.exists():
            name = LANG_CONFIGS.get(lang, {}).get("name", lang.upper())
            plot_logit_lens(str(ll_path), str(o / f"fig_logit_lens_{lang}.png"),
                            title=f"Logit Lens — {name}")

    # Attention subject scores per language
    for lang in ALL_LANGS:
        attn_path = r / f"attention_{lang}.npz"
        if attn_path.exists():
            name = LANG_CONFIGS.get(lang, {}).get("name", lang.upper())
            plot_attention_subject(str(attn_path), str(o / f"fig_attention_{lang}.png"),
                                   title=f"Subject Attention — {name}")

    # Knockout summary
    knockout_data = {}
    for lang in ALL_LANGS:
        ko_path = r / f"knockout_{lang}.npz"
        if ko_path.exists():
            ko = np.load(ko_path)
            knockout_data[lang] = {
                "baseline": float(ko["baseline_accuracy"]),
                "necessity": float(ko["necessity_accuracy"]),
                "sufficiency": float(ko["sufficiency_accuracy"]),
            }
    if knockout_data:
        plot_knockout_summary(knockout_data, str(o / "fig_knockout_summary.png"))


if __name__ == "__main__":
    main()
