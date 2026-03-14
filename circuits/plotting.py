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

from circuits.config import RESULTS_DIR, FIGURES_DIR


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    r = Path(args.results_dir)
    o = Path(args.out_dir)
    o.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
