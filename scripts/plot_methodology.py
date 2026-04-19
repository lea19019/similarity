"""
Figures about the methodological finding: weight-alignment interpretability
misleads you for early layers.

Key plots:
  1. EAP rank vs weight-importance rank per head — shows low correlation
     and the L0 dominance in weight-importance.
  2. Within-language head-pair cosine heatmap showing L0 heads cluster as
     identical operations (i.e., token reconstruction, not SVA circuitry).
  3. Causal (EAP) top-5 per language overlaid on weight-importance top-5.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"

GEMMA_LANGS = ["en", "es", "fr", "ru", "sw", "tr", "qu"]
LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "ru": "Russian",
              "sw": "Swahili", "tr": "Turkish", "qu": "Quechua"}


def fig_eap_vs_weight():
    """EAP rank vs weight-importance rank — shows the two methods disagree."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.flatten()

    correlations = []
    for ax, lang in zip(axes, GEMMA_LANGS):
        cm = np.load(RES / f"circuit_map_{lang}.npz")
        eap = np.load(RES / f"edge_patching_{lang}.npz")
        wi = cm["head_importance"].flatten()  # 144
        ei = eap["head_scores"].flatten()

        # Normalize each to [0, 1] for display
        wi_n = wi / (wi.max() + 1e-9)
        ei_n = ei / (ei.max() + 1e-9)

        # Scatter — color L0 heads red, L13H7 green
        colors = []
        for i in range(144):
            L, H = i // 8, i % 8
            if L == 0:
                colors.append("#d62728")  # L0 in red
            elif (L, H) == (13, 7):
                colors.append("#2ca02c")  # L13H7 in green
            elif (L, H) == (17, 4):
                colors.append("#ff7f0e")  # L17H4 in orange
            else:
                colors.append("#9e9e9e")

        ax.scatter(ei_n, wi_n, c=colors, alpha=0.7, s=30, edgecolor="black", linewidth=0.3)

        # Diagonal line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)

        # Label key points
        for L, H, lbl in [(0, 5, "L0H5"), (0, 3, "L0H3"), (13, 7, "L13H7"),
                          (17, 4, "L17H4")]:
            i = L * 8 + H
            ax.annotate(lbl, (ei_n[i], wi_n[i]), fontsize=8,
                        xytext=(3, 3), textcoords="offset points")

        r, p = spearmanr(wi, ei)
        correlations.append(r)
        ax.set_title(f"{LANG_NAMES[lang]}  (ρ={r:.2f})", fontsize=10)
        ax.set_xlabel("EAP (causal) — normalized")
        ax.set_ylabel("Weight importance — normalized")
        ax.grid(alpha=0.3)

    axes[-1].axis("off")
    axes[-1].text(0.5, 0.5,
                  "If the two methods agreed,\npoints would lie on the dashed\ndiagonal.\n\n"
                  f"Mean Spearman ρ = {np.mean(correlations):.2f}\n"
                  "The two methods largely\nDISAGREE.\n\n"
                  "Weight importance ranks L0H5/L0H3 first;\n"
                  "EAP puts them near the bottom.\n\n"
                  "This is the methodological\nfinding: weight-alignment tools\n"
                  "are fooled by embedding geometry\nin early layers.",
                  fontsize=10, ha="center", va="center",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.suptitle("Causal (EAP) vs structural (weight-importance) head rankings — they DISAGREE\n"
                 "L0 heads (red) score high structurally but are causally inert",
                 fontsize=12)
    fig.savefig(FIG / "fig_eap_vs_weight.png")
    plt.close(fig)
    print("  ✓ fig_eap_vs_weight.png")


def fig_l0_redundancy():
    """Within-language cosine similarity between head task-directions.
    Shows L0 heads form a tight cluster (all doing the same thing);
    L13H7 is isolated (unique computation)."""
    cm = np.load(RES / "circuit_map_en.npz")
    tw = cm["head_task_weights"]  # (18, 8, 2048)
    flat = tw.reshape(-1, tw.shape[-1])  # (144, 2048)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9
    normed = flat / norms
    cos = normed @ normed.T  # (144, 144)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    im = ax.imshow(np.abs(cos), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("Head (L0H0, L0H1, …, L17H7)")
    ax.set_ylabel("Head (L0H0, L0H1, …, L17H7)")
    ax.set_title("Pairwise cosine of head task-directions within English\n"
                 "Early-layer heads form a redundant cluster — doing the same operation",
                 fontsize=11)

    # Layer boundaries
    for L in range(1, 18):
        ax.axhline(L * 8 - 0.5, color="white", linewidth=0.3, alpha=0.3)
        ax.axvline(L * 8 - 0.5, color="white", linewidth=0.3, alpha=0.3)

    # Mark L0 cluster
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 8, 8, fill=False,
                                edgecolor="red", linewidth=2))
    ax.annotate("L0 cluster:\nall heads\nredundant",
                xy=(4, 4), xytext=(15, 10), color="white", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1))

    # Mark L13H7
    ax.axhline(13 * 8 + 7, color="#2ca02c", linewidth=0.8, alpha=0.7)
    ax.axvline(13 * 8 + 7, color="#2ca02c", linewidth=0.8, alpha=0.7)
    ax.annotate("L13H7:\nisolated\n(unique)",
                xy=(13 * 8 + 7, 13 * 8 + 7), xytext=(13 * 8 + 15, 13 * 8 - 10),
                color="white", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1))

    plt.colorbar(im, ax=ax, label="|cosine similarity|")
    fig.savefig(FIG / "fig_l0_redundancy.png")
    plt.close(fig)
    print("  ✓ fig_l0_redundancy.png")


def fig_eap_top5():
    """Horizontal bars showing the actual top-5 EAP components per language.
    This is the TRUSTWORTHY causal picture."""
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    y = 0
    for i, lang in enumerate(GEMMA_LANGS):
        d = np.load(RES / f"edge_patching_{lang}.npz")
        scores = d["node_scores"]; labels = d["component_labels"]
        idx = np.argsort(-scores)[:5]

        # Plot 5 bars stacked vertically
        for j, ii in enumerate(idx):
            lbl = str(labels[ii])
            color = "#d62728" if lbl.startswith("MLP") else "#1f77b4"
            ax.barh(y, float(scores[ii]), color=color, edgecolor="black", linewidth=0.5)
            ax.text(float(scores[ii]) + 1, y, lbl, va="center", fontsize=8)
            y += 1
        # Separator
        ax.axhline(y - 0.5, color="gray", alpha=0.3)
        ax.text(-5, y - 3, LANG_NAMES[lang], fontsize=11, fontweight="bold",
                ha="right", va="center")
        y += 0.5

    ax.set_xlabel("EAP score (causal importance, gradient-based)")
    ax.set_yticks([])
    ax.set_xlim(-30, 130)
    ax.set_title("Top-5 components by EAP per language — MLP17 dominates everywhere",
                 fontsize=12)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#d62728", label="MLP"),
        Patch(facecolor="#1f77b4", label="Attention head"),
    ], loc="upper right")
    fig.savefig(FIG / "fig_eap_top5.png")
    plt.close(fig)
    print("  ✓ fig_eap_top5.png")


def main():
    print("Methodological figures:")
    fig_eap_vs_weight()
    fig_l0_redundancy()
    fig_eap_top5()
    print("\nDone.")


if __name__ == "__main__":
    main()
