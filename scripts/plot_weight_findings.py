"""
Figures for the weight-level findings.

Main claim: the SVA circuit decomposes into a UNIVERSAL SUBSTRATE (early L0
heads, rank-1, near-identical task direction across languages) and a
LANGUAGE-SPECIFIC COMPUTATION HEAD (L13H7, high-rank, low cross-lingual
cosine at the weight level).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"

GEMMA_LANGS = ["en", "es", "fr", "ru", "sw", "tr", "qu"]
LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "ru": "Russian",
              "sw": "Swahili", "tr": "Turkish", "qu": "Quechua"}


def fig_universal_task_direction():
    """Heatmap of mean cross-lingual cosine similarity per head."""
    d = np.load(RES / "universal_task_direction.npz")
    M = d["mean_cos"]  # (18, 8)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(8)); ax.set_yticks(range(18))
    ax.set_title("Weight-level universality — mean cosine sim of task direction across 7 languages\n"
                 "GREEN = universal task direction; RED = language-specific",
                 fontsize=11)

    # Annotate strongly — label L0H3, L0H5, L13H7, L17H4 explicitly
    for (L, H, note) in [(0, 3, "UNIVERSAL\n(cos 1.00)"),
                         (0, 5, "UNIVERSAL\n(cos 0.99)"),
                         (13, 7, "paper's\nkey head\n(cos 0.35)"),
                         (17, 4, "2nd head\n(cos 0.26)")]:
        ax.annotate(note,
                    xy=(H, L),
                    xytext=(H + 1.2 if H < 6 else H - 2.5, L),
                    fontsize=9,
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # Colored text for all cells
    for L in range(M.shape[0]):
        for H in range(M.shape[1]):
            ax.text(H, L, f"{M[L, H]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if M[L, H] < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="mean cross-lingual cosine")
    fig.savefig(FIG / "fig_universal_task_direction.png")
    plt.close(fig)
    print("  ✓ fig_universal_task_direction.png")


def fig_rank_vs_universality():
    """Scatter: rank-1 fraction vs cross-lingual cosine.
    Shows the clean clustering: L0 heads = universal + rank-1;
    L13H7 etc = specific + high-rank."""
    d = np.load(RES / "universal_task_direction.npz")
    svd_d = np.load(RES / "svd_summary.npz")
    cos_M = d["mean_cos"]  # (18, 8)
    rank_M = svd_d["top1_frac_per_lang"].mean(axis=0)  # (18, 8) avg across langs

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    layers = np.arange(18)
    # Color by layer (early = blue, late = red)
    for L in range(18):
        for H in range(8):
            color = plt.cm.coolwarm(L / 17)
            ax.scatter(cos_M[L, H], rank_M[L, H], color=color, s=40, alpha=0.7,
                       edgecolor="black", linewidth=0.3)

    # Annotate key heads
    highlights = [
        ("L0H3", 0, 3, "top-right:\nuniversal + rank-1"),
        ("L0H5", 0, 5, ""),
        ("L13H7", 13, 7, "bottom-left:\nspecific + high-rank"),
        ("L17H4", 17, 4, ""),
    ]
    for lbl, L, H, note in highlights:
        ax.annotate(f"{lbl}\n{note}".strip(),
                    xy=(cos_M[L, H], rank_M[L, H]),
                    xytext=(cos_M[L, H] + 0.05, rank_M[L, H] + 0.08 if L < 5 else rank_M[L, H] - 0.15),
                    fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1))

    ax.set_xlabel("Cross-lingual cosine similarity of task direction")
    ax.set_ylabel("Rank-1 fraction (top-1 singular value² / total)")
    ax.set_title("Two kinds of SVA-adjacent heads: universal + simple vs. language-specific + complex",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)

    sm = mpl.cm.ScalarMappable(cmap="coolwarm", norm=mpl.colors.Normalize(vmin=0, vmax=17))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Layer index (early=blue, late=red)")

    fig.savefig(FIG / "fig_rank_vs_universality.png")
    plt.close(fig)
    print("  ✓ fig_rank_vs_universality.png")


def fig_universal_neurons():
    """Per-language importance for key neurons: n2069 (L13) and n13557 (L17).
    Shows Indo-European-universal neuron vs. low-resource neuron."""
    maps = {l: np.load(RES / f"circuit_map_{l}.npz") for l in GEMMA_LANGS}

    n13_2069 = [float(maps[l]["mlp_neuron_importance"][13, 2069]) for l in GEMMA_LANGS]
    n17_13557 = [float(maps[l]["mlp_neuron_importance"][17, 13557]) for l in GEMMA_LANGS]
    n17_1138 = [float(maps[l]["mlp_neuron_importance"][17, 1138]) for l in GEMMA_LANGS]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True, sharey=True)

    for ax, title, vals, highlight in zip(
        axes,
        ["L13 n2069 — paper's 'universal' neuron\n(actually Indo-European only)",
         "L17 n1138 — paper's claim\n(actually Spanish-specific)",
         "L17 n13557 — our finding\n(low-resource / agglutinative!)"],
        [n13_2069, n17_1138, n17_13557],
        [{"en", "es", "fr", "ru"}, {"es"}, {"sw", "qu"}],
    ):
        colors = ["#d62728" if l in highlight else "#7f7f7f" for l in GEMMA_LANGS]
        ax.bar(GEMMA_LANGS, vals, color=colors, edgecolor="black")
        ax.set_xticklabels([LANG_NAMES[l] for l in GEMMA_LANGS], rotation=15, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Weight-level neuron importance")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Universal MLP neurons are NOT actually universal — they cluster by language typology",
                 fontsize=11)
    fig.savefig(FIG / "fig_universal_neurons.png")
    plt.close(fig)
    print("  ✓ fig_universal_neurons.png")


def fig_universal_connections():
    """Strongest head→head connections, averaged across languages.
    Shows L14 → L16 dominates, not the paper's L13 → L17."""
    maps = {l: np.load(RES / f"circuit_map_{l}.npz") for l in GEMMA_LANGS}
    all_conn = np.stack([maps[l]["connection_matrix"] for l in GEMMA_LANGS])
    mean_conn = all_conn.mean(axis=0)
    std_conn = all_conn.std(axis=0)
    labels = maps["en"]["connection_labels"]

    # Rank edges by mean strength
    flat = mean_conn.flatten()
    top = np.argsort(-flat)[:20]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    edges = []
    for i in top:
        si, di = i // 144, i % 144
        edges.append({
            "src": labels[si], "dst": labels[di],
            "mean": float(mean_conn[si, di]), "std": float(std_conn[si, di]),
        })

    y = np.arange(len(edges))
    vals = [e["mean"] for e in edges]
    errs = [e["std"] for e in edges]
    lbls = [f"{e['src']} → {e['dst']}" for e in edges]

    ax.barh(y, vals, xerr=errs, color="#1f77b4", edgecolor="black", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean connection strength across 7 langs (± std)")
    ax.set_title("Top-20 universal head→head connections — dominated by L14→L16 (NOT L13→L17)",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(FIG / "fig_universal_connections.png")
    plt.close(fig)
    print("  ✓ fig_universal_connections.png")


def main():
    print("Weight-level figures:")
    fig_universal_task_direction()
    fig_rank_vs_universality()
    fig_universal_neurons()
    fig_universal_connections()
    print("\nDone.")


if __name__ == "__main__":
    main()
