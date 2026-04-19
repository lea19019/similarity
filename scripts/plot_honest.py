"""
Regenerate the figure set directly from raw NPZ results.

Produces a small set of comparison-focused figures that accurately tell the
gradient-universality story grounded in AUDIT.txt / HONEST_FINDINGS.md.

Run:
    uv run python scripts/plot_honest.py

Output: results/figures/*.png
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BLOOM = RES / "bloom-3b"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

GEMMA_LANGS = ["en", "es", "fr", "ru", "sw", "tr", "qu"]
LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "ru": "Russian",
              "sw": "Swahili", "tr": "Turkish", "qu": "Quechua"}
BLOOM_LANGS = ["en", "es", "fr", "sw"]

# Palette — consistent across all figures
LANG_COLOR = {
    "en": "#1f77b4",  # blue
    "es": "#ff7f0e",  # orange
    "fr": "#2ca02c",  # green
    "ru": "#d62728",  # red
    "sw": "#9467bd",  # purple
    "tr": "#8c564b",  # brown
    "qu": "#e377c2",  # pink
}


def load(path):
    return np.load(path, allow_pickle=True)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Patching heatmaps (all 7 langs, unified scale)
# ═══════════════════════════════════════════════════════════════════════════

def fig_patching_grid():
    """One heatmap per language, shared colorscale so magnitudes compare."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    axes = axes.flatten()

    # Determine global vmax so all heatmaps share scale
    mats = {}
    for lang in GEMMA_LANGS:
        d = load(RES / f"patching_{lang}.npz")
        mats[lang] = d["head_out"]
    global_vmax = max(np.abs(m).max() for m in mats.values())

    for ax, lang in zip(axes, GEMMA_LANGS):
        m = mats[lang]
        im = ax.imshow(m, cmap="RdBu_r", vmin=-global_vmax, vmax=global_vmax, aspect="auto")
        ax.set_title(f"{LANG_NAMES[lang]}  (peak {m.max():.2f} @ "
                     f"L{int(np.unravel_index(m.argmax(), m.shape)[0])}H"
                     f"{int(np.unravel_index(m.argmax(), m.shape)[1])})",
                     fontsize=10)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(m.shape[1]))
        ax.set_yticks(range(0, m.shape[0], 2))

    # Hide 8th panel
    axes[-1].axis("off")
    # Add a single colorbar
    cbar = fig.colorbar(im, ax=axes[-1], shrink=0.8, location="left", pad=0.0)
    cbar.set_label("Patching effect\n(normalized recovery)", fontsize=9)
    fig.suptitle("Activation patching — same model (Gemma 2B), 7 languages, shared colorscale",
                 fontsize=12)
    fig.savefig(FIG / "fig_patching_grid.png")
    plt.close(fig)
    print("  ✓ fig_patching_grid.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Top-1 patching head per language
# ═══════════════════════════════════════════════════════════════════════════

def fig_patching_top1():
    """Bar chart: top-1 patching effect per language, color-coded by layer."""
    tops = []
    for lang in GEMMA_LANGS:
        d = load(RES / f"patching_{lang}.npz")
        m = d["head_out"]
        peak = m.max()
        L, H = np.unravel_index(m.argmax(), m.shape)
        tops.append((lang, int(L), int(H), float(peak)))

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x = np.arange(len(tops))
    values = [t[3] for t in tops]
    # Color by layer
    cmap = plt.cm.viridis
    colors = [cmap(t[1] / 17) for t in tops]

    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    for i, (lang, L, H, v) in enumerate(tops):
        ax.text(i, v + 0.01, f"L{L}H{H}\n{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[t[0]] for t in tops])
    ax.set_ylabel("Patching effect (top head)")
    ax.set_title("Top-1 head per language — same head? Mostly L13, but not always L13H7\n"
                 "(color = layer index; darker = earlier)",
                 fontsize=11)
    ax.set_ylim(0, max(values) * 1.2)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=17))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Layer index")
    fig.savefig(FIG / "fig_patching_top1.png")
    plt.close(fig)
    print("  ✓ fig_patching_top1.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Steering peak flip rate per target
# ═══════════════════════════════════════════════════════════════════════════

def fig_steering_peaks_gemma():
    """RQ1 figure — Gemma only, 6 target languages."""
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)

    targets = ["es", "fr", "ru", "sw", "tr", "qu"]
    peaks = []
    for lang in targets:
        d = load(RES / f"steering_{lang}.npz")
        p = max(d["flip_rate_pos"].max(), d["flip_rate_neg"].max())
        peaks.append(float(p))

    # Colour-code by strength: strong > 0.5 green, moderate 0.3-0.5 amber, weak < 0.3 red
    colors = ["#2ca02c" if v >= 0.5 else "#ff7f0e" if v >= 0.3 else "#d62728"
              for v in peaks]

    x = np.arange(len(targets))
    ax.bar(x, peaks, color=colors, edgecolor="black")
    for i, v in enumerate(peaks):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[t] for t in targets])
    ax.set_ylabel("Peak flip rate (max across α)")
    ax.set_ylim(0, 0.8)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.set_title("Cross-lingual steering in Gemma 2B: English direction → target language\n"
                 "(strong for resource-rich Indo-European + Swahili; degrades for low-resource agglutinative)",
                 fontsize=11)
    fig.savefig(FIG / "fig_steering_peaks_gemma.png")
    plt.close(fig)
    print("  ✓ fig_steering_peaks_gemma.png")


def fig_steering_peaks_crossmodel():
    """RQ2 figure — Gemma vs BLOOM side-by-side, for languages in both."""
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)

    targets = ["es", "fr", "sw"]  # only languages BLOOM has
    gemma_peaks = []
    bloom_peaks = []
    for lang in targets:
        gd = load(RES / f"steering_{lang}.npz")
        bd = load(BLOOM / f"steering_{lang}.npz")
        gemma_peaks.append(float(max(gd["flip_rate_pos"].max(), gd["flip_rate_neg"].max())))
        bloom_peaks.append(float(max(bd["flip_rate_pos"].max(), bd["flip_rate_neg"].max())))

    x = np.arange(len(targets))
    w = 0.36
    ax.bar(x - w/2, gemma_peaks, w, label="Gemma 2B", color="#1f77b4", edgecolor="black")
    ax.bar(x + w/2, bloom_peaks, w, label="BLOOM 3B", color="#ff7f0e", edgecolor="black")

    for i, (g, b) in enumerate(zip(gemma_peaks, bloom_peaks)):
        ax.text(i - w/2, g + 0.01, f"{g:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[t] for t in targets])
    ax.set_ylabel("Peak flip rate (max across α)")
    ax.set_ylim(0, 0.95)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.set_title("Cross-model steering: same EN-derived direction, applied per model\n"
                 "(both models transfer to ES/FR; BLOOM × Swahili fails — a clear universality boundary)",
                 fontsize=11)
    ax.legend()
    fig.savefig(FIG / "fig_steering_peaks_crossmodel.png")
    plt.close(fig)
    print("  ✓ fig_steering_peaks_crossmodel.png")


def fig_steering_peaks():
    """Keep the combined figure as a deprecated alias so other scripts still work,
    but prefer the two split figures above."""
    fig_steering_peaks_gemma()
    fig_steering_peaks_crossmodel()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Steering curves small multiples
# ═══════════════════════════════════════════════════════════════════════════

def fig_steering_curves():
    """6 steering curves small multiples (Gemma targets only)."""
    targets = ["es", "fr", "ru", "sw", "tr", "qu"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True, sharey=True)
    for ax, lang in zip(axes.flatten(), targets):
        d = load(RES / f"steering_{lang}.npz")
        ax.plot(d["alphas"], d["flip_rate_pos"], "-o", color="#d62728", label="+PC1", markersize=5)
        ax.plot(d["alphas"], d["flip_rate_neg"], "-s", color="#1f77b4", label="−PC1", markersize=5)
        ax.set_title(LANG_NAMES[lang])
        ax.set_xlabel("α")
        ax.set_ylim(0, 0.8)
        ax.grid(alpha=0.3)
        if ax is axes[0, 0]:
            ax.legend(fontsize=9)
    axes[0, 0].set_ylabel("Flip rate")
    axes[1, 0].set_ylabel("Flip rate")
    fig.suptitle("Steering curves — EN direction applied to L13H7, sweep α in Gemma 2B\n"
                 "(non-monotonic — at high α we begin destroying the activation)",
                 fontsize=11)
    fig.savefig(FIG / "fig_steering_curves.png")
    plt.close(fig)
    print("  ✓ fig_steering_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — EAP MLP dominance
# ═══════════════════════════════════════════════════════════════════════════

def fig_eap_mlp():
    """Top-5 EAP components per language — highlight MLP dominance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Gemma (all 7)
    ax = axes[0]
    top5_by_lang = []
    for lang in GEMMA_LANGS:
        d = load(RES / f"edge_patching_{lang}.npz")
        scores = d["node_scores"]; labels = d["component_labels"]
        idx = np.argsort(-scores)[:5]
        top5_by_lang.append([(str(labels[i]), float(scores[i])) for i in idx])

    # Plot as grouped bars per language
    x = np.arange(len(GEMMA_LANGS))
    max_vals = [t[0][1] for t in top5_by_lang]  # top-1 value per lang
    second_vals = [t[1][1] for t in top5_by_lang]

    w = 0.35
    ax.bar(x - w/2, max_vals, w, color="#d62728", edgecolor="black", label="Top-1 component")
    ax.bar(x + w/2, second_vals, w, color="#1f77b4", edgecolor="black", label="Top-2 component")
    for i, t in enumerate(top5_by_lang):
        ax.text(i - w/2, t[0][1] + 1, t[0][0], ha="center", fontsize=8, rotation=0)
        ax.text(i + w/2, t[1][1] + 1, t[1][0], ha="center", fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in GEMMA_LANGS], rotation=15, ha="right")
    ax.set_ylabel("EAP score")
    ax.set_title("Gemma 2B — MLP17 is top in all 7 languages", fontsize=11)
    ax.legend()

    # BLOOM (4 langs)
    ax = axes[1]
    top5_bloom = []
    for lang in BLOOM_LANGS:
        d = load(BLOOM / f"edge_patching_{lang}.npz")
        scores = d["node_scores"]; labels = d["component_labels"]
        idx = np.argsort(-scores)[:5]
        top5_bloom.append([(str(labels[i]), float(scores[i])) for i in idx])

    x = np.arange(len(BLOOM_LANGS))
    max_vals = [t[0][1] for t in top5_bloom]
    second_vals = [t[1][1] for t in top5_bloom]
    ax.bar(x - w/2, max_vals, w, color="#d62728", edgecolor="black", label="Top-1")
    ax.bar(x + w/2, second_vals, w, color="#1f77b4", edgecolor="black", label="Top-2")
    for i, t in enumerate(top5_bloom):
        ax.text(i - w/2, t[0][1] + 0.3, t[0][0], ha="center", fontsize=8)
        ax.text(i + w/2, t[1][1] + 0.3, t[1][0], ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in BLOOM_LANGS], rotation=15, ha="right")
    ax.set_ylabel("EAP score")
    ax.set_title("BLOOM 3B — MLP28 dominates", fontsize=11)
    ax.legend()

    fig.suptitle("Edge attribution patching: late-layer MLPs dominate in every language, both models",
                 fontsize=12)
    fig.savefig(FIG / "fig_eap_mlp.png")
    plt.close(fig)
    print("  ✓ fig_eap_mlp.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Subject-attention vs patching divergence
# ═══════════════════════════════════════════════════════════════════════════

def fig_subject_vs_number():
    """Show L13H7's role differs by language — causal effect varies
    (including negative in French and Quechua!), and subject attention varies."""
    l13h7_patch = []
    l13h7_sa = []
    top_sa_head = []
    for lang in GEMMA_LANGS:
        p = load(RES / f"patching_{lang}.npz")["head_out"]
        a = load(RES / f"attention_{lang}.npz")["subject_attention"]
        l13h7_patch.append(float(p[13, 7]))
        l13h7_sa.append(float(a[13, 7]))
        L, H = np.unravel_index(a.argmax(), a.shape)
        top_sa_head.append((int(L), int(H), float(a[L, H])))

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    x = np.arange(len(GEMMA_LANGS))
    w = 0.35
    bars_red = ax.bar(x - w/2, l13h7_patch, w, color="#d62728", edgecolor="black",
                      label="L13H7 causal effect (patching)")
    bars_blue = ax.bar(x + w/2, l13h7_sa, w, color="#1f77b4", edgecolor="black",
                       label="L13H7 attention to subject")

    # Value labels on each bar
    for i, v in enumerate(l13h7_patch):
        offset = 0.02 if v >= 0 else -0.04
        ax.text(i - w/2, v + offset, f"{v:+.2f}", ha="center", fontsize=8,
                color="black")
    for i, v in enumerate(l13h7_sa):
        ax.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8,
                color="black")

    # Label top-SA head above each bar group
    for i, (L, H, v) in enumerate(top_sa_head):
        ax.text(i, 0.80,
                f"top SA: L{L}H{H}", ha="center", fontsize=8, style="italic")

    # Zero line + shaded negative region
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhspan(-0.25, 0, alpha=0.05, color="red", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in GEMMA_LANGS], rotation=15, ha="right")
    ax.set_ylabel("Score (patching signed; subject attention ∈ [0, 1])")
    ax.set_title("L13H7's role varies by language — in French & Quechua it actually HURTS SVA\n"
                 "English uses L13H7 to read the subject; other languages delegate that role",
                 fontsize=11)
    ax.legend(loc="upper right")
    ax.set_ylim(-0.25, 0.85)
    fig.savefig(FIG / "fig_subject_vs_number.png")
    plt.close(fig)
    print("  ✓ fig_subject_vs_number.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Cross-model key head relative depth
# ═══════════════════════════════════════════════════════════════════════════

def fig_cross_model_depth():
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    pairs = []
    for lang in BLOOM_LANGS:
        pg = load(RES / f"patching_{lang}.npz")["head_out"]
        pb = load(BLOOM / f"patching_{lang}.npz")["head_out"]
        Lg, Hg = np.unravel_index(pg.argmax(), pg.shape)
        Lb, Hb = np.unravel_index(pb.argmax(), pb.shape)
        pairs.append({
            "lang": lang,
            "gemma_depth": Lg / (pg.shape[0] - 1),
            "bloom_depth": Lb / (pb.shape[0] - 1),
            "gemma_label": f"L{Lg}H{Hg}",
            "bloom_label": f"L{Lb}H{Hb}",
        })

    x = np.arange(len(pairs))
    w = 0.35
    ax.bar(x - w/2, [p["gemma_depth"] * 100 for p in pairs], w,
           color="#1f77b4", edgecolor="black", label="Gemma 2B (18 layers)")
    ax.bar(x + w/2, [p["bloom_depth"] * 100 for p in pairs], w,
           color="#ff7f0e", edgecolor="black", label="BLOOM 3B (30 layers)")
    for i, p in enumerate(pairs):
        ax.text(i - w/2, p["gemma_depth"] * 100 + 2, p["gemma_label"],
                ha="center", fontsize=9)
        ax.text(i + w/2, p["bloom_depth"] * 100 + 2, p["bloom_label"],
                ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[p["lang"]] for p in pairs])
    ax.set_ylabel("Relative depth of top head (%)")
    ax.set_ylim(0, 115)
    ax.axhline(76, color="gray", linestyle=":", alpha=0.5, label="Gemma's L13 depth")
    ax.set_title("Top-1 patching head depth — Gemma and BLOOM converge for EN/ES, diverge for FR/SW",
                 fontsize=11)
    ax.legend(loc="lower right")
    fig.savefig(FIG / "fig_cross_model_depth.png")
    plt.close(fig)
    print("  ✓ fig_cross_model_depth.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Cross-model CKA heatmap (Spanish, honest)
# ═══════════════════════════════════════════════════════════════════════════

def fig_cross_model_cka():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, lang in zip(axes, ["en", "es"]):
        d = load(RES / f"cross_cka_gemma-2b_bloom-3b_{lang}.npz")
        M = d["cka_matrix"]
        im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.3, vmax=1.0)
        ax.set_xlabel("BLOOM 3B layer")
        ax.set_ylabel("Gemma 2B layer")
        ax.set_title(f"{LANG_NAMES[lang]}: mean CKA={M.mean():.3f}, "
                     f"range {M.min():.2f}–{M.max():.2f}")
        plt.colorbar(im, ax=ax, label="linear CKA")

        # Overlay the proportional diagonal
        gl = M.shape[0]; bl = M.shape[1]
        diag_x = [round(gi * (bl - 1) / (gl - 1)) for gi in range(gl)]
        diag_y = list(range(gl))
        ax.plot(diag_x, diag_y, "r--", linewidth=1, alpha=0.7, label="proportional depth")
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Cross-model CKA (Gemma × BLOOM) — Spanish shows real diagonal; English is uniform\n"
                 "(high English values likely inflated by shared task structure at final token)",
                 fontsize=11)
    fig.savefig(FIG / "fig_cross_model_cka.png")
    plt.close(fig)
    print("  ✓ fig_cross_model_cka.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 9 — RepE per-layer accuracy
# ═══════════════════════════════════════════════════════════════════════════

def fig_repe_profiles():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True, sharey=True)
    # Gemma
    ax = axes[0]
    for lang in ["en", "es", "fr", "sw"]:
        f = RES / f"repe_gemma_2b_{lang}_mean_diff.npz"
        if not f.exists(): continue
        acc = load(f)["accuracy"]
        layers = np.arange(len(acc)) / (len(acc) - 1) * 100
        ax.plot(layers, acc, "-o", color=LANG_COLOR[lang],
                label=LANG_NAMES[lang], markersize=4)
    ax.axvline(76, color="gray", linestyle=":", alpha=0.5, label="L13 (key head)")
    ax.set_xlabel("Relative depth (%)")
    ax.set_ylabel("Per-layer classifier accuracy")
    ax.set_title("Gemma 2B")
    ax.legend(fontsize=9, loc="lower center")
    ax.grid(alpha=0.3)

    # BLOOM
    ax = axes[1]
    for lang in BLOOM_LANGS:
        f = RES / f"repe_bloom_3b_{lang}_mean_diff.npz"
        if not f.exists(): continue
        acc = load(f)["accuracy"]
        layers = np.arange(len(acc)) / (len(acc) - 1) * 100
        ax.plot(layers, acc, "-o", color=LANG_COLOR[lang],
                label=LANG_NAMES[lang], markersize=4)
    ax.axvline(77, color="gray", linestyle=":", alpha=0.5, label="L23 (key head)")
    ax.set_xlabel("Relative depth (%)")
    ax.set_title("BLOOM 3B")
    ax.legend(fontsize=9, loc="lower center")
    ax.grid(alpha=0.3)

    fig.suptitle("RepE signal emergence — per-layer singular/plural classifier accuracy\n"
                 "(English is trivially separable early; others peak near the key layer)",
                 fontsize=11)
    fig.savefig(FIG / "fig_repe_profiles.png")
    plt.close(fig)
    print("  ✓ fig_repe_profiles.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 10 — PCA scatter
# ═══════════════════════════════════════════════════════════════════════════

def fig_pca_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Gemma L13H7
    ax = axes[0]
    d = load(RES / "pca_L13H7.npz")
    proj = d["projections"]; labels = d["labels"]
    ax.scatter(proj[labels == 0], np.random.normal(0, 0.03, sum(labels == 0)),
               color="#1f77b4", label="singular", alpha=0.7, s=40)
    ax.scatter(proj[labels == 1], np.random.normal(0, 0.03, sum(labels == 1)),
               color="#d62728", label="plural", alpha=0.7, s=40)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
    sg_m = proj[labels == 0].mean(); pl_m = proj[labels == 1].mean()
    ax.axvline(sg_m, color="#1f77b4", linestyle="--", alpha=0.8, linewidth=1)
    ax.axvline(pl_m, color="#d62728", linestyle="--", alpha=0.8, linewidth=1)
    ax.set_xlabel("PC1 projection")
    ax.set_title(f"Gemma 2B L13H7 — separation={abs(pl_m - sg_m):.2f}")
    ax.legend()
    ax.set_yticks([])

    # BLOOM L23H15
    ax = axes[1]
    d = load(BLOOM / "pca_L23H15.npz")
    proj = d["projections"]; labels = d["labels"]
    ax.scatter(proj[labels == 0], np.random.normal(0, 0.03, sum(labels == 0)),
               color="#1f77b4", label="singular", alpha=0.7, s=40)
    ax.scatter(proj[labels == 1], np.random.normal(0, 0.03, sum(labels == 1)),
               color="#d62728", label="plural", alpha=0.7, s=40)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
    sg_m = proj[labels == 0].mean(); pl_m = proj[labels == 1].mean()
    ax.axvline(sg_m, color="#1f77b4", linestyle="--", alpha=0.8, linewidth=1)
    ax.axvline(pl_m, color="#d62728", linestyle="--", alpha=0.8, linewidth=1)
    ax.set_xlabel("PC1 projection")
    ax.set_title(f"BLOOM 3B L23H15 — separation={abs(pl_m - sg_m):.2f}")
    ax.legend()
    ax.set_yticks([])

    fig.suptitle("PCA on key-head activations (English-fit): both models encode number as a clean 1-D direction",
                 fontsize=11)
    fig.savefig(FIG / "fig_pca_scatter.png")
    plt.close(fig)
    print("  ✓ fig_pca_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Knockout (honest, showing the anomaly)
# ═══════════════════════════════════════════════════════════════════════════

def fig_knockout_honest():
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    data = []
    for lang in GEMMA_LANGS:
        d = load(RES / f"knockout_{lang}.npz")
        data.append({
            "lang": lang,
            "baseline": float(d["baseline_accuracy"]),
            "necessity": float(d["necessity_accuracy"]),
            "sufficiency": float(d["sufficiency_accuracy"]),
            "n_heads": len(d["circuit_heads"]),
        })

    x = np.arange(len(data))
    w = 0.26
    ax.bar(x - w, [r["baseline"] for r in data], w, label="baseline",
           color="#7f7f7f", edgecolor="black")
    ax.bar(x, [r["necessity"] for r in data], w,
           label="necessity (ablate circuit — should drop)",
           color="#d62728", edgecolor="black")
    ax.bar(x + w, [r["sufficiency"] for r in data], w,
           label="sufficiency (keep only circuit — should stay high)",
           color="#2ca02c", edgecolor="black")

    for i, r in enumerate(data):
        ax.text(i, 1.02, f"n={r['n_heads']}", ha="center", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[r["lang"]] for r in data], rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title("Knockout — for 4 languages, sufficiency EXCEEDS baseline\n"
                 "(suggests baseline contains distractor heads; circuit threshold also varies wildly)",
                 fontsize=11)
    ax.legend(loc="lower right")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.3, linewidth=0.8)
    fig.savefig(FIG / "fig_knockout_honest.png")
    plt.close(fig)
    print("  ✓ fig_knockout_honest.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Weight-importance vs patching disagreement
# ═══════════════════════════════════════════════════════════════════════════

def fig_weight_vs_patch():
    """Show that structural (weight) importance and causal (patching) importance
    identify different heads."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    # Two heatmaps: mean weight importance vs mean patching (averaged across langs)
    patch_mats = []; weight_mats = []
    for lang in GEMMA_LANGS:
        patch_mats.append(load(RES / f"patching_{lang}.npz")["head_out"])
        weight_mats.append(load(RES / f"circuit_map_{lang}.npz")["head_importance"])
    patch_mean = np.mean(patch_mats, axis=0)
    weight_mean = np.mean(weight_mats, axis=0)

    # Normalize each to [0, 1] for comparable display
    p_norm = patch_mean / patch_mean.max()
    w_norm = weight_mean / weight_mean.max()

    ax = axes[0]
    im = ax.imshow(p_norm, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Causal importance (patching, mean across 7 langs)\nTop: L13H7, L17H4")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_xticks(range(8)); ax.set_yticks(range(0, 18, 2))
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(w_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Weight-level importance (circuit map, mean across 7 langs)\nTop: L0H5, L0H3")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_xticks(range(8)); ax.set_yticks(range(0, 18, 2))
    plt.colorbar(im, ax=ax)

    fig.suptitle("Two ways to score head importance give DIFFERENT answers\n"
                 "(L0 heads are 'wired for number' structurally but causally inert; "
                 "L13H7 is causally central but not weight-aligned)",
                 fontsize=11)
    fig.savefig(FIG / "fig_weight_vs_patch.png")
    plt.close(fig)
    print("  ✓ fig_weight_vs_patch.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Cross-lingual convergence (honest, showing weakness)
# ═══════════════════════════════════════════════════════════════════════════

def fig_crosslingual_cka():
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    d = load(RES / "geometry.npz")
    conv = d["convergence"]  # mean across pairs
    layers = np.arange(len(conv))

    ax.plot(layers, conv, "-o", color="#1f77b4", markersize=6)
    peak_L = int(conv.argmax())
    ax.axvline(peak_L, color="red", linestyle="--", alpha=0.6,
               label=f"peak @ L{peak_L} ({conv[peak_L]:.3f})")
    ax.axhline(conv.min(), color="gray", linestyle=":", alpha=0.5,
               label=f"floor ({conv.min():.3f})")

    ax.fill_between(layers, conv.min(), conv, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cross-lingual CKA\n(averaged across 21 lang pairs)")
    ax.set_title("Cross-lingual convergence (within Gemma) — peak at L13 is real but small\n"
                 f"(effect size: +{conv.max() - conv.min():.3f} over floor)",
                 fontsize=11)
    ax.legend()
    ax.set_ylim(0.18, 0.30)
    fig.savefig(FIG / "fig_crosslingual_cka.png")
    plt.close(fig)
    print("  ✓ fig_crosslingual_cka.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("Regenerating figures from raw NPZ data...")
    fig_patching_grid()
    fig_patching_top1()
    fig_steering_peaks()
    fig_steering_curves()
    fig_eap_mlp()
    fig_subject_vs_number()
    fig_cross_model_depth()
    fig_cross_model_cka()
    fig_repe_profiles()
    fig_pca_scatter()
    fig_knockout_honest()
    fig_weight_vs_patch()
    fig_crosslingual_cka()
    print("\nAll figures written to", FIG)


if __name__ == "__main__":
    main()
