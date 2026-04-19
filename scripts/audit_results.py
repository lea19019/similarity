"""
Comprehensive audit of raw result files.

Loads every .npz in results/ and results/bloom-3b/ and prints ground-truth
numbers for every analysis. Does NOT interpret — just reports.

Output: results/AUDIT.txt
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BLOOM = RES / "bloom-3b"

GEMMA_LANGS = ["en", "es", "fr", "ru", "sw", "tr", "qu"]
BLOOM_LANGS = ["en", "es", "fr", "sw"]

out_lines = []
def P(s=""):
    out_lines.append(s)
    print(s)


def topk_heads(arr, k=5):
    """Return top-k (layer, head, value) from a (L, H) array."""
    flat = arr.flatten()
    idx = np.argsort(-flat)[:k]
    return [(int(i // arr.shape[1]), int(i % arr.shape[1]), float(arr.flat[i])) for i in idx]


def topk_components(arr, labels, k=5):
    """Return top-k (label, value) from flat array with labels."""
    idx = np.argsort(-arr)[:k]
    return [(str(labels[i]), float(arr[i])) for i in idx]


def bottomk_heads(arr, k=3):
    flat = arr.flatten()
    idx = np.argsort(flat)[:k]
    return [(int(i // arr.shape[1]), int(i % arr.shape[1]), float(arr.flat[i])) for i in idx]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: ACTIVATION PATCHING (causal)
# ═══════════════════════════════════════════════════════════════════════════

P("═" * 78)
P(" 1. ACTIVATION PATCHING — top-5 heads per language")
P("═" * 78)
P(f"{'Lang':<6} Top-5 heads (layer, head, patch_effect)")

gemma_patching = {}
for lang in GEMMA_LANGS:
    f = RES / f"patching_{lang}.npz"
    if not f.exists():
        P(f"{lang:<6} MISSING"); continue
    d = np.load(f)
    top = topk_heads(d["head_out"], 5)
    gemma_patching[lang] = {"head_out": d["head_out"], "top": top,
                            "attn_out": d["attn_out"], "mlp_out": d["mlp_out"]}
    P(f"{lang:<6} " + "  ".join(f"L{l}H{h}={v:.3f}" for l, h, v in top))

P()
P("BLOOM patching:")
bloom_patching = {}
for lang in BLOOM_LANGS:
    f = BLOOM / f"patching_{lang}.npz"
    if not f.exists():
        P(f"{lang:<6} MISSING"); continue
    d = np.load(f)
    top = topk_heads(d["head_out"], 5)
    bloom_patching[lang] = {"head_out": d["head_out"], "top": top}
    P(f"{lang:<6} " + "  ".join(f"L{l}H{h}={v:.3f}" for l, h, v in top))

P()
P("Max patching effect per language (signal strength):")
P(f"{'Lang':<6} {'Gemma_max':>10} {'Gemma_top_head':>15} {'BLOOM_max':>10} {'BLOOM_top_head':>15}")
for lang in GEMMA_LANGS:
    g = gemma_patching.get(lang, {})
    b = bloom_patching.get(lang, {})
    gmax = g["top"][0] if g else ("--", "--", np.nan)
    bmax = b["top"][0] if b else ("--", "--", np.nan)
    P(f"{lang:<6} "
      f"{gmax[2]:>10.3f} {f'L{gmax[0]}H{gmax[1]}':>15} "
      f"{(bmax[2] if isinstance(bmax[2], float) else np.nan):>10.3f} "
      f"{(f'L{bmax[0]}H{bmax[1]}' if isinstance(bmax[0], int) else '--'):>15}")

# Layer concentration: what fraction of total patching is in which layer?
P()
P("Layer concentration (top-1 patching cell location):")
for lang in GEMMA_LANGS:
    if lang not in gemma_patching:
        continue
    mat = gemma_patching[lang]["head_out"]
    layer_sum = mat.max(axis=1)  # best head per layer
    top_layer = int(np.argmax(layer_sum))
    P(f"  {lang}: best layer = L{top_layer}  (max={layer_sum[top_layer]:.3f}); "
      f"Gemma has 18 layers so L{top_layer} = {100*top_layer/17:.0f}% depth")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: DLA
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 2. DIRECT LOGIT ATTRIBUTION (correlational)")
P("═" * 78)
P(f"{'Lang':<6} Top-3 heads by DLA  |  Bottom-3 (most negative)")
for lang in GEMMA_LANGS:
    f = RES / f"dla_{lang}.npz"
    if not f.exists():
        P(f"{lang:<6} MISSING"); continue
    d = np.load(f)
    top = topk_heads(d["head_dla"], 3)
    bot = bottomk_heads(d["head_dla"], 3)
    P(f"{lang:<6} "
      + "  ".join(f"L{l}H{h}={v:+.2f}" for l, h, v in top)
      + "  |  "
      + "  ".join(f"L{l}H{h}={v:+.2f}" for l, h, v in bot))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: EDGE ATTRIBUTION PATCHING
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 3. EDGE ATTRIBUTION PATCHING (attention + MLP together)")
P("═" * 78)
P(f"{'Lang':<6} Top-5 components (all types)")
for lang in GEMMA_LANGS:
    f = RES / f"edge_patching_{lang}.npz"
    if not f.exists():
        P(f"{lang:<6} MISSING"); continue
    d = np.load(f)
    top = topk_components(d["node_scores"], d["component_labels"], 5)
    P(f"{lang:<6} " + "  ".join(f"{lbl}={v:.3f}" for lbl, v in top))

P()
P("BLOOM EAP:")
for lang in BLOOM_LANGS:
    f = BLOOM / f"edge_patching_{lang}.npz"
    if not f.exists():
        P(f"{lang:<6} MISSING"); continue
    d = np.load(f)
    top = topk_components(d["node_scores"], d["component_labels"], 5)
    P(f"{lang:<6} " + "  ".join(f"{lbl}={v:.3f}" for lbl, v in top))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: STEERING
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 4. CROSS-LINGUAL STEERING (EN direction → target lang)")
P("═" * 78)
P(f"{'Target':<8} alphas     flip_rate_pos         flip_rate_neg            peak_pos   peak_neg")
for lang in ["es", "fr", "ru", "sw", "tr", "qu"]:
    f = RES / f"steering_{lang}.npz"
    if not f.exists():
        P(f"{lang:<8} MISSING"); continue
    d = np.load(f)
    a = d["alphas"]; p = d["flip_rate_pos"]; n = d["flip_rate_neg"]
    peak_p = (float(p.max()), float(a[int(p.argmax())]))
    peak_n = (float(n.max()), float(a[int(n.argmax())]))
    P(f"{lang:<8} {list(a.astype(int))}  pos={[round(x,2) for x in p]}  "
      f"neg={[round(x,2) for x in n]}  "
      f"pk_pos={peak_p[0]:.2f}@α={peak_p[1]:.0f}  pk_neg={peak_n[0]:.2f}@α={peak_n[1]:.0f}")

P()
P("BLOOM steering:")
for lang in ["es", "fr", "sw"]:
    f = BLOOM / f"steering_{lang}.npz"
    if not f.exists():
        P(f"{lang:<8} MISSING"); continue
    d = np.load(f)
    a = d["alphas"]; p = d["flip_rate_pos"]; n = d["flip_rate_neg"]
    peak_p = (float(p.max()), float(a[int(p.argmax())]))
    peak_n = (float(n.max()), float(a[int(n.argmax())]))
    P(f"{lang:<8} pos={[round(x,2) for x in p]}  "
      f"neg={[round(x,2) for x in n]}  "
      f"pk_pos={peak_p[0]:.2f}@α={peak_p[1]:.0f}  pk_neg={peak_n[0]:.2f}@α={peak_n[1]:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: KNOCKOUT (necessity + sufficiency)
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 5. CIRCUIT KNOCKOUT")
P("═" * 78)
P(f"{'Lang':<6} {'n_heads':>7} {'baseline':>9} {'necessity':>10} {'sufficiency':>12}  interpretation")
for lang in GEMMA_LANGS:
    f = RES / f"knockout_{lang}.npz"
    if not f.exists():
        P(f"{lang:<6} MISSING"); continue
    d = np.load(f)
    nh = len(d["circuit_heads"])
    bl = float(d["baseline_accuracy"]); nec = float(d["necessity_accuracy"]); suf = float(d["sufficiency_accuracy"])
    nec_drop = bl - nec
    suf_keep = suf / bl if bl > 0 else 0
    interp = f"drop={nec_drop:+.2f}, keep={suf_keep*100:.0f}%"
    P(f"{lang:<6} {nh:>7} {bl:>9.2f} {nec:>10.2f} {suf:>12.2f}  {interp}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: LOGIT LENS (layer of emergence)
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 6. LOGIT LENS — layer at which logit_diff first crosses 50% of max")
P("═" * 78)
for lang in GEMMA_LANGS:
    f = RES / f"logit_lens_{lang}.npz"
    if not f.exists():
        P(f"{lang:<6} MISSING"); continue
    d = np.load(f)
    ld = d["mean_logit_diff"]
    mx = ld.max()
    threshold = 0.5 * mx
    emergence = int(np.argmax(ld >= threshold))
    peak = int(np.argmax(ld))
    P(f"{lang:<6} ld_peak={mx:.2f} @L{peak} ({100*peak/(len(ld)-1):.0f}%)  "
      f"reaches 50% @L{emergence} ({100*emergence/(len(ld)-1):.0f}%)  "
      f"per-layer: {[round(x,1) for x in ld]}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: ATTENTION (who reads the subject)
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 7. ATTENTION — top-5 subject-reading heads per language")
P("═" * 78)
for lang in GEMMA_LANGS:
    f = RES / f"attention_{lang}.npz"
    if not f.exists():
        P(f"{lang:<6} MISSING"); continue
    d = np.load(f)
    sa = d["subject_attention"]
    top = topk_heads(sa, 5)
    P(f"{lang:<6} " + "  ".join(f"L{l}H{h}={v:.2f}" for l, h, v in top))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: NEURONS (MLP neurons driving prediction)
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 8. MLP NEURONS — top-5 in layer 13 and layer 17 per language")
P("═" * 78)
for lang in GEMMA_LANGS:
    f = RES / f"neurons_{lang}.npz"
    if not f.exists():
        P(f"{lang:<6} MISSING"); continue
    d = np.load(f)
    for L in [13, 17]:
        key = f"layer_{L}"
        if key not in d.files: continue
        arr = d[key]
        idx = np.argsort(-np.abs(arr))[:5]
        P(f"{lang:<6} L{L}: " + "  ".join(f"n{int(i)}={arr[i]:+.2f}" for i in idx))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: CIRCUIT MAP — weight-level importance
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 9. CIRCUIT MAP — top-5 heads by WEIGHT-LEVEL importance (vs patching)")
P("═" * 78)
for lang in GEMMA_LANGS:
    f = RES / f"circuit_map_{lang}.npz"
    if not f.exists():
        P(f"{lang:<6} MISSING"); continue
    d = np.load(f)
    top = topk_heads(d["head_importance"], 5)
    P(f"{lang:<6} " + "  ".join(f"L{l}H{h}={v:.2f}" for l, h, v in top))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: PCA on L13H7
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 10. PCA on L13H7 (English-fit)")
P("═" * 78)
for f in [RES / "pca_L13H7.npz", RES / "pca_L13H7_old_all_langs.npz"]:
    if not f.exists(): continue
    d = np.load(f)
    proj = d["projections"]; labels = d["labels"]
    sg_mean = proj[labels == 0].mean() if (labels == 0).any() else float("nan")
    pl_mean = proj[labels == 1].mean() if (labels == 1).any() else float("nan")
    sep = abs(sg_mean - pl_mean)
    P(f"{f.name}: sg_mean={sg_mean:.2f}  pl_mean={pl_mean:.2f}  separation={sep:.2f}  "
      f"n={len(proj)}")
    if "langs" in d.files:
        import collections
        P(f"  langs in sample: {dict(collections.Counter(d['langs']))}")

# BLOOM PCA
f = BLOOM / "pca_L23H15.npz"
if f.exists():
    d = np.load(f)
    proj = d["projections"]; labels = d["labels"]
    sg_mean = proj[labels == 0].mean() if (labels == 0).any() else float("nan")
    pl_mean = proj[labels == 1].mean() if (labels == 1).any() else float("nan")
    sep = abs(sg_mean - pl_mean)
    P(f"BLOOM pca_L23H15.npz: sg_mean={sg_mean:.2f}  pl_mean={pl_mean:.2f}  separation={sep:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: GEOMETRY — cross-lingual CKA across layers
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 11. CROSS-LINGUAL GEOMETRY (within Gemma, across language pairs)")
P("═" * 78)
f = RES / "geometry.npz"
if f.exists():
    d = np.load(f)
    cka = d["cka_per_layer"]; pair_labels = d["pair_labels"]
    conv = d["convergence"]
    P(f"Per-pair mean CKA across 18 layers:")
    for i, pair in enumerate(pair_labels):
        pair_cka = cka[:, i]
        P(f"  {pair}: mean={pair_cka.mean():.3f}  peak={pair_cka.max():.3f} @L{pair_cka.argmax()}  min={pair_cka.min():.3f}")
    P(f"\nConvergence (mean across all pairs) by layer:")
    for L, c in enumerate(conv):
        marker = " ←peak" if L == int(conv.argmax()) else ""
        P(f"  L{L}: {c:.3f}{marker}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: RepE — signal emergence
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 12. RepE — per-layer classifier accuracy (mean-diff method)")
P("═" * 78)
for model in ["gemma_2b", "bloom_3b"]:
    langs = GEMMA_LANGS if "gemma" in model else BLOOM_LANGS
    P(f"\n{model.upper()}:")
    for lang in langs:
        f = RES / f"repe_{model}_{lang}_mean_diff.npz"
        if not f.exists():
            continue
        d = np.load(f)
        acc = d["accuracy"]
        peak_L = int(acc.argmax())
        P(f"  {lang}: peak_acc={acc.max():.3f} @L{peak_L} ({100*peak_L/(len(acc)-1):.0f}%)  "
          f"mean={acc.mean():.3f}  min={acc.min():.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13: CROSS-MODEL CKA (Gemma vs BLOOM)
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 13. CROSS-MODEL CKA (Gemma × BLOOM)")
P("═" * 78)
for lang in ["en", "es"]:
    f = RES / f"cross_cka_gemma-2b_bloom-3b_{lang}.npz"
    if not f.exists():
        P(f"{lang}: MISSING"); continue
    d = np.load(f)
    M = d["cka_matrix"]  # 18 × 30
    P(f"\n{lang}: matrix shape={M.shape}, overall mean={M.mean():.3f}  min={M.min():.3f}  max={M.max():.3f}")
    # Diagonal (proportional mapping)
    P(f"  depth-proportional diagonal (Gemma_i → BLOOM_round(i*30/18)):")
    for gi in range(M.shape[0]):
        bj = round(gi * (M.shape[1] - 1) / (M.shape[0] - 1))
        P(f"    L{gi} ↔ L{bj}: CKA={M[gi, bj]:.3f}")
    # Off-diagonal magnitude check
    diag = np.array([M[gi, round(gi*(M.shape[1]-1)/(M.shape[0]-1))] for gi in range(M.shape[0])])
    # First column (BLOOM L0) compared to diagonal
    col0_mean = M[:, 0].mean()
    row0_mean = M[0, :].mean()
    P(f"  mean of first col (BLOOM L0 vs all Gemma) = {col0_mean:.3f}")
    P(f"  mean of first row (Gemma L0 vs all BLOOM) = {row0_mean:.3f}")
    P(f"  mean of diagonal = {diag.mean():.3f}")
    P(f"  if off-diag ≈ diag, the 'diagonal' story is weak.")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14: Cross-model JSON summary
# ═══════════════════════════════════════════════════════════════════════════

P()
P("═" * 78)
P(" 14. CROSS-MODEL JSON SUMMARY")
P("═" * 78)
f = RES / "cross_model_gemma-2b_bloom-3b.json"
if f.exists():
    j = json.loads(f.read_text())
    P(json.dumps(j, indent=2)[:3000])


# ═══════════════════════════════════════════════════════════════════════════
# WRITE TO FILE
# ═══════════════════════════════════════════════════════════════════════════

(RES / "AUDIT.txt").write_text("\n".join(out_lines))
print(f"\n\nWrote audit to {RES / 'AUDIT.txt'}")
