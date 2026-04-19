"""
Deep dive into weight-level structure.

Hypotheses to test:
  A. Head "task directions" (OV-projected onto verb-number direction) — are they
     consistent across languages? A cosine similarity map per head would show
     which heads have UNIVERSAL weight-level task orientation.
  B. "Universal" MLP neurons — neurons whose importance is high in all 7 langs.
     These are structurally universal circuit elements.
  C. Connection graph — do languages share the same wiring?
  D. SVD spectra — are key heads low-rank (clean computation) or high-rank?
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

GEMMA_LANGS = ["en", "es", "fr", "ru", "sw", "tr", "qu"]

# Load all circuit maps
maps = {}
for lang in GEMMA_LANGS:
    maps[lang] = np.load(RES / f"circuit_map_{lang}.npz")

# ─────────────────────────────────────────────────────────────────────────
# A. Task direction cosine similarity across languages
# ─────────────────────────────────────────────────────────────────────────
# head_task_weights: (18, 8, 2048) — per head, the OV-projected task direction
# in residual-stream space.
# For each head, compute pairwise cosine sim across languages → if close to 1,
# that head has a universal task orientation.

print("═" * 78)
print(" A. Task-direction cosine similarity across languages (per head)")
print("═" * 78)

H, L = 18, 8
all_vecs = np.stack([maps[l]["head_task_weights"] for l in GEMMA_LANGS])
# shape: (7 langs, 18, 8, 2048)

n_langs = len(GEMMA_LANGS)
mean_cos = np.zeros((H, L))  # mean pairwise cos per head
min_cos = np.zeros((H, L))
for layer in range(H):
    for head in range(L):
        vecs = all_vecs[:, layer, head, :]  # (7, 2048)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Cos matrix
        normed = vecs / (norms + 1e-12)
        cos = normed @ normed.T  # (7, 7)
        upper = cos[np.triu_indices(n_langs, k=1)]
        mean_cos[layer, head] = upper.mean()
        min_cos[layer, head] = upper.min()

# Rank heads by mean cosine (most universal task direction)
print("\nTop-10 heads by mean cross-lingual cosine similarity of task direction:")
flat_idx = np.argsort(-mean_cos.flatten())[:10]
for i in flat_idx:
    l, h = i // L, i % L
    print(f"  L{l}H{h}: mean_cos={mean_cos[l, h]:.3f}  min_cos={min_cos[l, h]:.3f}")

print("\nBottom-10 (most language-specific task direction):")
for i in np.argsort(mean_cos.flatten())[:10]:
    l, h = i // L, i % L
    print(f"  L{l}H{h}: mean_cos={mean_cos[l, h]:.3f}  min_cos={min_cos[l, h]:.3f}")

# Specifically look at the paper's head
print(f"\nL13H7 task direction consistency: mean_cos={mean_cos[13, 7]:.3f}  min_cos={min_cos[13, 7]:.3f}")
print(f"L17H4 task direction consistency: mean_cos={mean_cos[17, 4]:.3f}  min_cos={min_cos[17, 4]:.3f}")
print(f"L0H5  task direction consistency: mean_cos={mean_cos[0, 5]:.3f}  min_cos={min_cos[0, 5]:.3f}")
print(f"L0H3  task direction consistency: mean_cos={mean_cos[0, 3]:.3f}  min_cos={min_cos[0, 3]:.3f}")

# Save for plotting
np.savez(RES / "universal_task_direction.npz",
         mean_cos=mean_cos, min_cos=min_cos, all_vecs=all_vecs)


# ─────────────────────────────────────────────────────────────────────────
# B. Universal MLP neurons
# ─────────────────────────────────────────────────────────────────────────

print("\n\n" + "═" * 78)
print(" B. Universal MLP neurons")
print("═" * 78)

# mlp_neuron_importance: (18, 16384)
all_neurons = np.stack([maps[l]["mlp_neuron_importance"] for l in GEMMA_LANGS])
# (7, 18, 16384)

# For each (layer, neuron), compute mean importance across langs and consistency
# (1 - std/mean, high = consistent magnitude)
mean_imp = np.mean(np.abs(all_neurons), axis=0)  # (18, 16384)
std_imp = np.std(np.abs(all_neurons), axis=0)

# Rank: neurons with HIGH mean importance AND LOW variance (universal)
# Score = mean / (std + 0.01)
consistency_score = mean_imp / (std_imp + 0.01)

# Top universal neurons in each key layer (13, 17)
for layer in [13, 17]:
    print(f"\nLayer {layer} — top-10 universal neurons (high importance AND consistent):")
    top_idx = np.argsort(-mean_imp[layer])[:20]
    for idx in top_idx[:10]:
        per_lang = all_neurons[:, layer, idx]
        print(f"  n{idx}: mean_|imp|={mean_imp[layer, idx]:.3f}  std={std_imp[layer, idx]:.3f}  "
              f"per_lang={[round(x, 2) for x in per_lang]}")


# ─────────────────────────────────────────────────────────────────────────
# C. Connection matrix consistency
# ─────────────────────────────────────────────────────────────────────────

print("\n\n" + "═" * 78)
print(" C. Head-to-head connection graph consistency across languages")
print("═" * 78)
all_conn = np.stack([maps[l]["connection_matrix"] for l in GEMMA_LANGS])  # (7, 144, 144)

# Which edges are consistently strong across languages?
mean_conn = np.mean(all_conn, axis=0)
std_conn = np.std(all_conn, axis=0)

# Top universal edges
labels_en = maps["en"]["connection_labels"]
flat = mean_conn.flatten()
top_idx = np.argsort(-flat)[:15]
print("\nTop 15 universal connections (mean across langs):")
for i in top_idx:
    src_i = i // 144; dst_i = i % 144
    print(f"  {labels_en[src_i]} → {labels_en[dst_i]}: mean={mean_conn[src_i, dst_i]:.3f}  "
          f"std={std_conn[src_i, dst_i]:.3f}")


# ─────────────────────────────────────────────────────────────────────────
# D. SVD spectra
# ─────────────────────────────────────────────────────────────────────────

print("\n\n" + "═" * 78)
print(" D. SVD spectra — functional rank of each head")
print("═" * 78)

# svd_spectra: (18, 8, 10) — top 10 singular values of each head's OV-circuit
for lang in ["en", "es"]:
    spectra = maps[lang]["svd_spectra"]  # (18, 8, 10)
    # Participation ratio: fraction of variance in top-1 singular value
    top1_frac = np.zeros((18, 8))
    for L_i in range(18):
        for H_i in range(8):
            s = spectra[L_i, H_i]
            s2 = s ** 2
            top1_frac[L_i, H_i] = s2[0] / (s2.sum() + 1e-12)

    # L13H7 specifically
    print(f"\n{lang}:")
    print(f"  L13H7 top-1 variance frac: {top1_frac[13, 7]:.3f}  (1.0 = rank-1 head)")
    print(f"  L17H4 top-1 variance frac: {top1_frac[17, 4]:.3f}")
    print(f"  L0H5  top-1 variance frac: {top1_frac[0, 5]:.3f}")
    # Most "peaked" heads
    flat = top1_frac.flatten()
    top = np.argsort(-flat)[:5]
    print("  Most peaked (~rank-1) heads:")
    for i in top:
        print(f"    L{i // 8}H{i % 8}: top1_frac={flat[i]:.3f}")

np.savez(RES / "svd_summary.npz",
         top1_frac_per_lang=np.stack([
             np.array([[maps[l]["svd_spectra"][L_i, H_i, 0]**2 /
                       (maps[l]["svd_spectra"][L_i, H_i]**2).sum()
                       for H_i in range(8)] for L_i in range(18)])
             for l in GEMMA_LANGS
         ]))
