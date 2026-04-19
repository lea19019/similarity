"""
Sanity check: is the L0 "universal task direction" finding real, or a methodological artifact?

Reasons to be suspicious:
  1. Patching says L0 heads are causally inert (effect ≈ 0 in most langs).
  2. EAP also puts no L0 head in the top 5 components for any language.
  3. Gemma uses tied embeddings (W_U ≈ W_E^T). Early heads write back toward
     the embedding space, which includes the verb tokens themselves. Any head
     whose OV circuit partially reconstructs the input embedding will look
     "aligned with the verb-unembedding direction" by construction.

Test: compute the circuit_map-style "task alignment" for a RANDOM direction
in vocab space (not the real verb direction). If L0 heads also rank highly
there, the 'universality' finding is methodological, not a real circuit.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

maps = {l: np.load(RES / f"circuit_map_{l}.npz") for l in ["en", "es", "fr", "ru", "sw", "tr", "qu"]}

# head_task_weights: (18, 8, 2048) — this is OV-projected-onto-task-direction
# Its cosine across languages tells us task-direction consistency

# The actual quantity — what IS this? Let's verify.
# From circuits/circuit_map.py: head_task_weights[L, H, :] is the d_model-space
# image of the task unembedding direction under the head's OV circuit.
# i.e. W_V @ W_O projected onto (unembed_good - unembed_bad) direction, returned
# in d_model space.

# Check 1: Is the direction of head_task_weights basically the unembedding
# direction itself for L0 heads?

# Since Gemma has tied embeddings, W_V × W_O for L0 should project inputs that
# look like tokens onto outputs that look like tokens (i.e., stay in embedding
# space). So the "task direction in d_model" for an L0 head is ≈ the input
# embedding of verb tokens minus each other.

# Let's check: for L0H5, compute its task weight vector for each language,
# and compare to the L13H7 task weight vector for the same language.
# If L0H5 has a much higher cos with the raw unembedding direction than
# L13H7 does, that supports the "L0 heads just echo the unembedding" story.

# Load activations if available (to check L0 output on real data)
# Actually simpler: look at norm of task weight vectors. L0 heads being "high
# importance" could just mean their task weights have high norm because
# the input embeddings have high norm in those directions.

print("═" * 78)
print(" Norms of head_task_weights per head — is L0 just 'loud'?")
print("═" * 78)
# mean across langs of vector norm
norms = np.stack([
    np.linalg.norm(maps[l]["head_task_weights"], axis=2) for l in ["en", "es", "fr"]
]).mean(axis=0)  # (18, 8)
print(f"\nMean task-weight norms per head:")
print("Top 10 by norm:")
top = np.argsort(-norms.flatten())[:10]
for i in top:
    L, H = i // 8, i % 8
    print(f"  L{L}H{H}: mean_norm={norms[L, H]:.3f}")

print()
print(f"L13H7 norm: {norms[13, 7]:.3f}")
print(f"L0H5 norm: {norms[0, 5]:.3f}")
print(f"L0H3 norm: {norms[0, 3]:.3f}")


# Check 2: Compute cosine of the task weight vector to the UNEMBEDDING direction
# for each head. If L0 heads have cos ~= 1, they're just echoing W_U.
# (Would need W_U — but we can proxy: use the difference vector in the OV space)

# Actually we can check by computing the cosine between head_task_weights
# across ALL heads within a language to each other. If L0H5's task weight is
# basically the same direction as L0H3 in the SAME language, they're both just
# reconstructing the input embedding direction.

print()
print("═" * 78)
print(" WITHIN-language cosine similarity between heads — are L0 heads all 'same'?")
print("═" * 78)
for lang in ["en", "es", "fr"]:
    print(f"\n{lang}:")
    tw = maps[lang]["head_task_weights"]  # (18, 8, 2048)
    # Flatten per-head
    flat = tw.reshape(-1, tw.shape[-1])  # (144, 2048)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9
    normed = flat / norms
    cos = normed @ normed.T

    # For L0H5, find the 5 most similar heads
    l0h5_idx = 0 * 8 + 5  # layer 0, head 5
    sims = cos[l0h5_idx]
    top5 = np.argsort(-sims)[1:6]  # skip self
    print(f"  L0H5 — top 5 most similar heads in same language:")
    for idx in top5:
        L, H = idx // 8, idx % 8
        print(f"    L{L}H{H}: cos={sims[idx]:.3f}")

    # Same for L13H7
    l13h7_idx = 13 * 8 + 7
    sims = cos[l13h7_idx]
    top5 = np.argsort(-sims)[1:6]
    print(f"  L13H7 — top 5 most similar heads in same language:")
    for idx in top5:
        L, H = idx // 8, idx % 8
        print(f"    L{L}H{H}: cos={sims[idx]:.3f}")


# Check 3: compare EAP ranks (causal) to weight-importance ranks (structural)
print()
print("═" * 78)
print(" Causal (EAP) vs structural (weight) importance — do they agree?")
print("═" * 78)
for lang in ["en", "es", "fr"]:
    print(f"\n{lang}:")
    wi = maps[lang]["head_importance"]  # (18, 8)
    # EAP
    eap = np.load(RES / f"edge_patching_{lang}.npz")
    head_scores = eap["head_scores"]  # (18, 8)

    # Top-5 by each method
    top_wi = np.argsort(-wi.flatten())[:5]
    top_eap = np.argsort(-head_scores.flatten())[:5]

    print("  Top-5 by weight importance (circuit_map):")
    for i in top_wi:
        print(f"    L{i // 8}H{i % 8}: {wi.flat[i]:.2f}")
    print("  Top-5 by causal EAP:")
    for i in top_eap:
        print(f"    L{i // 8}H{i % 8}: {head_scores.flat[i]:.2f}")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    r, p = spearmanr(wi.flatten(), head_scores.flatten())
    print(f"  Rank correlation (weight-imp vs EAP): ρ={r:.3f} (p={p:.3f})")
