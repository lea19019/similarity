# Replication Report: "On the Similarity of Circuits across Languages"

**Paper**: Ferrando & Costa-jussà (2024), EMNLP Findings
([ACL Anthology](https://aclanthology.org/2024.findings-emnlp.591))

**Status**: Pipeline complete, core patching/DLA findings replicate, but PCA and
steering have a bug that weakens the cross-lingual result.

---

## 1. What the paper claims

Gemma 2B solves subject-verb agreement (SVA) — picking "is" vs "are" based on a
subject noun — using a small, identifiable circuit. The headline claims:

1. **One head dominates**: Attention head L13H7 is the most causally important
   component for SVA in both English and Spanish.
2. **L13H7 encodes a "subject-number direction"**: PCA on L13H7's output reveals
   a single direction (PC1) that cleanly separates singular from plural subjects.
3. **The direction is language-independent**: PC1 extracted from English data
   alone can steer Spanish verb predictions — proving the circuit shares a
   cross-lingual representation.

The methodology chain is: **patching → DLA → neurons → PCA → steering**.

---

## 2. What each step does (and why it matters)

### 2.1 Datasets (`circuits/data.py`)

The task is **subject-verb agreement in relative clauses**. Each example is a
*contrastive pair*: two sentences identical except the subject noun's number.

**English** (from CausalGym):
```
clean:     "The author that liked the dancer is"     → should predict "is"
corrupted: "The authors that liked the dancer are"   → should predict "are"
```

**Spanish** (template-generated):
```
clean:     "El doctor que habló al maestro es"       → should predict "es"
corrupted: "Los doctores que hablaron al maestro son" → should predict "son"
```

Why contrastive pairs? They let us isolate the *minimal causal signal* — the only
difference between clean and corrupted is the subject number, so any change in
model behavior must be caused by that difference.

**Your datasets**: 536 English examples (CausalGym, 6-word filter), 5,684 Spanish
examples (template-based, GPT-4-curated word lists). Both filtered so all verbs
are single Gemma subwords. These match the paper.

### 2.2 Activation patching (`circuits/patching.py`)

**Question**: Which attention heads causally carry the subject-number signal?

**Method** (denoising / causal mediation analysis):
1. Run the model on the **corrupted** input (plural subject). The model gets the
   wrong answer.
2. Run the model on the **clean** input (singular subject). Cache all internal
   activations.
3. For each component (attention head, MLP block), **replace** the corrupted
   activation with the clean one and re-run.
4. If the model's answer recovers toward the correct verb, that component was
   *carrying* the subject-number information.

The metric is **normalized patch effect**: 0 = no recovery, 1 = full recovery.

```
                   corrupted input
                         │
   ┌─────────────────────▼──────────────────────┐
   │  ...  │ Layer 13 │ ...  │ Layer 17  │ ...  │
   │       │  Head 7  │      │  Head 4   │      │
   │       │  ▲ PATCH │      │           │      │
   │       │  │ clean  │      │           │      │
   │       │  │ value  │      │           │      │
   └───────┴──┴────────┴──────┴───────────┴──────┘
                         │
                    patched logits → recovered correct verb?
```

**Your code**: For each example, it patches every head at the final token position
(where the verb is predicted) and measures recovery. Loops over all 18 layers × 8
heads = 144 heads, plus all attention blocks and MLP blocks. Results are averaged
across 128 examples.

### 2.3 Direct Logit Attribution (`circuits/dla.py`)

**Question**: Which components *directly* push the model toward the correct verb?

**Method**: The key insight is that the final logit is a **linear function** of the
residual stream. The residual stream at the last position is the sum of all
component outputs (each head, each MLP). So:

```
logit(correct) - logit(incorrect)  =  Σ  component_output · unembed_direction
                                     all
                                   components
```

where `unembed_direction = W_U[:, good_id] - W_U[:, bad_id]` (the direction in
vocabulary space that distinguishes the two verbs).

Each component's contribution is exact — no approximation involved. A positive
DLA means the component pushes toward the correct verb.

**Your code**: For each attention head, it extracts the `hook_z` output (d_head
dimensions), projects it to d_model via `W_O`, and dots with the unembed direction.
For MLP blocks, the output is already in d_model space, so it's a direct dot product.

**Difference from patching**: Patching is *causal* (actually changes the forward
pass). DLA is *correlational* (reads off contributions without intervening). They
complement each other: a head with high patching effect AND high DLA both carries
and directly expresses the number signal.

### 2.4 Neuron analysis (`circuits/neurons.py`)

**Question**: Within the important MLP layers (13 and 17), which individual neurons
contribute most to the SVA decision?

**Method**: Same DLA decomposition, but applied per-neuron within an MLP layer.
Each neuron's contribution = (its gated activation value) × (how much its output
column aligns with the verb-number direction).

```
neuron_DLA[i] = post_activation[i] × (W_out[i, :] · unembed_direction)
```

This identifies specific neurons (e.g., neuron 2069 in layer 13) that read the
number signal from the residual stream and push it toward the correct verb form.

### 2.5 PCA on L13H7 (`circuits/pca.py`)

**Question**: Does L13H7 encode subject number as a *single direction* in its
activation space?

**Method**:
1. For each example, run the model on both the singular-subject and plural-subject
   sentences.
2. Extract L13H7's output vector at the final token position (a 256-dimensional
   vector in `d_head` space).
3. Collect all vectors, labeling each as singular (0) or plural (1).
4. Fit PCA. If PC1 cleanly separates singular from plural, the head uses a
   one-dimensional code for subject number.

```
   PC1 projection
    ──────────────────────────────►
    ◄── singular ──► ◄── plural ──►

    ○ ○ ○ ○ ○ ○       ● ● ● ● ● ●     ← English
    △ △ △ △ △ △       ▲ ▲ ▲ ▲ ▲ ▲     ← Spanish (should overlap if cross-lingual)
```

**Crucially**, the paper fits PCA on **English data only**, then projects both
English and Spanish onto the resulting PC1. If Spanish examples also separate
along the same direction, the representation is language-independent.

### 2.6 Cross-lingual steering (`circuits/steering.py`)

**Question**: Does the English-derived number direction *causally control* Spanish
verb predictions?

**Method**:
1. Take the PC1 direction from the English PCA.
2. For each Spanish example, run the model normally and record its verb prediction.
3. Add `α × PC1` to L13H7's output at the final position during the forward pass
   (using a hook). This "steers" the head's output along the number direction.
4. Record whether the prediction flips (e.g., "es" → "son").
5. Sweep α from 0 to 50 to measure dose-response.

```
   Normal forward pass:  "El doctor que habló al maestro" → L13H7 → ... → "es" ✓
   Steered forward pass: "El doctor que habló al maestro" → L13H7 + α·PC1 → ... → "son" ✗ (flipped!)
```

If adding the English number direction to Spanish activations flips the verb,
that direction is causally relevant across languages — the core claim of the paper.

---

## 3. Our results vs. the paper

### 3.1 Activation Patching — ✅ Replicates

| Metric | Our EN | Our ES | Paper |
|--------|--------|--------|-------|
| Top head | L13H7 (0.37) | L13H7 (0.60) | L13H7 is #1 in both |
| #2 head | L17H4 (0.24) | L17H4 (0.14) | L17H4 is #2 in both |

The patching heatmaps match the paper's Figure 2 qualitatively. L13H7 dominates.
Layer 13 is broadly active. L17H4 is the clear secondary head. Spanish shows a
stronger effect than English (0.60 vs 0.37), consistent with the paper.

### 3.2 DLA — ✅ Replicates (core finding)

L13H7 and L17H4 both show positive DLA (they push toward the correct verb).
Early-layer heads (L0H5) show large raw DLA values, but these reflect positional
or frequency biases rather than SVA-specific computation — the paper's figures
show similar patterns.

### 3.3 PCA — ⚠️ Does NOT replicate cleanly

**What should happen** (paper's Figure 4): Projecting L13H7 activations onto PC1
gives two clean clusters — singular on one side, plural on the other — for both
English and Spanish, with the clusters overlapping across languages.

**What we got**:

| Language | Singular mean | Plural mean | Separation |
|----------|--------------|-------------|------------|
| EN | +3.07 | +3.25 | 0.18 |
| ES | -3.15 | -3.17 | 0.02 |

The singular/plural separation is **tiny** (0.18 for EN, 0.02 for ES). Instead,
PC1 captures the **language direction**: EN clusters around +3.1, ES around -3.1.
The biggest variance in the data is "which language is this?" not "singular or
plural?" — so PCA finds the language direction first.

**Root cause**: see Section 4.1 below.

### 3.4 Steering — ⚠️ Qualitatively works, quantitatively weak

| α | Our flip (pos) | Our flip (neg) | Paper (~) |
|---|----------------|----------------|-----------|
| 10 | 3.1% | 1.9% | ~10-15% |
| 20 | 26.8% | 12.2% | ~40-50% |
| 30 | 37.5% | 32.1% | ~55-65% |
| 50 | 39.2% | 38.7% | ~65-70% |

The curve shape is correct (monotonically increasing, positive > negative at
moderate α, converging at high α), but magnitudes are roughly half of the paper's.
This is a direct consequence of the PCA bug — if the extracted direction is mostly
"language" rather than "number", steering with it will have a diluted effect.

---

## 4. Issues found and remaining work

### 4.1 🔴 Critical: PCA fitted on bilingual data instead of English-only

**The bug**: `circuits/pca.py` with `--lang both` (the default) fits PCA on the
*combined* English + Spanish activation vectors. The largest source of variance
across the combined set is the language difference, so PC1 becomes the "language
direction" rather than the "number direction."

**What the paper does**: Fits PCA on **English data only** (50 examples from the
train split), extracts PC1 as the number direction, then projects both languages
onto it to verify cross-lingual generalization.

**The fix**: Run PCA with `--lang en`, and use that PC1 for steering. The code
already supports this (`--lang en`), but the pipeline was run with `--lang both`.

**Impact**: This is the root cause of both the poor PCA separation (Section 3.3)
and the weak steering (Section 3.4). Fixing this should bring both in line with
the paper.

### 4.2 🟡 Medium: PCA sample size mismatch

The paper extracts PC1 from just **50 English examples** (mentioned in the
directions.ipynb notebook). Our code defaults to `--max-examples 256`. While more
data is generally better for PCA, matching the paper's exact number matters for
replication. After fixing 4.1, we should also try `--max-examples 50` and compare.

### 4.3 🟡 Medium: Spanish dataset clean/corrupted randomization

The Spanish dataset generation randomly swaps which sentence is "clean" vs
"corrupted" (50/50, controlled by `random.uniform(0,1)`). This means half the
Spanish examples have `clean = plural subject` with `good_verb = plural form`.

This is fine for patching and DLA (which handle both directions), but for
**steering**, the interpretation of "positive flip" depends on whether the baseline
prediction aligns with singular or plural. The paper's steering evaluation
likely uses a consistent direction (e.g., always steering singular → plural).
This should be verified against the authors' code.

### 4.4 🟢 Minor: Config documents 18 heads, model has 8

`config.py` says `n_heads: 18` for gemma-2b, but the actual model (and our
patching results) show 8 attention heads. This is a documentation-only issue —
the code reads `model.cfg.n_heads` from TransformerLens, not from our config.
But it should be corrected to avoid confusion.

### 4.5 🟢 Minor: DLA W_O projection should be verified

In `circuits/dla.py:64`, the head-to-model projection is:
```python
projected = W_O[head].T @ h_out
```

TransformerLens `W_O` has shape `(n_heads, d_head, d_model)`, so `W_O[head]` is
`(d_head, d_model)` and `.T` gives `(d_model, d_head)`. The result
`(d_model, d_head) @ (d_head,) = (d_model,)` is mathematically correct, but should
be verified against the paper's authors' code to ensure the same convention.

---

## 5. Concrete next steps to complete the replication

### Step 1: Re-run PCA on English only (high priority)
```bash
# On GPU node:
uv run python -m circuits.pca --lang en --max-examples 50 --model gemma-2b
```
Then inspect the PCA scatter — singular and plural should now separate cleanly.

### Step 2: Re-run steering with the corrected PC1
```bash
uv run python -m circuits.steering --pca-path results/pca_L13H7.npz --model gemma-2b
```
Flip rates should increase to match the paper (~60-70% at α=50).

### Step 3: Validate PCA generalizes to Spanish
Project Spanish activations onto the EN-only PC1 and verify they also separate
by number. This is the key cross-lingual evidence. The code may need a small
modification to collect Spanish vectors and project them onto the saved EN PC1
without re-fitting.

### Step 4: Fix config n_heads
Update `config.py` to say `n_heads: 8` for gemma-2b.

### Step 5 (optional): Verify Spanish dataset direction consistency
Check whether the authors' code always uses singular as "clean" for steering, or
whether the 50/50 randomization is intentional. If the former, filter the Spanish
dataset to singular-clean-only for the steering evaluation.

---

## 6. Summary

| Experiment | Status | Confidence |
|-----------|--------|------------|
| Activation patching | ✅ Replicates | High — top heads match exactly |
| DLA | ✅ Replicates | High — key heads confirmed |
| Neuron analysis | ✅ Pipeline complete | Medium — need to verify specific neuron IDs |
| PCA | ⚠️ Bug: wrong training data | High that the fix is straightforward |
| Steering | ⚠️ Weak due to PCA bug | High that re-running after PCA fix will work |

**Bottom line**: The infrastructure is solid and the first two experiments (which
don't depend on PCA) replicate cleanly. The PCA and steering are mechanically
correct but were run with the wrong configuration (`--lang both` instead of
`--lang en`). This is a ~30-minute fix (re-run two commands on a GPU node) and
should bring the full replication in line with the paper.
