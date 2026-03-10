# On the Similarity of Circuits across Languages

Replication of **"On the Similarity of Circuits across Languages: a Case Study on the Subject-verb Agreement Task"**
by Ferrando & Costa-jussà (EMNLP Findings 2024).

- Paper: https://aclanthology.org/2024.findings-emnlp.591
- Authors' code: https://github.com/javiferran/circuits_languages

## Overview

This project reverse-engineers how **Gemma 2B** solves subject-verb agreement (SVA) in English and Spanish using mechanistic interpretability techniques. The central claim is that a single attention head (L13H7) encodes a language-independent "subject number" direction in the residual stream, and that intervening on this direction in English transfers causally to Spanish.

### Pipeline

```
01_create_dataset.py        # Build English + Spanish SVA contrastive pairs
02_activation_patching.py   # Identify important components via logit difference
03_direct_logit_attribution.py  # DLA: per-component effect on output logits
04_neuron_analysis.py       # Pin down MLP neurons reading the number signal
05_pca_directions.py        # Extract subject-number direction via PCA on L13H7
06_activation_steering.py   # Cross-lingual causal intervention (EN direction → ES)
```

---

## Setup

### Requirements

- Python 3.11+
- CUDA GPU (A100 or V100 recommended for Gemma 2B)
- HuggingFace account with access to [google/gemma-2b](https://huggingface.co/google/gemma-2b)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### HuggingFace Token

```bash
export HF_TOKEN=your_token_here
huggingface-cli login --token $HF_TOKEN
```

---

## Data

### English
From Arora et al. (2024) via SyntaxGym — subset `agr_sv_num_subj-relc`.
Each example is a **contrastive pair**: a clean sentence (subject agrees with verb) and a corrupted one (subject number flipped).

```
clean:     "The executive that embarrassed the manager has"   → singular verb
corrupted: "The executives that embarrassed the manager"      → plural verb
```

### Spanish
A parallel dataset constructed by GPT-4 following the same structure.
Words tokenized into multiple subwords are excluded to keep logit analysis clean.

```
clean:     "El ingeniero que ayudó al cantante era"    → singular verb
corrupted: "Los ingenieros que ayudaron al cantante"   → plural verb
```

Run `01_create_dataset.py` to produce both datasets under `data/processed/`.

---

## Experiments

### 1. Activation Patching (Script 02)

For each component (residual stream position, attention block, MLP block, individual attention head), substitute activations from the **clean** run into the **corrupted** run and measure the change in logit difference:

```
logit_diff = logit(correct_verb) - logit(incorrect_verb)
patched_effect = (logit_diff_patched - logit_diff_corrupted) /
                 (logit_diff_clean   - logit_diff_corrupted)
```

Expected key finding: **L13H7** dominates at the final token position.

### 2. Direct Logit Attribution (Script 03)

Decompose the logit difference into per-component additive contributions via the unembedding matrix `W_U`:

```
DLA_c = f_c(x̃) · W_U[:, good_verb] - f_c(x̃) · W_U[:, bad_verb]
```

### 3. Neuron Analysis (Script 04)

Identify which MLP neurons read the subject-number signal written by L13H7.
Focus on the gated MLP structure:

```
GMLP(x) = (σ(x W_gate) ⊙ x W_in) W_out
```

Key neurons expected: **2069 in MLP13**, **1138 in MLP17**.

### 4. PCA on L13H7 Output (Script 05)

Apply PCA to the outputs of L13H7 across the dataset (both languages).
The first principal component (PC1) should encode singular vs. plural.
Verify this is language-independent by checking PC1 alignment across EN/ES.

### 5. Activation Steering (Script 06)

Add/subtract `α * PC1_English` to L13H7's output during Spanish forward passes:

```
Attn_L13H7_patched = Attn_L13H7 ± α * PC1_English
```

Measure how often the model flips its predicted verb number as α varies.
This is the key cross-lingual causal evidence.

---

## Results

Outputs are saved to `results/`:
- `results/figures/` — activation patching heatmaps, DLA bar charts, PCA scatter plots, steering accuracy curves
- `results/metrics.json` — numerical summaries

---

## Models

| Model        | HF ID                  | Notes                        |
|--------------|------------------------|------------------------------|
| Gemma 2B     | `google/gemma-2b`      | Primary model                |
| Gemma 7B     | `google/gemma-7b`      | Replication check            |
| Gemma 2 2B   | `google/gemma-2-2b`    | Key head: L19H3 instead      |

---

## References

- Ferrando & Costa-jussà (2024). *On the Similarity of Circuits across Languages.* EMNLP Findings.
- Arora et al. (2024). *SyntaxGym SVA dataset.*
- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.*
- Wang et al. (2022). *Interpretability in the Wild* (IOI circuit; activation patching methodology).
