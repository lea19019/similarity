# On the Similarity of Circuits across Languages

Replication of **"On the Similarity of Circuits across Languages: a Case Study on the Subject-verb Agreement Task"**
by Ferrando & Costa-jussà (EMNLP Findings 2024).

- Paper: https://aclanthology.org/2024.findings-emnlp.591
- Authors' code: https://github.com/javiferran/circuits_languages

## Overview

This project reverse-engineers how **Gemma 2B** solves subject-verb agreement (SVA) in English and Spanish using mechanistic interpretability techniques. The central claim is that a single attention head (L13H7) encodes a language-independent "subject number" direction in the residual stream, and that intervening on this direction in English transfers causally to Spanish.

### Pipeline

```
circuits/data.py        # Build English + Spanish SVA contrastive pairs
circuits/patching.py    # Identify important components via logit difference
circuits/dla.py         # DLA: per-component effect on output logits
circuits/neurons.py     # Pin down MLP neurons reading the number signal
circuits/pca.py         # Extract subject-number direction via PCA on L13H7
circuits/steering.py    # Cross-lingual causal intervention (EN direction → ES)
circuits/plotting.py    # Generate publication figures
```

## Setup

### Requirements

- Python 3.11+
- CUDA GPU (A100 recommended for Gemma 2B)
- HuggingFace account with access to [google/gemma-2b](https://huggingface.co/google/gemma-2b)

### Installation

```bash
uv sync
```

### HuggingFace Token

```bash
# Add to .env:
export HF_TOKEN=your_token_here
```

## Usage

### Generate datasets

```bash
uv run python -m circuits.data --lang both --model gemma-2b
```

### Run individual experiments

```bash
uv run python -m circuits.patching --lang en
uv run python -m circuits.dla --lang en
uv run python -m circuits.neurons --lang en
uv run python -m circuits.pca --lang both
uv run python -m circuits.steering
uv run python -m circuits.plotting
```

### Submit full pipeline (SLURM)

```bash
sbatch run.sh
```

## Data

### English
Template-based generation using curated lists of subjects, relative-clause verbs, objects, and main verb pairs. Examples are filtered to ensure verbs tokenize to a single Gemma subword.

### Spanish
Same template structure with Spanish nouns/verbs. RC verbs agree in number with the subject.

Both datasets are written to `data/processed/` as JSONL.

## Models

| Model      | HF ID               | Key Head |
|------------|----------------------|----------|
| Gemma 2B   | `google/gemma-2b`    | L13H7    |
| Gemma 7B   | `google/gemma-7b`    | TBD      |
| Gemma 2 2B | `google/gemma-2-2b`  | L19H3    |

## References

- Ferrando & Costa-jussà (2024). *On the Similarity of Circuits across Languages.* EMNLP Findings.
- Arora et al. (2024). *SyntaxGym SVA dataset.*
- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.*
- Wang et al. (2022). *Interpretability in the Wild* (IOI circuit; activation patching methodology).
