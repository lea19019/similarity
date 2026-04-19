# Gradient Universality in SVA Circuits

A cross-lingual and cross-model mechanistic interpretability study of **subject–verb agreement (SVA)** in Gemma 2B and BLOOM 3B, across 7 languages and 5 language families.

Replication and extension of Ferrando & Costa-jussà (2024), [*On the Similarity of Circuits across Languages*](https://aclanthology.org/2024.findings-emnlp.591) (EMNLP Findings). Authors' code: https://github.com/javiferran/circuits_languages.

---

## TL;DR

- The paper's headline claim — that attention head **L13H7** in Gemma 2B is the SVA circuit and that its subject-number direction transfers cross-lingually — replicates **cleanly only for Spanish**. Other languages replicate partially or not at all.
- **Late-layer MLPs dominate** in every language and both models: MLP17 in Gemma (94 % depth), MLP28 in BLOOM (93 % depth). This is the single most robust cross-lingual, cross-model finding and is under-emphasised by the paper's attention-head framing.
- **Cross-model circuit convergence** (Gemma ↔ BLOOM) holds only for English and Spanish. French and Swahili diverge in head and depth.
- **Methodological caveat:** weight-alignment interpretability tools (circuit-map / Wanda-style) systematically flag early-layer heads (L0H3, L0H5) as "universal" — these are artefacts of tied-embedding geometry, not SVA circuitry. Causal methods (patching, EAP) rank them near zero. Trust causal over structural when they disagree.

---

## Highlight results

### 1. Activation patching — Gemma 2B, top-1 head per language

| Lang | Top head | Peak effect | Concentration (top1 / top2) | Honest read |
|---|---|---|---|---|
| **Spanish** (n=5684) | **L13H7** | **0.602** | **4.43×** | Cleanest replication of the paper |
| English (n=536)  | L13H7 | 0.371 | 1.55× | Moderate; L17H4 shares load |
| French (n=2496)  | **L13H3** | 0.606 | 1.20× | Different head; distributed circuit |
| Swahili (n=2366) | L13H7 | 0.285 | 1.18× | L14H5 close second |
| Russian (n=260)  | L13H7 | 0.157 | 1.05× | Weak; virtually tied top-2 |
| Quechua (n=60)   | **L14H0** | 0.216 | 1.61× | Different layer |
| Turkish (n=18)   | L2H1 | 0.521 | 1.02× | n=18 — not reliable |

Figure: `results/figures/fig_patching_grid.png`, `results/figures/fig_patching_top1.png`.

### 2. Cross-lingual steering — EN direction → target language

PC1 extracted from Gemma L13H7 on English only (50 examples, cluster separation 3.29), injected into target languages as α·PC1.

| Target  | Peak flip | Best α | Flip @ α=10 | Character |
|---|---|---|---|---|
| **Spanish** | **0.66** | **10** | **0.66** | Clean low-dose response |
| Swahili | 0.64 | 50 | 0.16 | Saturation-only (needs max α) |
| French  | 0.45 | 50 | 0.04 | Flat until α=30 |
| Russian | 0.40 | 50 | 0.14 | Monotonic but modest |
| Turkish | 0.33 | 50 | 0.00 | Flat — unreliable n |
| Quechua | 0.20 | 50 | 0.05 | Weak |

Only Spanish shows a clean causal handle at moderate α; other languages require maximum injection. Figure: `results/figures/fig_steering_curves.png`, `fig_steering_peaks_gemma.png`.

### 3. Cross-model patching — Gemma vs BLOOM

| Lang | Gemma top | Depth | BLOOM top | Depth | Match? |
|---|---|---|---|---|---|
| English | L13H7 | 76 % | L23H15 | 79 % | ✓ |
| Spanish | L13H7 | 76 % | L23H15 | 79 % | ✓ |
| French  | L13H3 | 76 % | L29H12 | 100 % | ✗ (24 % gap) |
| Swahili | L13H7 | 76 % | L20H30 |  69 % | ✗ ( 7 % gap) |

Head-level cross-model convergence holds only for the two languages both models were extensively trained on. Figure: `results/figures/fig_cross_model_depth.png`.

### 4. Edge Attribution Patching — MLP dominance (the most robust finding)

| Model | Top EAP component (every language tested) | Relative depth | Dominance over top attention head |
|---|---|---|---|
| Gemma 2B | **MLP17** | 94 % | 3–10× |
| BLOOM 3B | **MLP28** | 93 % | 3–10× |

Figures: `results/figures/fig_eap_top5.png`, `fig_eap_mlp.png`, `fig_eap_vs_weight.png`.

### 5. Weight-level analysis — methodological caveat

Weight-alignment (`circuit_map`) ranks **L0H5 / L0H3 universally #1** across all 7 languages (cross-lingual task-direction cosine 0.99+). But:

- Within a single language, L0H5, L0H3, L0H2 have cos 0.99 with each other (same operation, not SVA-specific).
- L13H7 has cos ≈ 0.2 with other L13 heads (a real unique computation).
- No L0 head appears in the top-5 by causal EAP for any language.
- Gemma uses tied embeddings (W_U ≈ W_Eᵀ) — early-layer OV weights overlap with any verb-unembedding direction by construction.

**Conclusion:** the "L0 universality" is a measurement artefact of tied-embedding geometry. Weight-alignment tools should not be trusted on early layers without a causal sanity check. Figure: `results/figures/fig_eap_vs_weight.png`, `fig_l0_redundancy.png`.

---

## Repository layout

```
circuits/               # Source package (see table below)
data/processed/         # Generated SVA datasets (gitignored; 8 MB)
results/                # Raw experiment outputs (gitignored .npz)
  AUDIT.txt             # Ground-truth dump of every raw number
  cross_model_*.json    # Cross-model JSON summaries
  figures/              # Publication figures (PNG + interactive HTML)
scripts/                # Helper scripts
  slurm/                # SLURM batch scripts for the BYU RC cluster
  *.py                  # Plotting, audit, sanity-check helpers
tests/                  # pytest suite (runs on login node, no GPU needed)
```

### `circuits/` modules

| Module | Purpose |
|---|---|
| `config.py`        | Model + language registry, path defaults |
| `data.py`          | SVA dataset generation for EN (CausalGym) and ES/FR/RU/SW/TR/QU (templates) |
| `model.py`         | `HookedTransformer` loader (transformer-lens) + token helpers |
| `metrics.py`       | `logit_diff`, `normalized_patch_effect` |
| `patching.py`      | Activation patching (denoising) over heads + MLPs |
| `dla.py`           | Direct logit attribution |
| `neurons.py`       | MLP neuron analysis (layers 13, 17) |
| `pca.py`           | PCA on L13H7 → subject-number direction |
| `steering.py`      | Cross-lingual activation steering |
| `attention.py`     | Attention-pattern analysis |
| `logit_lens.py`    | Per-layer unembedding projection |
| `knockout.py`      | Necessity/sufficiency of discovered circuits |
| `circuit_map.py`   | Weight-level OV/QK decomposition, SVD, task projection |
| `edge_patching.py` | Edge Attribution Patching (gradient-based) |
| `geometry.py`      | Cross-lingual CKA, SVCCA, RSA, Procrustes |
| `repe.py`          | RepE layer scanning (contrastive residual reading vectors) |
| `cross_model.py`   | Gemma ↔ BLOOM flow-topology + CKA |
| `viz3d.py`         | Interactive 3D Plotly visualisations |
| `plotting.py`      | Publication 2D figures |

### Navigating `results/`

- `AUDIT.txt` — the single source of truth. Raw top-k / peaks / effects per language and experiment.
- `patching_<lang>.npz`, `dla_<lang>.npz`, `neurons_<lang>.npz`, `edge_patching_<lang>.npz`, `circuit_map_<lang>.npz`, `attention_<lang>.npz`, `logit_lens_<lang>.npz`, `knockout_<lang>.npz` — per-language outputs for each experiment (Gemma).
- `bloom-3b/` — same structure but for BLOOM (gitignored, 6.3 GB of `.npz`).
- `pca_L13H7.npz` — PC1 direction fit on English, used for cross-lingual steering.
- `steering_<lang>.npz` — flip-rate vs α curves per target language.
- `geometry.npz`, `cross_cka_*.npz`, `cross_model_*.json` — cross-lingual and cross-model similarity matrices.
- `repe_*.npz` — RepE reading vectors and per-layer signal profiles.
- `figures/fig_*.png` — 2D publication figures.
- `figures/viz_*.html` — interactive 3D Plotly visualisations (open in a browser).

---

## Setup

### Requirements

- Python 3.11 (pinned in `.python-version`)
- CUDA-capable GPU for experiments (A100 / H200 tested); CPU-only is fine for tests
- HuggingFace account with access to [`google/gemma-2b`](https://huggingface.co/google/gemma-2b)
- Package manager: [`uv`](https://github.com/astral-sh/uv) (not pip/conda)

### Install

```bash
uv sync                      # main deps
uv sync --group test         # + pytest
```

### HuggingFace credentials

Create `.env` at the repo root:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

The `export` prefix matters — `source .env` is how the SLURM scripts pick it up.

---

## Running experiments

All modules expose a `main()` with argparse. Run from the repo root:

```bash
uv run python -m circuits.<module> [--lang en] [--model gemma-2b] [--device cuda] ...
```

### Generate datasets (needs internet, login node)

```bash
uv run python -m circuits.data --lang all --model gemma-2b
# BLOOM uses its own tokenizer:
uv run python -m circuits.data --lang all --model bloom-3b
```

Datasets land in `data/processed/` (Gemma) or `data/processed/bloom-3b/` (BLOOM).

### Single experiment

```bash
uv run python -m circuits.patching --lang es --model gemma-2b --device cuda
uv run python -m circuits.steering --model gemma-2b --layer 13 --head 7 \
    --pca-path results/pca_L13H7.npz \
    --target-data data/processed/es_sva.jsonl \
    --alphas 0 5 10 20 30 50 --device cuda
```

Defaults for `--max-examples` match the paper's notebooks: 128 (patching), 256 (DLA), 300 (neurons), 50 (PCA, EN-only), all (steering).

### Full pipelines (SLURM)

All scripts live in `scripts/slurm/` and target the BYU RC cluster (`m13h` / `m13l` partitions, `sdrich` account). Adapt `#SBATCH` lines for other clusters.

| Script | What it runs | Time | GPU |
|---|---|---|---|
| `scripts/slurm/test_slurm.sh`        | Smoke test — env, imports, model load, forward pass | ~10 min | 1× H200 |
| `scripts/slurm/run_bloom_smoke.sh`   | BLOOM-specific smoke test                           | ~30 min | 1× L40S |
| `scripts/slurm/run_data_gen.sh`      | Dataset generation (login node, no GPU)             | ~5 min  | — |
| `scripts/slurm/run.sh`               | Paper replication: EN/ES patching → DLA → neurons → PCA → steering → plots | ~12 h | 1× H200 |
| `scripts/slurm/run_all.sh`           | Full 7-language pipeline (adds FR/RU/TR/SW/QU + circuit_map + EAP + geometry + 3D viz) | ~24 h | 1× H200 |
| `scripts/slurm/run_cross_model.sh`   | BLOOM 3B replication + RepE on both models + cross-model CKA/flow | ~24 h | 1× L40S |
| `scripts/slurm/run_viz.sh`           | Regenerate figures (login node, no GPU)             | ~5 min  | — |

Submit a job:

```bash
sbatch scripts/slurm/run.sh
squeue -u $USER
```

### Cluster gotchas (BYU RC)

- No internet on compute nodes. Pre-download models on the login node:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download('google/gemma-2b')
  snapshot_download('bigscience/bloom-3b')
  ```
- Always set `HF_HUB_OFFLINE=1` in sbatch scripts. `HF_HOME=$HOME/hf_cache`.
- `module load cuda/12.8.1` before GPU work.
- Don't use `srun` with backslash line-continuations (causes "No such file" errors).

### Tests

```bash
uv run pytest -v
```

All tests run on the login node and don't need a GPU. Run after any code change.

---

## Dependencies

- `transformer-lens` (HookedTransformer)
- `transformers < 5.0.0` — pinned for transformer-lens compatibility, do **not** upgrade
- `torch >= 2.2.0` with CUDA
- `plotly` for 3D interactive HTML
- `scipy` for Procrustes / distance metrics

### TransformerLens hook names (v2.x)

- Per-head attention output: `hook_z` (shape `(batch, seq, n_heads, d_head)`) — not the old `hook_result`
- Other key hooks: `hook_attn_out`, `hook_mlp_out`, `mlp.hook_post`

---

## References

- Ferrando & Costa-jussà (2024). *On the Similarity of Circuits across Languages.* EMNLP Findings. [aclanthology.org/2024.findings-emnlp.591](https://aclanthology.org/2024.findings-emnlp.591)
- Arora et al. (2024). *CausalGym* — SVA dataset used for English.
- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.* Anthropic.
- Wang et al. (2022). *Interpretability in the Wild* — IOI circuit, activation-patching methodology.
- Zou et al. (2023). *Representation Engineering* — reading-vector methodology used in `circuits/repe.py`.
