# CLAUDE.md

## Project Overview

Replication of "On the Similarity of Circuits across Languages" (Ferrando & Costa-jussa, EMNLP Findings 2024). Uses mechanistic interpretability to reverse-engineer how Gemma 2B solves subject-verb agreement (SVA) in English and Spanish.

Core hypothesis: attention head L13H7 in Gemma 2B encodes a language-independent subject-number direction in the residual stream. Intervening on this direction trained from English data causally transfers to Spanish.

Paper: https://aclanthology.org/2024.findings-emnlp.591

## Architecture

Flat package layout — `circuits/` is the only package, no `src/` directory.

| Module | Purpose |
|--------|---------|
| `config.py` | Model registry (gemma-2b/7b/2-2b), path defaults (DATA_DIR, RESULTS_DIR) |
| `data.py` | Template-based SVA contrastive pair generation + single-token filtering |
| `model.py` | HookedTransformer loading via transformer-lens, token helpers |
| `metrics.py` | logit_diff, normalized_patch_effect |
| `patching.py` | Activation patching (denoising) over attention heads, MLP blocks |
| `dla.py` | Direct logit attribution per component |
| `neurons.py` | MLP neuron analysis for layers 13, 17 |
| `pca.py` | PCA on L13H7 activations to extract subject-number direction |
| `steering.py` | Cross-lingual activation steering (EN direction applied to ES) |
| `plotting.py` | Publication figure generation |

Entry points: `uv run python -m circuits.<module>` (each module with a `main()` has CLI args via argparse).

Data flow: `data` -> `patching` -> `dla` -> `neurons` -> `pca` -> `steering` -> `plotting`

Key model config (gemma-2b): 18 layers, 18 heads, d_model=2048, key head (13, 7), key neurons (13, 2069) and (17, 1138).

## Development

- **Package manager**: `uv` at `~/.local/bin/uv` — NOT pip, NOT conda
- **Install deps**: `uv sync`
- **Run anything**: `uv run python -m circuits.<module>`
- **Python version**: 3.11 (pinned in `.python-version`)
- **Tests**: `uv run pytest -v` (tests/ directory, must run on login node, no GPU needed)
- **IMPORTANT**: Run `uv run pytest -v` after ANY code change. All tests must pass before committing.
- **Linting**: not configured; no pre-commit hooks

### Key dependencies
- `transformer-lens` for HookedTransformer (circuit analysis)
- `transformers` pinned `<5.0.0` (transformer-lens compatibility — do not upgrade)
- `torch>=2.2.0` with CUDA

## Cluster (BYU RC)

- **Username**: vacl2, **Account**: sdrich
- **Best partition**: `m13h` (H200 150GB), `--qos=gpu`
- **Every sbatch must include**: `--partition=m13h --qos=gpu --account=sdrich`

### CRITICAL: No internet on compute nodes
- Always set `export HF_HUB_OFFLINE=1` in sbatch scripts
- Pre-download models on the login node:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download('google/gemma-2b')
  ```
- `HF_HOME=$HOME/hf_cache` — model cache lives at `$HF_HOME/hub/`
- Do NOT pass `cache_dir=` to `snapshot_download`; let `HF_HOME` handle it

### Other rules
- `module load cuda/12.8.1` before GPU work
- Don't use `srun` with backslash line continuations
- `.env` must have `export` prefix for `source .env` to work (contains HF_TOKEN)
- Check jobs: `squeue -u vacl2`
- Logs go to `logs/` directory

## Running Experiments

### Generate data (login node OK, needs HF tokenizer access)
```bash
uv run python -m circuits.data --lang both --out-dir data/processed --model gemma-2b
```
Use `--no-filter` to skip subword filtering (faster, no model download needed).

### Smoke test
```bash
sbatch test_slurm.sh
```
Verifies GPU access, imports, model loading, and a forward pass (~10 min limit).

### Full pipeline
```bash
mkdir -p logs
sbatch run.sh
```
Runs all 7 steps sequentially (data -> patching -> dla -> neurons -> pca -> steering -> plotting). Requests 1x H200, 64GB RAM, 12h.

### Individual steps (on compute node)
Each module accepts `--lang`, `--model`, `--device`, `--data-dir`, `--out-dir`. See `run.sh` for exact invocations.

## Key Gotchas

- **transformers <5.0.0** is required — transformer-lens breaks on v5+
- **Dataset sizes**: ~6000 EN and ~5800 ES examples from templates. The paper uses subsamples matching their notebook defaults (built into each module's `--max-examples`):
  - Patching: 128, DLA: 256, Neurons: 300, PCA: 256, Steering: full Spanish set
- **English source**: CausalGym (`aryaman/causalgym`, subset `agr_sv_num_subj-relc`, 536 examples) — matches the paper exactly
- **Spanish source**: Paper's GPT-4-curated word lists (100 nouns, 52 RC verbs, 5 prediction verbs), same template/seed — matches the paper exactly
- Patching is O(examples × layers × heads) — the subsampling defaults keep runtime reasonable
- **All verbs are single Gemma subwords** — filtered during data generation
- **Results saved as .npz** files in `results/`; figures in `results/figures/`
- **Paths in config.py** are relative to PROJECT_ROOT (parent of `circuits/`), so always run from repo root
