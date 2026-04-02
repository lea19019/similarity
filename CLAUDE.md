# CLAUDE.md

## Project Overview

Replication and extension of "On the Similarity of Circuits across Languages" (Ferrando & Costa-jussa, EMNLP Findings 2024). Uses mechanistic interpretability to reverse-engineer how Gemma 2B solves subject-verb agreement (SVA) across four languages: English, Spanish, Turkish, and Swahili.

Core hypothesis: attention head L13H7 in Gemma 2B encodes a language-independent subject-number direction in the residual stream. Intervening on this direction trained from English data causally transfers across typologically diverse languages.

- Paper: https://aclanthology.org/2024.findings-emnlp.591
- Authors' code: https://github.com/javiferran/circuits_languages

## Architecture

Flat package layout — `circuits/` is the only package, no `src/` directory.

| Module | Purpose |
|--------|---------|
| `config.py` | Model registry (gemma-2b/7b/2-2b), language registry, path defaults |
| `data.py` | CausalGym (EN) + template-based (ES/TR/SW) SVA dataset generation |
| `model.py` | HookedTransformer loading via transformer-lens, token helpers |
| `metrics.py` | logit_diff, normalized_patch_effect |
| `patching.py` | Activation patching (denoising) over attention heads, MLP blocks |
| `dla.py` | Direct logit attribution per component |
| `neurons.py` | MLP neuron analysis for layers 13, 17 |
| `pca.py` | PCA on L13H7 activations to extract subject-number direction |
| `steering.py` | Cross-lingual activation steering (EN direction applied to target lang) |
| `circuit_map.py` | Weight-level OV/QK decomposition, SVD, task projection |
| `edge_patching.py` | Edge Attribution Patching — gradient-based fast circuit discovery |
| `geometry.py` | Cross-lingual geometry comparison (CKA, SVCCA, RSA, Procrustes) |
| `viz3d.py` | Interactive 3D Plotly visualizations (standalone HTML) |
| `plotting.py` | Publication figure generation (2D matplotlib/seaborn) |

Entry points: `uv run python -m circuits.<module>` (each module with a `main()` has CLI args via argparse).

Data flow: `data` → `patching` → `dla` → `neurons` → `pca` → `steering` → `plotting`
Extended flow: `data` → `circuit_map` + `edge_patching` → `geometry` → `viz3d`

Key model config (gemma-2b): 18 layers, 18 heads, d_model=2048, key head (13, 7), key neurons (13, 2069) and (17, 1138).

## Development

- **Package manager**: `uv` at `~/.local/bin/uv` — NOT pip, NOT conda
- **Install deps**: `uv sync` (add `--group test` for pytest)
- **Run anything**: `uv run python -m circuits.<module>`
- **Python version**: 3.11 (pinned in `.python-version`)
- **Tests**: `uv run pytest -v` — tests/ directory, runs on login node, no GPU needed
- **IMPORTANT**: Run `uv run pytest -v` after ANY code change. All tests must pass before committing.
- **Linting**: not configured; no pre-commit hooks

### Key dependencies
- `transformer-lens` for HookedTransformer (circuit analysis)
- `transformers` pinned `<5.0.0` (transformer-lens compatibility — do not upgrade)
- `torch>=2.2.0` with CUDA
- `plotly` for 3D interactive visualizations
- `scipy` for Procrustes alignment, distance metrics

### TransformerLens hook names (v2.x)
The per-head attention output hook is `hook_z` (NOT `hook_result` from older versions).
Shape: `(batch, seq, n_heads, d_head)`. This is used in patching, DLA, PCA, and steering.
Other key hooks: `hook_attn_out`, `hook_mlp_out`, `mlp.hook_post`.

## Datasets

EN/ES match the original paper exactly. TR/SW are new additions for cross-lingual extension:

- **English**: CausalGym (`aryaman/causalgym`, subset `agr_sv_num_subj-relc`), filtered to 6-word sentences → 536 examples
- **Spanish**: Paper's GPT-4-curated word lists (100 nouns, 52 RC verbs, 5 prediction verbs), template + seed=10 → 5,684 examples
- **Turkish**: Template-based (75 nouns, 38 RC verbs, 5 pred verbs), Turkic family, SOV order, agglutinative
- **Swahili**: Template-based (41 nouns, 30 RC verbs, 5 pred verbs), Bantu family, SVO, prefix agreement
- All verbs are single Gemma subwords (filtered during generation)
- Datasets are pre-generated on login node and saved to `data/processed/`
- Generate new languages: `uv run python -m circuits.data --lang all`

### `--max-examples` defaults (matching paper's notebooks)
| Step | Default | Paper source |
|------|---------|-------------|
| Patching | 128 | patching.ipynb |
| DLA | 256 | DLA_attn_maps.ipynb |
| Neurons | 300 | components_neurons.ipynb |
| PCA | 256 | directions.ipynb |
| Steering | all | full Spanish set |

## Cluster (BYU RC)

- **Username**: vacl2, **Account**: sdrich
- **Best partition**: `m13h` (H200 150GB), `--qos=gpu`
- **Every sbatch must include**: `--partition=m13h --qos=gpu --account=sdrich`
- **dw partition**: user does NOT have `dw87` QOS — do not use

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
- Don't use `srun` with backslash line continuations — causes "No such file" errors
- `.env` must have `export` prefix for `source .env` to work (contains HF_TOKEN)
- Check jobs: `squeue -u vacl2`
- Logs go to `logs/` directory

## Running Experiments

### Generate data (login node, needs internet for CausalGym download)
```bash
source .env && export HF_HOME=$HOME/hf_cache
uv run python -m circuits.data --lang both --model gemma-2b
```

### Smoke test (GPU)
```bash
sbatch test_slurm.sh
```
Verifies GPU access, imports, model loading, and a forward pass (~10 min).

### End-to-end test (GPU, 2 examples)
```bash
srun --partition=m13h --qos=gpu --account=sdrich --gres=gpu:h200:1 --mem=32G --time=00:15:00 \
  bash -c "source .env && export HF_HOME=\$HOME/hf_cache && export HF_HUB_OFFLINE=1 && module load cuda/12.8.1 && ~/.local/bin/uv run python -m circuits.patching --lang en --max-examples 2 --out-dir /tmp/test"
```

### Full pipeline (original EN/ES)
```bash
mkdir -p logs
sbatch run.sh
```
Runs patching → DLA → neurons → PCA → steering → plotting. Requests 1x H200, 64GB RAM, 12h.

### Extended pipeline (all 4 languages + weight analysis)
```bash
# Step 1: Generate TR/SW data on login node (needs internet)
bash run_data_gen.sh

# Step 2: Run all GPU experiments (24h, 1x H200)
sbatch run_extended.sh

# Step 3: Generate visualizations on login node (no GPU)
bash run_viz.sh
```
Runs original pipeline for TR/SW, then circuit_map + edge_patching + geometry for all 4 languages, plus 3D interactive visualizations.

## Key Gotchas

- **transformers <5.0.0** is required — transformer-lens breaks on v5+
- **hook_z not hook_result** — TransformerLens 2.x renamed the per-head output hook
- **`.detach()` before `.numpy()`** — required for tensors that track gradients (neurons, PCA)
- **PCA n_components** must be ≤ min(n_samples, n_features) — capped in `fit_pca()`
- **Results saved as .npz** files in `results/`; figures in `results/figures/`
- **Paths in config.py** are relative to PROJECT_ROOT (parent of `circuits/`), so always run from repo root
- **Datasets are gitignored** — regenerate with `circuits.data` if missing
