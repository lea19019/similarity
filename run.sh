#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job script — BYU RC cluster
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=circuits_languages
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
module load cuda/12.8.1

source .env
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1

UV="$HOME/.local/bin/uv"

MODEL="gemma-2b"
DATA_DIR="data/processed"
OUT_DIR="results"
DEVICE="cuda"

mkdir -p logs "$OUT_DIR" "$OUT_DIR/figures"

# Datasets are pre-generated on the login node (data/processed/*.jsonl).
# No need to regenerate here — compute nodes are offline anyway.
#
# Default --max-examples per step (matching the original paper):
#   patching: 128    (patching.ipynb)
#   DLA:      256    (DLA_attn_maps.ipynb)
#   neurons:  300    (components_neurons.ipynb)
#   PCA:      256    (directions.ipynb)
#   steering: all    (evaluated on full Spanish set)

echo "=== Step 1: Activation patching ==="
$UV run python -m circuits.patching --lang en --model "$MODEL" --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
$UV run python -m circuits.patching --lang es --model "$MODEL" --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== Step 2: Direct logit attribution ==="
$UV run python -m circuits.dla --lang en --model "$MODEL" --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
$UV run python -m circuits.dla --lang es --model "$MODEL" --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== Step 3: Neuron analysis ==="
$UV run python -m circuits.neurons --lang en --model "$MODEL" --layers 13 17 --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
$UV run python -m circuits.neurons --lang es --model "$MODEL" --layers 13 17 --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== Step 4: PCA on L13H7 ==="
$UV run python -m circuits.pca --lang both --model "$MODEL" --layer 13 --head 7 --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== Step 5: Cross-lingual activation steering ==="
$UV run python -m circuits.steering --model "$MODEL" --layer 13 --head 7 --pca-path "$OUT_DIR/pca_L13H7.npz" --es-data "$DATA_DIR/es_sva.jsonl" --alphas 0 5 10 20 30 50 --device "$DEVICE" --out-dir "$OUT_DIR"

echo "=== Step 6: Plot figures ==="
$UV run python -m circuits.plotting --results-dir "$OUT_DIR" --out-dir "$OUT_DIR/figures"

echo "=== Done! Results in $OUT_DIR ==="
