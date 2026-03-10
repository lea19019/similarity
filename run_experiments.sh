#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job script — run the full replication pipeline on a GPU cluster.
# Adjust partition, account, and module names for your specific cluster.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=circuits_languages
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
module load python/3.11   # adjust to your cluster's module system
source .venv/bin/activate

export HF_TOKEN="${HF_TOKEN:?Need HF_TOKEN env var}"
export HF_HOME="$SCRATCH/hf_cache"     # cache HF models on scratch storage
export PYTHONPATH="$PWD/scripts:$PYTHONPATH"

MODEL="gemma-2b"
DATA_DIR="data/processed"
OUT_DIR="results"
DEVICE="cuda"

mkdir -p logs "$DATA_DIR" "$OUT_DIR" "$OUT_DIR/figures"

echo "=== Step 1: Create datasets ==="
python scripts/01_create_dataset.py --lang both --out-dir "$DATA_DIR"

echo "=== Step 2: Activation patching ==="
python scripts/02_activation_patching.py --lang en --model "$MODEL" --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
python scripts/02_activation_patching.py --lang es --model "$MODEL" --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== Step 3: Direct logit attribution ==="
python scripts/03_direct_logit_attribution.py --lang en --model "$MODEL" --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
python scripts/03_direct_logit_attribution.py --lang es --model "$MODEL" --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== Step 4: Neuron analysis ==="
python scripts/04_neuron_analysis.py --lang en --model "$MODEL" --layers 13 17 --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
python scripts/04_neuron_analysis.py --lang es --model "$MODEL" --layers 13 17 --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== Step 5: PCA on L13H7 ==="
python scripts/05_pca_directions.py --lang both --model "$MODEL" --layer 13 --head 7 --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== Step 6: Cross-lingual activation steering ==="
python scripts/06_activation_steering.py \
    --model "$MODEL" \
    --layer 13 --head 7 \
    --pca-path "$OUT_DIR/pca_L13H7.npz" \
    --es-data "$DATA_DIR/es_sva.jsonl" \
    --alphas 0 5 10 20 30 50 \
    --device "$DEVICE" \
    --out-dir "$OUT_DIR"

echo "=== Step 7: Plot figures ==="
python scripts/07_plot_results.py --results-dir "$OUT_DIR" --out-dir "$OUT_DIR/figures"

echo "=== Done! Results in $OUT_DIR ==="
