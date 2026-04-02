#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: full 7-language circuit analysis — targeting cs partition (A100s)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=circuits_7lang
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load cuda/12.8.1

source .env
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1

UV="$HOME/.local/bin/uv"
MODEL="gemma-2b"
DATA_DIR="data/processed"
OUT_DIR="results"
DEVICE="cuda"

LANGS="en es fr ru tr sw qu"

mkdir -p logs "$OUT_DIR" "$OUT_DIR/figures"

echo "=== Step 1: Activation patching ==="
for LANG in $LANGS; do
    echo "--- Patching: $LANG ---"
    $UV run python -m circuits.patching --lang $LANG --model "$MODEL" --device "$DEVICE" \
        --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
done

echo "=== Step 2: Direct logit attribution ==="
for LANG in $LANGS; do
    echo "--- DLA: $LANG ---"
    $UV run python -m circuits.dla --lang $LANG --model "$MODEL" --device "$DEVICE" \
        --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
done

echo "=== Step 3: Neuron analysis ==="
for LANG in $LANGS; do
    echo "--- Neurons: $LANG ---"
    $UV run python -m circuits.neurons --lang $LANG --model "$MODEL" --layers 13 17 \
        --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
done

echo "=== Step 4: Weight-level circuit map ==="
for LANG in $LANGS; do
    echo "--- Circuit map: $LANG ---"
    $UV run python -m circuits.circuit_map --lang $LANG --model "$MODEL" --device "$DEVICE" \
        --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
done

echo "=== Step 5: Edge attribution patching ==="
for LANG in $LANGS; do
    echo "--- EAP: $LANG ---"
    $UV run python -m circuits.edge_patching --lang $LANG --model "$MODEL" --device "$DEVICE" \
        --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
done

echo "=== Step 6: PCA (all 7 languages) ==="
$UV run python -m circuits.pca --lang all --model "$MODEL" --layer 13 --head 7 \
    --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

echo "=== Step 7: Cross-lingual steering ==="
for TARGET in es fr ru tr sw qu; do
    echo "--- Steering: EN -> $TARGET ---"
    $UV run python -m circuits.steering --model "$MODEL" --layer 13 --head 7 \
        --pca-path "$OUT_DIR/pca_L13H7.npz" \
        --target-data "$DATA_DIR/${TARGET}_sva.jsonl" \
        --alphas 0 5 10 20 30 50 \
        --device "$DEVICE" --out-dir "$OUT_DIR"
done

echo "=== Step 8: Cross-lingual geometry comparison ==="
$UV run python -m circuits.geometry --langs $LANGS --model "$MODEL" \
    --device "$DEVICE" --data-dir "$DATA_DIR" --results-dir "$OUT_DIR" --out-dir "$OUT_DIR"

echo "=== Step 9: Plotting ==="
$UV run python -m circuits.plotting --results-dir "$OUT_DIR" --out-dir "$OUT_DIR/figures"
$UV run python -m circuits.viz3d --results-dir "$OUT_DIR" --out-dir "$OUT_DIR/figures"

echo "=== ALL DONE! ==="
