#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: full 7-language circuit analysis pipeline
# Single-token: English, Spanish, French, Russian
# Multi-token (first-subword): Turkish, Swahili, Quechua
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

# ── Environment ──────────────────────────────────────────────────────────────
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

# Datasets are pre-generated on the login node (data/processed/*.jsonl).
# EN (536), ES (5684), FR (2496), RU (260), TR (18), SW (2366), QU (60)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Activation patching — causal importance per head
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Step 1: Activation patching ==="
for LANG in $LANGS; do
    echo "--- Patching: $LANG ---"
    $UV run python -m circuits.patching --lang $LANG --model "$MODEL" --device "$DEVICE" \
        --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
done

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Direct logit attribution
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Step 2: Direct logit attribution ==="
for LANG in $LANGS; do
    echo "--- DLA: $LANG ---"
    $UV run python -m circuits.dla --lang $LANG --model "$MODEL" --device "$DEVICE" \
        --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
done

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Neuron analysis
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Step 3: Neuron analysis ==="
for LANG in $LANGS; do
    echo "--- Neurons: $LANG ---"
    $UV run python -m circuits.neurons --lang $LANG --model "$MODEL" --layers 13 17 \
        --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
done

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Weight-level circuit map (OV/QK decomposition, SVD, task projection)
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Step 4: Weight-level circuit map ==="
for LANG in $LANGS; do
    echo "--- Circuit map: $LANG ---"
    $UV run python -m circuits.circuit_map --lang $LANG --model "$MODEL" --device "$DEVICE" \
        --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
done

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Edge Attribution Patching (gradient-based fast circuit discovery)
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Step 5: Edge attribution patching ==="
for LANG in $LANGS; do
    echo "--- EAP: $LANG ---"
    $UV run python -m circuits.edge_patching --lang $LANG --model "$MODEL" --device "$DEVICE" \
        --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
done

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: PCA on L13H7 (all 7 languages)
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Step 6: PCA (all 7 languages) ==="
$UV run python -m circuits.pca --lang all --model "$MODEL" --layer 13 --head 7 \
    --device "$DEVICE" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Cross-lingual steering (EN direction → each target language)
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Step 7: Cross-lingual steering ==="
for TARGET in es fr ru tr sw qu; do
    echo "--- Steering: EN -> $TARGET ---"
    $UV run python -m circuits.steering --model "$MODEL" --layer 13 --head 7 \
        --pca-path "$OUT_DIR/pca_L13H7.npz" \
        --target-data "$DATA_DIR/${TARGET}_sva.jsonl" \
        --alphas 0 5 10 20 30 50 \
        --device "$DEVICE" --out-dir "$OUT_DIR"
done

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: Cross-lingual geometry (CKA, SVCCA, RSA, Procrustes)
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Step 8: Cross-lingual geometry comparison ==="
$UV run python -m circuits.geometry --langs $LANGS --model "$MODEL" \
    --device "$DEVICE" --data-dir "$DATA_DIR" --results-dir "$OUT_DIR" --out-dir "$OUT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: Generate all figures (2D + 3D)
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Step 9: Plotting ==="
$UV run python -m circuits.plotting --results-dir "$OUT_DIR" --out-dir "$OUT_DIR/figures"
$UV run python -m circuits.viz3d --results-dir "$OUT_DIR" --out-dir "$OUT_DIR/figures"

echo "=== ALL DONE! ==="
echo "Results: $OUT_DIR/*.npz"
echo "Figures: $OUT_DIR/figures/"
echo "3D HTML: $OUT_DIR/figures/viz_*.html"
