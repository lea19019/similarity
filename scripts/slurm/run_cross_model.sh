#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: BLOOM 3B (all experiments) + RepE on both models + comparison
#
# Runs the SAME experiments on BLOOM as we ran on Gemma, then compares.
# BLOOM results go to results/bloom-3b/ to avoid overwriting Gemma results.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=cross_model
#SBATCH --partition=m13l
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load cuda/12.8.1
source .env
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
UV="$HOME/.local/bin/uv"

# Languages with good data for both models
BLOOM_LANGS="en es fr sw"
BLOOM_DATA="data/processed/bloom-3b"
GEMMA_DATA="data/processed"
BLOOM_RESULTS="results/bloom-3b"

mkdir -p logs results $BLOOM_RESULTS

# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: BLOOM-3b Patching + DLA (discover key heads)
# ═════════════════════════════════════════════════════════════════════════════
echo "=== Phase 1a: BLOOM-3b Patching ==="
for LANG in $BLOOM_LANGS; do
    echo "--- Patching: BLOOM-3b / $LANG ---"
    $UV run python -m circuits.patching --lang $LANG --model bloom-3b \
        --data-dir $BLOOM_DATA --out-dir $BLOOM_RESULTS --max-examples 128 --device cuda
done

echo "=== Phase 1b: BLOOM-3b DLA ==="
for LANG in $BLOOM_LANGS; do
    echo "--- DLA: BLOOM-3b / $LANG ---"
    $UV run python -m circuits.dla --lang $LANG --model bloom-3b \
        --data-dir $BLOOM_DATA --out-dir $BLOOM_RESULTS --max-examples 256 --device cuda
done

# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Find BLOOM's key head from patching results, then run PCA + steering
# ═════════════════════════════════════════════════════════════════════════════
echo "=== Phase 2: Identify key head and run PCA + Steering ==="

# Extract BLOOM's top head from English patching results
KEY_HEAD=$($UV run python -c "
import numpy as np
d = np.load('$BLOOM_RESULTS/patching_en.npz')
he = d['head_out']
idx = np.unravel_index(he.argmax(), he.shape)
print(f'{idx[0]} {idx[1]}')
")
KEY_LAYER=$(echo $KEY_HEAD | cut -d' ' -f1)
KEY_H=$(echo $KEY_HEAD | cut -d' ' -f2)
echo "BLOOM key head: L${KEY_LAYER}H${KEY_H}"

# PCA on BLOOM's key head (English only, 50 examples, matching paper methodology)
echo "--- PCA: BLOOM-3b L${KEY_LAYER}H${KEY_H} ---"
$UV run python -m circuits.pca --lang en --model bloom-3b \
    --layer $KEY_LAYER --head $KEY_H \
    --data-dir $BLOOM_DATA --out-dir $BLOOM_RESULTS --max-examples 50 --device cuda

# Cross-lingual steering using BLOOM's English PCA direction
PCA_FILE="$BLOOM_RESULTS/pca_L${KEY_LAYER}H${KEY_H}.npz"
for LANG in es fr sw; do
    echo "--- Steering: BLOOM-3b / $LANG ---"
    $UV run python -m circuits.steering --model bloom-3b \
        --layer $KEY_LAYER --head $KEY_H \
        --pca-path $PCA_FILE \
        --target-data $BLOOM_DATA/${LANG}_sva.jsonl \
        --target-lang $LANG \
        --out-dir $BLOOM_RESULTS --device cuda
done

# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: BLOOM-3b Neuron analysis
# ═════════════════════════════════════════════════════════════════════════════
echo "=== Phase 3: BLOOM-3b Neurons ==="
for LANG in $BLOOM_LANGS; do
    echo "--- Neurons: BLOOM-3b / $LANG ---"
    $UV run python -m circuits.neurons --lang $LANG --model bloom-3b \
        --layers $KEY_LAYER 29 \
        --data-dir $BLOOM_DATA --out-dir $BLOOM_RESULTS --max-examples 300 --device cuda
done

# ═════════════════════════════════════════════════════════════════════════════
# Phase 4: BLOOM-3b Edge Attribution Patching
# ═════════════════════════════════════════════════════════════════════════════
echo "=== Phase 4: BLOOM-3b EAP ==="
for LANG in $BLOOM_LANGS; do
    echo "--- EAP: BLOOM-3b / $LANG ---"
    $UV run python -m circuits.edge_patching --lang $LANG --model bloom-3b \
        --data-dir $BLOOM_DATA --out-dir $BLOOM_RESULTS --max-examples 128 --device cuda
done

# ═════════════════════════════════════════════════════════════════════════════
# Phase 5: BLOOM-3b Weight-level circuit analysis
# ═════════════════════════════════════════════════════════════════════════════
echo "=== Phase 5: BLOOM-3b Circuit Map ==="
for LANG in $BLOOM_LANGS; do
    echo "--- Circuit Map: BLOOM-3b / $LANG ---"
    $UV run python -m circuits.circuit_map --lang $LANG --model bloom-3b \
        --data-dir $BLOOM_DATA --out-dir $BLOOM_RESULTS --device cuda
done

# ═════════════════════════════════════════════════════════════════════════════
# Phase 6: BLOOM-3b Attention patterns + Logit lens
# ═════════════════════════════════════════════════════════════════════════════
echo "=== Phase 6a: BLOOM-3b Attention Patterns ==="
for LANG in $BLOOM_LANGS; do
    echo "--- Attention: BLOOM-3b / $LANG ---"
    $UV run python -m circuits.attention --lang $LANG --model bloom-3b \
        --data-dir $BLOOM_DATA --out-dir $BLOOM_RESULTS --max-examples 128 --device cuda
done

echo "=== Phase 6b: BLOOM-3b Logit Lens ==="
for LANG in $BLOOM_LANGS; do
    echo "--- Logit Lens: BLOOM-3b / $LANG ---"
    $UV run python -m circuits.logit_lens --lang $LANG --model bloom-3b \
        --data-dir $BLOOM_DATA --out-dir $BLOOM_RESULTS --max-examples 128 --device cuda
done

# ═════════════════════════════════════════════════════════════════════════════
# Phase 7: BLOOM-3b Circuit Knockout
# ═════════════════════════════════════════════════════════════════════════════
echo "=== Phase 7: BLOOM-3b Knockout ==="
for LANG in $BLOOM_LANGS; do
    echo "--- Knockout: BLOOM-3b / $LANG ---"
    $UV run python -m circuits.knockout --lang $LANG --model bloom-3b \
        --data-dir $BLOOM_DATA --results-dir $BLOOM_RESULTS --out-dir $BLOOM_RESULTS \
        --max-examples 128 --device cuda
done

# ═════════════════════════════════════════════════════════════════════════════
# Phase 8: RepE scanning — BOTH models, all shared languages
# ═════════════════════════════════════════════════════════════════════════════
echo "=== Phase 8: RepE Scanning ==="

echo "--- RepE: Gemma-2b ---"
$UV run python -m circuits.repe --model gemma-2b --langs $BLOOM_LANGS \
    --data-dir $GEMMA_DATA --max-examples 128 --device cuda

echo "--- RepE: BLOOM-3b ---"
$UV run python -m circuits.repe --model bloom-3b --langs $BLOOM_LANGS \
    --data-dir $BLOOM_DATA --max-examples 128 --device cuda

# ═════════════════════════════════════════════════════════════════════════════
# Phase 9: Cross-model flow topology comparison
# ═════════════════════════════════════════════════════════════════════════════
echo "=== Phase 9: Flow Topology Comparison ==="
$UV run python -m circuits.cross_model --model-a gemma-2b --model-b bloom-3b \
    --langs $BLOOM_LANGS --skip-cka --device cuda

# ═════════════════════════════════════════════════════════════════════════════
# Phase 10: Cross-model CKA (requires both models loaded simultaneously)
# ═════════════════════════════════════════════════════════════════════════════
echo "=== Phase 10: Cross-Model CKA ==="
$UV run python -m circuits.cross_model --model-a gemma-2b --model-b bloom-3b \
    --langs en es --data-dir $GEMMA_DATA --max-examples 64 --device cuda

echo "=== All cross-model experiments complete ==="
echo "BLOOM results: $BLOOM_RESULTS/"
echo "RepE profiles: results/repe_*.npz"
echo "Cross-model: results/cross_model_*.json, results/cross_cka_*.npz"
