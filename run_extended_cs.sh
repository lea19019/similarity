#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Extended analyses: Wanda, attention patterns, logit lens, circuit knockout
# All 7 languages
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=circuits_extended
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --account=sdrich
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load cuda/12.8.1
source .env
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
UV="$HOME/.local/bin/uv"
LANGS="en es fr ru tr sw qu"

mkdir -p logs results

echo "=== Step 1: Wanda-style activation × weight importance ==="
for LANG in $LANGS; do
    echo "--- Wanda: $LANG ---"
    $UV run python -m circuits.wanda --lang $LANG --model gemma-2b --device cuda
done

echo "=== Step 2: Attention pattern analysis ==="
for LANG in $LANGS; do
    echo "--- Attention: $LANG ---"
    $UV run python -m circuits.attention --lang $LANG --model gemma-2b --device cuda
done

echo "=== Step 3: Logit lens ==="
for LANG in $LANGS; do
    echo "--- Logit lens: $LANG ---"
    $UV run python -m circuits.logit_lens --lang $LANG --model gemma-2b --device cuda
done

echo "=== Step 4: Circuit knockout validation ==="
for LANG in $LANGS; do
    echo "--- Knockout: $LANG ---"
    $UV run python -m circuits.knockout --lang $LANG --model gemma-2b --device cuda \
        --results-dir results --threshold 0.1
done

echo "=== ALL DONE! ==="
