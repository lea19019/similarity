#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Per-weight circuit map for all 7 languages (updated code with weight-level importance)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=weight_map
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load cuda/12.8.1
source .env
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
UV="$HOME/.local/bin/uv"

echo "=== Per-weight circuit map (all 7 languages) ==="
for LANG in en es fr ru tr sw qu; do
    echo "--- Circuit map: $LANG ---"
    $UV run python -m circuits.circuit_map --lang $LANG --model gemma-2b --device cuda
done

echo "=== Regenerating visualizations ==="
$UV run python -m circuits.plotting --results-dir results --out-dir results/figures
$UV run python -m circuits.viz3d --results-dir results --out-dir results/figures

echo "=== Done ==="
