#!/bin/bash
#SBATCH --job-name=circuit_map_fix
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load cuda/12.8.1
source .env
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
UV="$HOME/.local/bin/uv"

echo "=== Rerunning circuit_map with float32 fix ==="
for LANG in en es fr ru; do
    echo "--- Circuit map: $LANG ---"
    $UV run python -m circuits.circuit_map --lang $LANG --model gemma-2b --device cuda
done
echo "=== Done ==="
