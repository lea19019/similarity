#!/bin/bash
#SBATCH --job-name=circuits_test
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load cuda/12.8.1

source .env
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1

UV="$HOME/.local/bin/uv"

echo "=== Environment ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python: $($UV run python --version)"
echo "CUDA available:"
$UV run python -c "import torch; print(f'  torch.cuda: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo ""
echo "=== Import test ==="
$UV run python -c "
from circuits.config import MODEL_CONFIGS, DATA_DIR
from circuits.data import load_sva_dataset
from circuits.model import load_model

# Load dataset
en = load_sva_dataset(str(DATA_DIR / 'en_sva.jsonl'))
es = load_sva_dataset(str(DATA_DIR / 'es_sva.jsonl'))
print(f'Datasets: {len(en)} EN, {len(es)} ES')

# Load model
print('Loading gemma-2b...')
model = load_model('gemma-2b', device='cuda')
print(f'Model loaded: {model.cfg.model_name}, {model.cfg.n_layers} layers, device={model.cfg.device}')

# Quick forward pass
import torch
tokens = model.to_tokens('The cat sat on the')
with torch.no_grad():
    logits = model(tokens)
print(f'Forward pass OK: logits shape {tuple(logits.shape)}')
print('=== All good! ===')"
