#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Smoke test: Load BLOOM-3b, verify forward pass, test SVA on 2 examples
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=bloom_smoke
#SBATCH --partition=m13l
#SBATCH --qos=gpu
#SBATCH --account=sdrich
#SBATCH --gres=gpu:l40s:1
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

mkdir -p logs

echo "=== BLOOM-3b Smoke Test ==="
$UV run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test 1: Load model
print('\n--- Loading BLOOM-3b via TransformerLens ---')
from circuits.model import load_model
model = load_model('bloom-3b', device='cuda')
print(f'Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, d_model={model.cfg.d_model}')

# Test 2: Forward pass
print('\n--- Forward pass test ---')
tokens = model.to_tokens('The doctor that helped the teacher is')
logits = model(tokens)
print(f'Input shape: {tokens.shape}, Output shape: {logits.shape}')

# Test 3: Check verb token logits
from circuits.model import get_token_id
is_id = get_token_id(model, 'is')
are_id = get_token_id(model, 'are')
last_logits = logits[0, -1]
print(f'P(is)={last_logits[is_id].item():.3f}, P(are)={last_logits[are_id].item():.3f}')
print(f'Model prefers: {\"is\" if last_logits[is_id] > last_logits[are_id] else \"are\"}')

# Test 4: Patching on 2 examples
print('\n--- Patching test (2 examples) ---')
from circuits.data import load_sva_dataset
dataset = load_sva_dataset('data/processed/bloom-3b/en_sva.jsonl')[:2]
from circuits.patching import run_patching
results = run_patching(model, dataset, device='cuda')
print(f'Patching shape: {results[\"head_effects\"].shape}')
# Find top head
import numpy as np
he = results['head_effects']
idx = np.unravel_index(he.argmax(), he.shape)
print(f'Top head (2 examples): L{idx[0]}H{idx[1]} = {he[idx]:.3f}')

# Test 5: RepE on 2 examples
print('\n--- RepE test (2 examples) ---')
from circuits.repe import collect_contrastive_residuals, compute_reading_vectors, compute_signal_profile
diffs, clean, corrupted, labels = collect_contrastive_residuals(model, dataset, device='cuda')
print(f'Diffs shape: {diffs.shape}')
rv, ev = compute_reading_vectors(diffs)
print(f'Reading vectors shape: {rv.shape}')
profile = compute_signal_profile(diffs, rv)
peak_layer = np.argmax(profile['signal_magnitude'])
print(f'Peak signal magnitude at layer {peak_layer}')

print('\n=== All BLOOM-3b smoke tests passed ===')
"
