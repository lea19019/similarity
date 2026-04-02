#!/bin/bash
# Generate datasets for Turkish and Swahili on login node (internet required for tokenizer)
set -euo pipefail
source .env
export HF_HOME="$HOME/hf_cache"
UV="$HOME/.local/bin/uv"

echo "=== Generating Turkish dataset ==="
$UV run python -m circuits.data --lang tr --model gemma-2b

echo "=== Generating Swahili dataset ==="
$UV run python -m circuits.data --lang sw --model gemma-2b

echo "=== Done! ==="
ls -la data/processed/
