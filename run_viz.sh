#!/bin/bash
# Generate all visualizations (no GPU needed, loads .npz files)
set -euo pipefail
UV="$HOME/.local/bin/uv"

echo "=== 2D Plots ==="
$UV run python -m circuits.plotting --results-dir results --out-dir results/figures

echo "=== 3D Interactive Visualizations ==="
$UV run python -m circuits.viz3d --results-dir results --out-dir results/figures

echo "=== Done! ==="
ls -la results/figures/
