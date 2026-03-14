"""
Direct Logit Attribution: decompose logit difference into per-component contributions.

Usage:
    uv run python -m circuits.dla --lang en --model gemma-2b
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from circuits.config import DATA_DIR, RESULTS_DIR
from circuits.model import load_model, tokenize_pair
from circuits.data import load_sva_dataset


def compute_dla(
    model: HookedTransformer,
    examples: list,
    device: str,
) -> dict:
    """
    Compute mean DLA scores across the dataset.

    Returns:
        "head_dla" : (n_layers, n_heads)
        "mlp_dla"  : (n_layers,)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    head_sums = np.zeros((n_layers, n_heads))
    mlp_sums = np.zeros(n_layers)
    count = 0

    W_U = model.W_U

    for ex in tqdm(examples, desc="DLA"):
        tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"]
        )

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        unembed_dir = W_U[:, good_id] - W_U[:, bad_id]

        for layer in range(n_layers):
            head_out = cache[f"blocks.{layer}.attn.hook_result"]
            W_O = model.blocks[layer].attn.W_O
            for head in range(n_heads):
                h_out = head_out[0, -1, head, :]
                projected = W_O[head].T @ h_out
                dla = (projected @ unembed_dir).item()
                head_sums[layer, head] += dla

        for layer in range(n_layers):
            mlp_out = cache[f"blocks.{layer}.hook_mlp_out"]
            m_out = mlp_out[0, -1, :]
            dla = (m_out @ unembed_dir).item()
            mlp_sums[layer] += dla

        count += 1

    return {
        "head_dla": head_sums / count,
        "mlp_dla": mlp_sums / count,
        "n_examples": count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "es"], required=True)
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = load_sva_dataset(f"{args.data_dir}/{args.lang}_sva.jsonl")
    model = load_model(args.model, device=args.device)

    results = compute_dla(model, dataset, device=args.device)

    out_path = Path(args.out_dir) / f"dla_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, head_dla=results["head_dla"], mlp_dla=results["mlp_dla"])
    print(f"Saved DLA results → {out_path}")


if __name__ == "__main__":
    main()
