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

    # Key insight: the final logit is a linear function of the residual stream.
    # So we can decompose the logit difference into additive per-component contributions
    # without any approximation — each head/MLP's output dot the unembedding direction.
    W_U = model.W_U  # (d_model, vocab_size) — the unembedding matrix

    for ex in tqdm(examples, desc="DLA"):
        tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"]
        )

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # The "verb number direction" in vocab space: the difference between the
        # unembedding columns for the correct vs incorrect verb. A component whose
        # output aligns with this direction pushes the model toward the correct verb.
        unembed_dir = W_U[:, good_id] - W_U[:, bad_id]

        for layer in range(n_layers):
            # hook_result is the per-head output in d_head space (before W_O projection)
            head_out = cache[f"blocks.{layer}.attn.hook_result"]
            W_O = model.blocks[layer].attn.W_O
            for head in range(n_heads):
                h_out = head_out[0, -1, head, :]
                # Project from d_head → d_model via W_O so we can dot with unembed_dir
                projected = W_O[head].T @ h_out
                dla = (projected @ unembed_dir).item()
                head_sums[layer, head] += dla

        for layer in range(n_layers):
            # MLP output is already in d_model space, so no projection needed
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
    # Paper uses 256 examples for DLA (DLA_attn_maps.ipynb, train split)
    parser.add_argument("--max-examples", type=int, default=256,
                        help="Max examples to use (default: 256, matching paper)")
    args = parser.parse_args()

    dataset = load_sva_dataset(f"{args.data_dir}/{args.lang}_sva.jsonl")
    if args.max_examples and len(dataset) > args.max_examples:
        dataset = dataset[:args.max_examples]
        print(f"Using {len(dataset)} examples (subsampled from full dataset)")
    model = load_model(args.model, device=args.device)

    results = compute_dla(model, dataset, device=args.device)

    out_path = Path(args.out_dir) / f"dla_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, head_dla=results["head_dla"], mlp_dla=results["mlp_dla"])
    print(f"Saved DLA results → {out_path}")


if __name__ == "__main__":
    main()
