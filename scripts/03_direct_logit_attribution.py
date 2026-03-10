"""
03_direct_logit_attribution.py

Direct Logit Attribution (DLA): decompose the model's logit difference
into per-component additive contributions via the unembedding matrix W_U.

Because each component writes a vector into the residual stream, and the
final logit is a linear read of the residual stream through W_U, we can
attribute the logit difference directly:

    DLA_c = f_c(x̃) · W_U[:, good_verb] - f_c(x̃) · W_U[:, bad_verb]

where f_c(x̃) is the output of component c on the clean input x̃.

We compute DLA for:
  - Each attention head (layer, head) at the last token position
  - Each MLP block at the last token position

Results saved to results/dla_{lang}.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from utils import load_model, load_sva_dataset, tokenize_pair


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
    n_heads  = model.cfg.n_heads

    head_sums = np.zeros((n_layers, n_heads))
    mlp_sums  = np.zeros(n_layers)
    count = 0

    # W_U: (d_model, vocab_size) — the unembedding matrix
    W_U = model.W_U  # shape: (d_model, vocab_size)

    for ex in tqdm(examples, desc="DLA"):
        tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"]
        )

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Direction in vocab space: W_U[:, good] - W_U[:, bad]
        # shape: (d_model,)
        unembed_dir = W_U[:, good_id] - W_U[:, bad_id]

        # Per-head DLA
        for layer in range(n_layers):
            head_out = cache[f"blocks.{layer}.attn.hook_result"]  # (1, seq, n_heads, d_head)
            # Project each head's last-token output onto unembed_dir
            # Need to fold d_head back to d_model via W_O
            # TransformerLens stores W_O as (n_heads, d_head, d_model)
            W_O = model.blocks[layer].attn.W_O  # (n_heads, d_head, d_model)
            for head in range(n_heads):
                h_out = head_out[0, -1, head, :]           # (d_head,)
                projected = W_O[head].T @ h_out            # (d_model,) — attn output in resid space
                dla = (projected @ unembed_dir).item()
                head_sums[layer, head] += dla

        # Per-MLP DLA
        for layer in range(n_layers):
            mlp_out = cache[f"blocks.{layer}.hook_mlp_out"]  # (1, seq, d_model)
            m_out = mlp_out[0, -1, :]                         # (d_model,)
            dla = (m_out @ unembed_dir).item()
            mlp_sums[layer] += dla

        count += 1

    return {
        "head_dla": head_sums / count,
        "mlp_dla":  mlp_sums  / count,
        "n_examples": count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "es"], required=True)
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = load_sva_dataset(f"{args.data_dir}/{args.lang}_sva.jsonl")
    model   = load_model(args.model, device=args.device)

    results = compute_dla(model, dataset, device=args.device)

    out_path = Path(args.out_dir) / f"dla_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, head_dla=results["head_dla"], mlp_dla=results["mlp_dla"])
    print(f"Saved DLA results → {out_path}")


if __name__ == "__main__":
    main()
