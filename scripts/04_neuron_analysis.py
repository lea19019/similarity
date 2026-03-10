"""
04_neuron_analysis.py

Identify MLP neurons that read the subject-number signal written by L13H7.

Gemma uses gated MLPs:
    GMLP(x) = (σ(x W_gate) ⊙ x W_in) W_out

Each neuron i has pre-activation value:
    n_i = σ(x W_gate[:, i]) * (x W_in[:, i])

The output of neuron i is: n_i * W_out[i, :]

We identify important neurons by their direct contribution to the logit diff:
    neuron_dla_i = n_i * (W_out[i, :] · unembed_dir)

Key neurons expected (Gemma 2B):
    - Neuron 2069 in MLP layer 13  (English-dominant)
    - Neuron 1138 in MLP layer 17  (multilingual / Spanish)

Results saved to results/neurons_{lang}.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from utils import load_model, load_sva_dataset, tokenize_pair


def compute_neuron_dla(
    model: HookedTransformer,
    examples: list,
    target_layers: list,
    device: str,
) -> dict:
    """
    Compute per-neuron DLA in the specified MLP layers.

    Returns:
        neuron_dla: dict mapping layer → np.ndarray of shape (d_mlp,)
    """
    W_U = model.W_U  # (d_model, vocab_size)
    neuron_sums = {layer: np.zeros(model.cfg.d_mlp) for layer in target_layers}
    count = 0

    for ex in tqdm(examples, desc="Neuron DLA"):
        tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"]
        )
        unembed_dir = (W_U[:, good_id] - W_U[:, bad_id]).float()  # (d_model,)

        # Hook names for gated MLP internals — adjust if TransformerLens uses different names
        hook_names = {
            layer: f"blocks.{layer}.mlp.hook_post"   # post-activation neuron values
            for layer in target_layers
        }

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=list(hook_names.values()))

        for layer in target_layers:
            # post shape: (1, seq, d_mlp)
            post = cache[hook_names[layer]][0, -1, :]  # (d_mlp,)

            # W_out: (d_mlp, d_model)
            W_out = model.blocks[layer].mlp.W_out

            # DLA for each neuron: n_i * (W_out[i, :] · unembed_dir)
            w_proj = W_out @ unembed_dir   # (d_mlp,)
            neuron_dla = (post * w_proj).cpu().numpy()
            neuron_sums[layer] += neuron_dla

        count += 1

    return {layer: sums / count for layer, sums in neuron_sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "es"], required=True)
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--layers", type=int, nargs="+", default=[13, 17],
                        help="MLP layers to analyse (default: 13 17)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Print top-k neurons by absolute DLA")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = load_sva_dataset(f"{args.data_dir}/{args.lang}_sva.jsonl")
    model   = load_model(args.model, device=args.device)

    neuron_dla = compute_neuron_dla(
        model, dataset, target_layers=args.layers, device=args.device
    )

    out_path = Path(args.out_dir) / f"neurons_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{f"layer_{l}": v for l, v in neuron_dla.items()})
    print(f"Saved neuron DLA → {out_path}")

    # Print top neurons per layer
    for layer, scores in neuron_dla.items():
        top_idx = np.argsort(np.abs(scores))[::-1][:args.top_k]
        print(f"\nLayer {layer} top-{args.top_k} neurons:")
        for rank, idx in enumerate(top_idx):
            print(f"  #{rank+1:2d}  neuron {idx:5d}  DLA={scores[idx]:+.4f}")


if __name__ == "__main__":
    main()
