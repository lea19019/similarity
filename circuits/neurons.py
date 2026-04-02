"""
Neuron analysis: identify MLP neurons that read the subject-number signal.

Usage:
    uv run python -m circuits.neurons --lang en --model gemma-2b
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from circuits.config import DATA_DIR, RESULTS_DIR
from circuits.model import load_model, tokenize_pair, is_multi_token_lang
from circuits.data import load_sva_dataset


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
    W_U = model.W_U
    neuron_sums = {layer: np.zeros(model.cfg.d_mlp) for layer in target_layers}
    count = 0

    for ex in tqdm(examples, desc="Neuron DLA"):
        mt = is_multi_token_lang(ex.get("lang", "en"))
        tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )
        unembed_dir = (W_U[:, good_id] - W_U[:, bad_id]).float()

        # hook_post captures the gated MLP activations: σ(x·W_gate) ⊙ (x·W_in),
        # i.e. the post-activation values after the gating nonlinearity
        hook_names = {
            layer: f"blocks.{layer}.mlp.hook_post"
            for layer in target_layers
        }

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=list(hook_names.values()))

        for layer in target_layers:
            # post: per-neuron activation values after gating, shape (d_mlp,)
            post = cache[hook_names[layer]][0, -1, :]
            W_out = model.blocks[layer].mlp.W_out
            # w_proj: each neuron's W_out row dotted with the verb-number direction
            # — how much each neuron's output column aligns with the decision direction
            w_proj = W_out @ unembed_dir
            # Neuron DLA = activation * alignment. This decomposes the MLP's total
            # DLA into individual neuron contributions (exact, not an approximation).
            neuron_dla = (post * w_proj).detach().cpu().numpy()
            neuron_sums[layer] += neuron_dla

        count += 1

    return {layer: sums / count for layer, sums in neuron_sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "es", "fr", "ru", "tr", "sw", "qu"], required=True)
    parser.add_argument("--model", default="gemma-2b")
    # Layers 13 and 17 are the defaults because they contain the neurons with
    # highest DLA for SVA in gemma-2b (neuron 2069@MLP13, neuron 1138@MLP17)
    parser.add_argument("--layers", type=int, nargs="+", default=[13, 17],
                        help="MLP layers to analyse (default: 13 17)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Print top-k neurons by absolute DLA")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    # Paper uses 300 examples for neuron analysis (components_neurons.ipynb, train split)
    parser.add_argument("--max-examples", type=int, default=300,
                        help="Max examples to use (default: 300, matching paper)")
    args = parser.parse_args()

    dataset = load_sva_dataset(f"{args.data_dir}/{args.lang}_sva.jsonl")
    if args.max_examples and len(dataset) > args.max_examples:
        dataset = dataset[:args.max_examples]
        print(f"Using {len(dataset)} examples (subsampled from full dataset)")
    model = load_model(args.model, device=args.device)

    neuron_dla = compute_neuron_dla(
        model, dataset, target_layers=args.layers, device=args.device
    )

    out_path = Path(args.out_dir) / f"neurons_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{f"layer_{l}": v for l, v in neuron_dla.items()})
    print(f"Saved neuron DLA → {out_path}")

    for layer, scores in neuron_dla.items():
        top_idx = np.argsort(np.abs(scores))[::-1][:args.top_k]
        print(f"\nLayer {layer} top-{args.top_k} neurons:")
        for rank, idx in enumerate(top_idx):
            print(f"  #{rank+1:2d}  neuron {idx:5d}  DLA={scores[idx]:+.4f}")


if __name__ == "__main__":
    main()
