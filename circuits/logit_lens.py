"""
Logit lens: project residual stream onto unembedding at each layer.

Shows what the model "thinks" the next token is at each layer, revealing
where in the network the correct verb prediction first emerges. Comparing
across languages shows whether different languages form their predictions
at different depths.

Usage:
    uv run python -m circuits.logit_lens --lang en --model gemma-2b
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


def run_logit_lens(
    model: HookedTransformer,
    examples: list,
    device: str,
) -> dict:
    """
    At each layer, project the residual stream at the last position onto W_U
    to get logits, then measure:
    1. Rank of the correct verb token
    2. Logit diff (good - bad verb)
    3. Probability of correct verb

    Returns:
        "logit_diff_by_layer": (n_layers + 1, n_examples) — includes final layer
        "correct_rank_by_layer": (n_layers + 1, n_examples)
        "correct_prob_by_layer": (n_layers + 1, n_examples)
        "mean_logit_diff": (n_layers + 1,) — mean across examples
        "mean_correct_prob": (n_layers + 1,) — mean across examples
    """
    n_layers = model.cfg.n_layers
    W_U = model.W_U  # (d_model, d_vocab)

    all_logit_diff = []
    all_rank = []
    all_prob = []

    for ex in tqdm(examples, desc="Logit lens"):
        mt = is_multi_token_lang(ex.get("lang", "en"))
        tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )

        hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        ld_per_layer = []
        rank_per_layer = []
        prob_per_layer = []

        for layer in range(n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]  # (d_model,)
            logits = resid @ W_U  # (d_vocab,)

            ld = (logits[good_id] - logits[bad_id]).item()
            ld_per_layer.append(ld)

            # Rank of correct token (0 = top prediction)
            sorted_idx = torch.argsort(logits, descending=True)
            rank = (sorted_idx == good_id).nonzero(as_tuple=True)[0].item()
            rank_per_layer.append(rank)

            # Probability of correct token
            probs = torch.softmax(logits, dim=0)
            prob_per_layer.append(probs[good_id].item())

        # Final layer (after all blocks, before final LN — approximate with last resid)
        # Use the actual model output for the final entry
        with torch.no_grad():
            final_logits = model(tokens)
        final_ld = (final_logits[0, -1, good_id] - final_logits[0, -1, bad_id]).item()
        final_probs = torch.softmax(final_logits[0, -1, :], dim=0)
        final_rank = (torch.argsort(final_logits[0, -1, :], descending=True) == good_id).nonzero(as_tuple=True)[0].item()

        ld_per_layer.append(final_ld)
        rank_per_layer.append(final_rank)
        prob_per_layer.append(final_probs[good_id].item())

        all_logit_diff.append(ld_per_layer)
        all_rank.append(rank_per_layer)
        all_prob.append(prob_per_layer)

    logit_diff_arr = np.array(all_logit_diff).T  # (n_layers+1, n_examples)
    rank_arr = np.array(all_rank).T
    prob_arr = np.array(all_prob).T

    return {
        "logit_diff_by_layer": logit_diff_arr,
        "correct_rank_by_layer": rank_arr,
        "correct_prob_by_layer": prob_arr,
        "mean_logit_diff": logit_diff_arr.mean(axis=1),
        "mean_correct_prob": prob_arr.mean(axis=1),
        "mean_correct_rank": rank_arr.mean(axis=1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Logit lens: layer-by-layer prediction analysis"
    )
    parser.add_argument("--lang", choices=["en", "es", "fr", "ru", "tr", "sw", "qu"], required=True)
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=128)
    args = parser.parse_args()

    dataset = load_sva_dataset(f"{args.data_dir}/{args.lang}_sva.jsonl")
    if args.max_examples and len(dataset) > args.max_examples:
        dataset = dataset[:args.max_examples]

    model = load_model(args.model, device=args.device)
    results = run_logit_lens(model, dataset, device=args.device)

    out_path = Path(args.out_dir) / f"logit_lens_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **results)
    print(f"Saved logit lens → {out_path}")

    # Print layer-by-layer summary
    mean_ld = results["mean_logit_diff"]
    mean_prob = results["mean_correct_prob"]
    mean_rank = results["mean_correct_rank"]
    print(f"\nLogit lens ({args.lang}):")
    print(f"{'Layer':>6s} {'LogitDiff':>10s} {'P(correct)':>11s} {'Rank':>8s}")
    for i in range(len(mean_ld)):
        label = f"L{i}" if i < len(mean_ld) - 1 else "final"
        bar = "█" * max(0, int(mean_prob[i] * 30))
        print(f"{label:>6s} {mean_ld[i]:>10.3f} {mean_prob[i]:>10.4f} {mean_rank[i]:>8.0f} {bar}")


if __name__ == "__main__":
    main()
