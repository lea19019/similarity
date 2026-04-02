"""
Attention pattern analysis: what does each head attend to?

For SVA, the critical question is whether the key heads (e.g., L13H7) attend
from the verb prediction position back to the subject noun, and whether this
attention routing is consistent across languages.

Usage:
    uv run python -m circuits.attention --lang en --model gemma-2b
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


def collect_attention_patterns(
    model: HookedTransformer,
    examples: list,
    device: str,
) -> dict:
    """
    Collect mean attention patterns at the last token position for all heads.

    For each head, extracts the attention row from the last position (where
    the verb is predicted) showing how much it attends to each earlier position.

    Returns:
        "patterns": (n_layers, n_heads, max_seq_len) — mean attention from last pos
        "max_seq_len": int — padded sequence length
        "n_examples": int
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # First pass: find max sequence length
    max_len = 0
    for ex in examples[:10]:
        mt = is_multi_token_lang(ex.get("lang", "en"))
        tokens, _, _ = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )
        max_len = max(max_len, tokens.shape[1])

    # Add padding buffer
    max_len = min(max_len + 5, 64)

    patterns = np.zeros((n_layers, n_heads, max_len))
    count = 0

    for ex in tqdm(examples, desc="Attention patterns"):
        mt = is_multi_token_lang(ex.get("lang", "en"))
        tokens, _, _ = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )

        # Cache patterns in batches of 6 layers to avoid OOM
        seq_len = tokens.shape[1]
        for batch_start in range(0, n_layers, 6):
            batch_end = min(batch_start + 6, n_layers)
            hook_names = [f"blocks.{l}.attn.hook_pattern" for l in range(batch_start, batch_end)]

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=hook_names)

            for layer in range(batch_start, batch_end):
                attn = cache[f"blocks.{layer}.attn.hook_pattern"][0, :, -1, :seq_len]
                attn_np = attn.detach().cpu().numpy()
                patterns[layer, :, :seq_len] += attn_np

            del cache

        count += 1

    if count > 0:
        patterns /= count

    return {
        "patterns": patterns,
        "max_seq_len": max_len,
        "n_examples": count,
    }


def compute_subject_attention_score(
    patterns: np.ndarray,
) -> np.ndarray:
    """
    Compute how much each head attends to early positions (likely the subject).

    For SVA sentences, the subject noun is typically in positions 1-3.
    The attention to these positions indicates whether the head is reading
    the subject number signal.

    Args:
        patterns: (n_layers, n_heads, max_seq_len)

    Returns: (n_layers, n_heads) — fraction of attention on positions 1-3
    """
    # Subject positions: 1-3 (skip BOS at 0)
    subject_attn = patterns[:, :, 1:4].sum(axis=-1)  # (n_layers, n_heads)
    return subject_attn


def main():
    parser = argparse.ArgumentParser(
        description="Attention pattern analysis for SVA"
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

    results = collect_attention_patterns(model, dataset, device=args.device)
    subject_scores = compute_subject_attention_score(results["patterns"])

    out_path = Path(args.out_dir) / f"attention_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
        patterns=results["patterns"],
        subject_attention=subject_scores,
        max_seq_len=results["max_seq_len"],
        n_examples=results["n_examples"],
    )
    print(f"Saved attention patterns → {out_path}")

    # Print heads with highest subject attention
    flat_idx = np.argsort(subject_scores.ravel())[::-1][:10]
    print(f"\nTop 10 heads by subject attention ({args.lang}):")
    for rank, idx in enumerate(flat_idx):
        layer, head = divmod(idx, subject_scores.shape[1])
        print(f"  #{rank+1:2d}  L{layer}H{head}  subject_attn={subject_scores[layer, head]:.4f}")


if __name__ == "__main__":
    main()
