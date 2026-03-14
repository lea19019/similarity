"""
Activation patching (denoising) to identify causally important model components.

Usage:
    uv run python -m circuits.patching --lang en --model gemma-2b
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from circuits.config import DATA_DIR, RESULTS_DIR
from circuits.model import load_model, tokenize_pair
from circuits.metrics import logit_diff, normalized_patch_effect
from circuits.data import load_sva_dataset


def run_patching(
    model: HookedTransformer,
    examples: list,
    device: str,
) -> dict:
    """
    Run activation patching over all examples and return result arrays.

    Returns a dict with keys:
        "attn_out"    : (n_layers,)         — attention block output, last pos
        "mlp_out"     : (n_layers,)         — MLP block output, last pos
        "head_out"    : (n_layers, n_heads) — per-head output, last pos
    All values are normalized patch effects in [0, 1].
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    attn_sums = np.zeros(n_layers)
    mlp_sums = np.zeros(n_layers)
    head_sums = np.zeros((n_layers, n_heads))
    count = 0

    for ex in tqdm(examples, desc="Patching"):
        clean_tokens, good_id, bad_id = _tokenize_example(model, ex)
        corrupted_tokens, _, _ = _tokenize_example(model, ex, use_corrupted=True)

        good_ids = torch.tensor([good_id], device=device)
        bad_ids = torch.tensor([bad_id], device=device)

        # Cache clean activations — these will be "patched in" to the corrupted run
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)
            corr_logits, _ = model.run_with_cache(corrupted_tokens)

        clean_ld = logit_diff(clean_logits, good_ids, bad_ids).item()
        corr_ld = logit_diff(corr_logits, good_ids, bad_ids).item()

        # Skip if clean and corrupted produce nearly the same logit diff —
        # normalization would divide by ~0, and these examples don't distinguish behavior
        if abs(clean_ld - corr_ld) < 1e-4:
            continue

        # Denoising patching: run on the corrupted input but replace one
        # component's activation with its clean-run value. If this restores
        # correct behavior (normalized_patch_effect → 1), that component
        # causally carries the subject-number signal.
        for layer in range(n_layers):
            for head in range(n_heads):
                patched_ld = _patch_head(
                    model, corrupted_tokens, clean_cache,
                    layer, head, good_ids, bad_ids,
                )
                head_sums[layer, head] += normalized_patch_effect(
                    patched_ld, clean_ld, corr_ld
                )

        for layer in range(n_layers):
            patched_ld = _patch_hook(
                model, corrupted_tokens, clean_cache,
                f"blocks.{layer}.hook_attn_out", -1,
                good_ids, bad_ids,
            )
            attn_sums[layer] += normalized_patch_effect(patched_ld, clean_ld, corr_ld)

        for layer in range(n_layers):
            patched_ld = _patch_hook(
                model, corrupted_tokens, clean_cache,
                f"blocks.{layer}.hook_mlp_out", -1,
                good_ids, bad_ids,
            )
            mlp_sums[layer] += normalized_patch_effect(patched_ld, clean_ld, corr_ld)

        count += 1

    if count == 0:
        raise RuntimeError("No valid examples found.")

    return {
        "head_out": head_sums / count,
        "attn_out": attn_sums / count,
        "mlp_out": mlp_sums / count,
        "n_examples": count,
    }


def _tokenize_example(model, ex: dict, use_corrupted: bool = False):
    prompt = ex["corrupted"] if use_corrupted else ex["clean"]
    return tokenize_pair(model, prompt, ex["good_verb"], ex["bad_verb"])


def _patch_hook(
    model, corrupted_tokens, clean_cache,
    hook_name: str, pos: int,
    good_ids, bad_ids,
) -> float:
    """Patch a full activation (attn_out or mlp_out) at one position."""
    def hook_fn(value, hook):
        # Replace the corrupted activation at this position with the clean one
        if pos == -1:
            value[:, -1, :] = clean_cache[hook_name][:, -1, :]
        else:
            value[:, pos, :] = clean_cache[hook_name][:, pos, :]
        return value

    # run_with_hooks: forward pass that fires hook_fn when the named activation is computed
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
        )
    return logit_diff(patched_logits, good_ids, bad_ids).item()


def _patch_head(
    model, corrupted_tokens, clean_cache,
    layer: int, head: int,
    good_ids, bad_ids,
) -> float:
    """Patch a single attention head's output at the final token position."""
    # hook_result has shape (batch, seq, n_heads, d_head) — the per-head output
    # before summing across heads
    hook_name = f"blocks.{layer}.attn.hook_result"

    def hook_fn(value, hook):
        # Only patch this specific head, leaving all other heads at their corrupted values
        value[:, -1, head, :] = clean_cache[hook_name][:, -1, head, :]
        return value

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
        )
    return logit_diff(patched_logits, good_ids, bad_ids).item()


def save_results(results: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{k: v for k, v in results.items() if isinstance(v, np.ndarray)})
    print(f"Saved patching results → {out_path}")


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

    results = run_patching(model, dataset, device=args.device)
    save_results(results, Path(args.out_dir) / f"patching_{args.lang}.npz")


if __name__ == "__main__":
    main()
