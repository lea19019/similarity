"""
Cross-lingual activation steering: does the English subject-number direction
causally control Spanish verb predictions?

Usage:
    uv run python -m circuits.steering --model gemma-2b
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


def steer_and_measure(
    model: HookedTransformer,
    examples: list,
    pc1: np.ndarray,
    layer: int,
    head: int,
    alpha: float,
    direction: str,
    device: str,
) -> float:
    """
    Intervene on L{layer}H{head} by adding ±alpha*pc1, return flip fraction.
    """
    hook_name = f"blocks.{layer}.attn.hook_z"
    pc1_tensor = torch.tensor(pc1, dtype=torch.float32, device=device)
    # "pos" pushes toward one end of the number direction (e.g. plural),
    # "neg" pushes the other way (singular). Which is which depends on
    # the sign convention learned by PCA.
    sign = +1.0 if direction == "pos" else -1.0

    flips = 0
    total = 0

    for ex in examples:
        mt = is_multi_token_lang(ex.get("lang", "en"))
        tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )
        tokens = tokens.to(device)

        # Get the model's unsteered prediction as baseline
        with torch.no_grad():
            base_logits = model(tokens)
        base_pred = _top1_choice(base_logits, good_id, bad_id)

        def hook_fn(value, hook):
            # Add the EN-derived number direction to the head's output.
            # alpha controls the steering magnitude — larger alpha = stronger intervention.
            value[:, -1, head, :] += sign * alpha * pc1_tensor
            return value

        # The cross-lingual causal claim: if adding the English-derived PC1
        # direction to a Spanish forward pass flips the verb prediction,
        # then this direction is causally relevant across languages.
        with torch.no_grad():
            steered_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_name, hook_fn)]
            )
        steered_pred = _top1_choice(steered_logits, good_id, bad_id)

        if steered_pred != base_pred:
            flips += 1
        total += 1

    return flips / total if total > 0 else 0.0


def _top1_choice(logits: torch.Tensor, good_id: int, bad_id: int) -> int:
    """Return 0 if model prefers the correct (good) verb, 1 otherwise."""
    last = logits[0, -1, :]
    return 0 if last[good_id] > last[bad_id] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--head", type=int, default=7)
    parser.add_argument("--pca-path", default=str(RESULTS_DIR / "pca_L13H7.npz"))
    parser.add_argument("--target-data", default=str(DATA_DIR / "es_sva.jsonl"),
                        help="Path to target language dataset (default: Spanish)")
    parser.add_argument("--target-lang", default=None,
                        help="Target language code for output naming (auto-detected if omitted)")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 5.0, 10.0, 20.0, 30.0, 50.0])
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Max target examples for steering evaluation")
    args = parser.parse_args()

    # Load the PC1 direction extracted from English (or bilingual) head outputs
    pca_data = np.load(args.pca_path)
    pc1 = pca_data["pc1"]

    model = load_model(args.model, device=args.device)
    dataset = load_sva_dataset(args.target_data)
    if args.max_examples and len(dataset) > args.max_examples:
        dataset = dataset[:args.max_examples]
        print(f"Using {len(dataset)} examples for steering")

    # Auto-detect target language from dataset
    target_lang = args.target_lang or dataset[0].get("lang", "target")

    results = {"alphas": [], "flip_rate_pos": [], "flip_rate_neg": []}

    for alpha in tqdm(args.alphas, desc="Sweeping alpha"):
        flip_pos = steer_and_measure(
            model, dataset, pc1, args.layer, args.head, alpha, "pos", args.device
        )
        flip_neg = steer_and_measure(
            model, dataset, pc1, args.layer, args.head, alpha, "neg", args.device
        )
        results["alphas"].append(alpha)
        results["flip_rate_pos"].append(flip_pos)
        results["flip_rate_neg"].append(flip_neg)
        print(f"  α={alpha:.1f}  flip(+)={flip_pos:.3f}  flip(-)={flip_neg:.3f}")

    out_path = Path(args.out_dir) / f"steering_{target_lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{k: np.array(v) for k, v in results.items()})
    print(f"Saved steering results → {out_path}")


if __name__ == "__main__":
    main()
