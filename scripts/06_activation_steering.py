"""
06_activation_steering.py

Cross-lingual causal intervention: does the English subject-number direction
causally control Spanish verb predictions?

Procedure (§4.3 of paper):
1. Load PC1 from the English PCA (extracted in script 05).
2. For each Spanish example, run the model and add ±α * PC1_English
   to L13H7's output at the last token position.
3. Measure whether the predicted verb number flips.
4. Sweep α to produce a steering accuracy curve.

Expected finding: adding PC1_English in the "plural direction" causes the
model to predict the plural Spanish verb (and vice versa), demonstrating
a language-independent causal effect.

Results saved to results/steering.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from utils import load_model, load_sva_dataset, tokenize_pair


def steer_and_measure(
    model: HookedTransformer,
    examples: list,
    pc1: np.ndarray,       # (d_head,)
    layer: int,
    head: int,
    alpha: float,
    direction: str,        # "pos" or "neg"
    device: str,
) -> float:
    """
    For each example, intervene on L{layer}H{head} output by adding
    alpha * pc1 (direction="pos") or -alpha * pc1 (direction="neg"),
    then check if the model's top-1 prediction at the last token flips
    from the base (unsteered) prediction.

    Returns the fraction of examples where the prediction flips.
    """
    hook_name = f"blocks.{layer}.attn.hook_result"
    pc1_tensor = torch.tensor(pc1, dtype=torch.float32, device=device)
    sign = +1.0 if direction == "pos" else -1.0

    flips = 0
    total = 0

    for ex in examples:
        tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"]
        )
        tokens = tokens.to(device)

        # Unsteered prediction
        with torch.no_grad():
            base_logits = model(tokens)
        base_pred = _top1_choice(base_logits, good_id, bad_id)

        # Steered prediction
        def hook_fn(value, hook):
            value[:, -1, head, :] += sign * alpha * pc1_tensor
            return value

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
    """
    Return 0 if the model prefers good_id over bad_id at the last token,
    else return 1.
    """
    last = logits[0, -1, :]
    return 0 if last[good_id] > last[bad_id] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--head",  type=int, default=7)
    parser.add_argument("--pca-path", default="results/pca_L13H7.npz",
                        help="Path to PC1 from script 05 (English PCA)")
    parser.add_argument("--es-data", default="data/processed/es_sva.jsonl")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 5.0, 10.0, 20.0, 30.0, 50.0],
                        help="Steering magnitudes to sweep")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load English PC1
    pca_data = np.load(args.pca_path)
    pc1 = pca_data["pc1"]   # (d_head,) — English subject-number direction

    model   = load_model(args.model, device=args.device)
    dataset = load_sva_dataset(args.es_data)

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

    out_path = Path(args.out_dir) / "steering.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{k: np.array(v) for k, v in results.items()})
    print(f"Saved steering results → {out_path}")


if __name__ == "__main__":
    main()
