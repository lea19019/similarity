"""
Circuit knockout validation: verify the identified circuit is complete and sufficient.

Two tests:
1. NECESSITY: Ablate the full circuit (all important heads) simultaneously.
   If SVA accuracy drops to chance (~50%), the circuit is necessary.
2. SUFFICIENCY: Ablate everything EXCEPT the circuit.
   If SVA accuracy is preserved, the circuit is sufficient.

Together these validate that our weight maps capture a real, complete circuit.

Usage:
    uv run python -m circuits.knockout --lang en --model gemma-2b
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from circuits.config import DATA_DIR, RESULTS_DIR
from circuits.model import load_model, tokenize_pair, is_multi_token_lang
from circuits.metrics import logit_diff
from circuits.data import load_sva_dataset


def identify_circuit_heads(
    patching_path: str,
    threshold: float = 0.1,
) -> list:
    """
    Identify circuit heads from patching results.

    A head is "in the circuit" if its normalized patch effect > threshold.

    Returns: list of (layer, head) tuples
    """
    data = np.load(patching_path)
    head_out = data["head_out"]
    circuit = []
    for layer in range(head_out.shape[0]):
        for head in range(head_out.shape[1]):
            if head_out[layer, head] > threshold:
                circuit.append((layer, head))
    return circuit


def run_knockout(
    model: HookedTransformer,
    examples: list,
    circuit_heads: list,
    mode: str,
    device: str,
) -> dict:
    """
    Run circuit knockout experiment.

    Args:
        circuit_heads: list of (layer, head) tuples in the circuit
        mode: "ablate_circuit" (test necessity) or "ablate_complement" (test sufficiency)

    Returns:
        "accuracy": float — fraction where model picks correct verb
        "mean_logit_diff": float — mean logit diff across examples
        "n_examples": int
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    circuit_set = set(circuit_heads)

    correct = 0
    total_ld = 0.0
    count = 0

    for ex in tqdm(examples, desc=f"Knockout ({mode})"):
        mt = is_multi_token_lang(ex.get("lang", "en"))
        tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )

        good_ids = torch.tensor([good_id], device=device)
        bad_ids = torch.tensor([bad_id], device=device)

        # Determine which heads to ablate (zero out their output)
        if mode == "ablate_circuit":
            ablate_set = circuit_set
        else:  # ablate_complement
            ablate_set = {
                (l, h) for l in range(n_layers) for h in range(n_heads)
            } - circuit_set

        def make_hook(layer, head):
            def hook_fn(value, hook):
                # Zero out this head's output at the last position
                value[:, -1, head, :] = 0.0
                return value
            return hook_fn

        hook_pairs = []
        for layer, head in ablate_set:
            hook_pairs.append(
                (f"blocks.{layer}.attn.hook_z", make_hook(layer, head))
            )

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hook_pairs)

        ld = logit_diff(logits, good_ids, bad_ids).item()
        total_ld += ld
        if ld > 0:
            correct += 1
        count += 1

    return {
        "accuracy": correct / count if count > 0 else 0.0,
        "mean_logit_diff": total_ld / count if count > 0 else 0.0,
        "n_examples": count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Circuit knockout validation"
    )
    parser.add_argument("--lang", choices=["en", "es", "fr", "ru", "tr", "sw", "qu"], required=True)
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Patching threshold for circuit membership")
    args = parser.parse_args()

    dataset = load_sva_dataset(f"{args.data_dir}/{args.lang}_sva.jsonl")
    if args.max_examples and len(dataset) > args.max_examples:
        dataset = dataset[:args.max_examples]

    # Load patching results to identify circuit
    patching_path = f"{args.results_dir}/patching_{args.lang}.npz"
    circuit_heads = identify_circuit_heads(patching_path, threshold=args.threshold)
    print(f"Circuit for {args.lang}: {len(circuit_heads)} heads (threshold={args.threshold})")
    for l, h in sorted(circuit_heads):
        print(f"  L{l}H{h}")

    model = load_model(args.model, device=args.device)

    # Baseline: no ablation
    print("\nBaseline (no ablation)...")
    baseline = run_knockout(model, dataset, circuit_heads, mode="none_placeholder", device=args.device)
    # Actually run without hooks for baseline
    base_correct = 0
    base_ld = 0.0
    for ex in tqdm(dataset, desc="Baseline"):
        mt = is_multi_token_lang(ex.get("lang", "en"))
        tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )
        with torch.no_grad():
            logits = model(tokens)
        ld = logit_diff(logits, torch.tensor([good_id], device=args.device),
                        torch.tensor([bad_id], device=args.device)).item()
        base_ld += ld
        if ld > 0:
            base_correct += 1
    n = len(dataset)
    baseline_acc = base_correct / n
    baseline_ld = base_ld / n

    # Test 1: Ablate circuit (test necessity)
    print("\nTest 1: Ablating circuit (testing necessity)...")
    necessity = run_knockout(model, dataset, circuit_heads, "ablate_circuit", args.device)

    # Test 2: Ablate complement (test sufficiency)
    print("\nTest 2: Ablating complement (testing sufficiency)...")
    sufficiency = run_knockout(model, dataset, circuit_heads, "ablate_complement", args.device)

    out_path = Path(args.out_dir) / f"knockout_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
        circuit_heads=np.array(circuit_heads),
        threshold=args.threshold,
        baseline_accuracy=baseline_acc,
        baseline_logit_diff=baseline_ld,
        necessity_accuracy=necessity["accuracy"],
        necessity_logit_diff=necessity["mean_logit_diff"],
        sufficiency_accuracy=sufficiency["accuracy"],
        sufficiency_logit_diff=sufficiency["mean_logit_diff"],
    )
    print(f"\nSaved knockout results → {out_path}")

    print(f"\n{'='*50}")
    print(f"KNOCKOUT RESULTS ({args.lang})")
    print(f"{'='*50}")
    print(f"Circuit size: {len(circuit_heads)} heads")
    print(f"Baseline accuracy:    {baseline_acc:.3f} (logit_diff={baseline_ld:.3f})")
    print(f"Ablate circuit:       {necessity['accuracy']:.3f} (logit_diff={necessity['mean_logit_diff']:.3f})")
    print(f"Ablate complement:    {sufficiency['accuracy']:.3f} (logit_diff={sufficiency['mean_logit_diff']:.3f})")
    print(f"\nNecessity: {'PASS' if necessity['accuracy'] < baseline_acc * 0.7 else 'WEAK'}")
    print(f"  (circuit ablation drops accuracy by {(baseline_acc - necessity['accuracy'])*100:.1f}%)")
    print(f"Sufficiency: {'PASS' if sufficiency['accuracy'] > baseline_acc * 0.7 else 'WEAK'}")
    print(f"  (complement ablation retains {sufficiency['accuracy']/baseline_acc*100:.1f}% of accuracy)")


if __name__ == "__main__":
    main()
