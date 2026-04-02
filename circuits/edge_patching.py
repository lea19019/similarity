"""
Edge Attribution Patching (EAP): gradient-based approximation of activation patching.

Instead of O(layers * heads) forward passes per example, EAP uses 2 forward
passes + 1 backward pass to estimate every component's importance simultaneously.

The key insight: the effect of patching component C is approximately
    delta_activation[C] * gradient_of_metric_wrt_activation[C]
This is exact to first order (linear approximation of the patching effect).

Usage:
    uv run python -m circuits.edge_patching --lang en --model gemma-2b
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


def _make_component_labels(n_layers: int, n_heads: int) -> list:
    """Build ordered list of component labels: L0H0, L0H1, ..., MLP0, ..."""
    labels = []
    for layer in range(n_layers):
        for head in range(n_heads):
            labels.append(f"L{layer}H{head}")
    for layer in range(n_layers):
        labels.append(f"MLP{layer}")
    return labels


def compute_eap_scores(
    model: HookedTransformer,
    examples: list,
    device: str,
) -> dict:
    """
    Compute Edge Attribution Patching scores for all components.

    For each example:
    1. Run clean forward pass → cache activations
    2. Run corrupted forward pass with gradient tracking
    3. Backprop through logit_diff to get gradients at each component
    4. Node score = |activation_diff * gradient| summed over dimensions

    Returns:
        "node_scores": (n_components,) — per-node importance
        "head_scores": (n_layers, n_heads) — reshaped head-only scores
        "mlp_scores": (n_layers,) — MLP-only scores
        "component_labels": array of str labels
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_head_components = n_layers * n_heads
    n_components = n_head_components + n_layers

    node_sums = np.zeros(n_components)
    count = 0

    for ex in tqdm(examples, desc="EAP"):
        mt = is_multi_token_lang(ex.get("lang", "en"))
        clean_tokens, good_id, bad_id = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )
        corrupted_tokens, _, _ = tokenize_pair(
            model, ex["corrupted"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )

        good_ids = torch.tensor([good_id], device=device)
        bad_ids = torch.tensor([bad_id], device=device)

        # Step 1: Clean forward pass (no grad) — get reference activations
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens)

        # Step 2: Corrupted forward pass WITH gradients
        # We hook into each component to capture activations and enable grad tracking
        activations = {}

        def make_fwd_hook(name):
            def hook_fn(value, hook):
                activations[name] = value
                value.requires_grad_(True)
                value.retain_grad()
                return value
            return hook_fn

        hook_pairs = []
        for layer in range(n_layers):
            hook_pairs.append(
                (f"blocks.{layer}.attn.hook_z", make_fwd_hook(f"head_{layer}"))
            )
            hook_pairs.append(
                (f"blocks.{layer}.hook_mlp_out", make_fwd_hook(f"mlp_{layer}"))
            )

        corrupted_logits = model.run_with_hooks(
            corrupted_tokens, fwd_hooks=hook_pairs
        )

        # Step 3: Backward pass through logit_diff
        ld = logit_diff(corrupted_logits, good_ids, bad_ids)
        ld.backward()

        # Step 4: Compute node scores = |activation_diff * gradient|
        with torch.no_grad():
            for layer in range(n_layers):
                head_act = activations[f"head_{layer}"]
                if head_act.grad is not None:
                    clean_z = clean_cache[f"blocks.{layer}.attn.hook_z"]
                    for head in range(n_heads):
                        delta = clean_z[0, -1, head, :] - head_act[0, -1, head, :].detach()
                        grad = head_act.grad[0, -1, head, :]
                        comp_idx = layer * n_heads + head
                        node_sums[comp_idx] += (delta * grad).abs().sum().item()

                mlp_act = activations[f"mlp_{layer}"]
                if mlp_act.grad is not None:
                    clean_mlp = clean_cache[f"blocks.{layer}.hook_mlp_out"]
                    delta_mlp = clean_mlp[0, -1, :] - mlp_act[0, -1, :].detach()
                    grad_mlp = mlp_act.grad[0, -1, :]
                    mlp_idx = n_head_components + layer
                    node_sums[mlp_idx] += (delta_mlp * grad_mlp).abs().sum().item()

        count += 1
        model.zero_grad()

    if count == 0:
        raise RuntimeError("No valid examples for EAP.")

    node_scores = node_sums / count

    # Reshape into head and MLP views
    head_scores = node_scores[:n_head_components].reshape(n_layers, n_heads)
    mlp_scores = node_scores[n_head_components:]

    labels = _make_component_labels(n_layers, n_heads)

    return {
        "node_scores": node_scores,
        "head_scores": head_scores,
        "mlp_scores": mlp_scores,
        "component_labels": np.array(labels),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Edge Attribution Patching for fast circuit discovery"
    )
    parser.add_argument("--lang", choices=["en", "es", "fr", "ru", "tr", "sw", "qu"], required=True)
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=128,
                        help="Max examples (default: 128, matching patching)")
    args = parser.parse_args()

    dataset = load_sva_dataset(f"{args.data_dir}/{args.lang}_sva.jsonl")
    if args.max_examples and len(dataset) > args.max_examples:
        dataset = dataset[:args.max_examples]
        print(f"Using {len(dataset)} examples")

    model = load_model(args.model, device=args.device)
    results = compute_eap_scores(model, dataset, device=args.device)

    out_path = Path(args.out_dir) / f"edge_patching_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **results)
    print(f"Saved EAP results → {out_path}")

    # Print top components
    scores = results["node_scores"]
    labels = results["component_labels"]
    top_idx = np.argsort(scores)[::-1][:15]
    print(f"\nTop 15 components by EAP score ({args.lang}):")
    for rank, idx in enumerate(top_idx):
        print(f"  #{rank+1:2d}  {labels[idx]:8s}  score={scores[idx]:.4f}")


if __name__ == "__main__":
    main()
