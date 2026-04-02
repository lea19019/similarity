"""
Wanda-style activation × weight importance scoring.

Computes importance = |W| × ||X|| for each weight, where X is the input
activation norm collected during inference. This accounts for what the model
actually activates, not just static weight structure.

Serves as the standard baseline for comparison against our circuit-derived
weight importance maps.

Usage:
    uv run python -m circuits.wanda --lang en --model gemma-2b
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from circuits.config import DATA_DIR, RESULTS_DIR, LANG_CONFIGS
from circuits.model import load_model, tokenize_pair, is_multi_token_lang
from circuits.data import load_sva_dataset


def collect_activation_norms(
    model: HookedTransformer,
    examples: list,
    device: str,
) -> dict:
    """
    Collect mean input activation norms at the last token position for
    each attention head's W_V input and each MLP's input.

    Returns:
        "attn_input_norms": (n_layers, d_model) — mean ||x|| at attention input
        "mlp_input_norms": (n_layers, d_model) — mean ||x|| at MLP input
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    attn_norms = np.zeros((n_layers, d_model))
    mlp_norms = np.zeros((n_layers, d_model))
    count = 0

    for ex in tqdm(examples, desc="Collecting activation norms"):
        mt = is_multi_token_lang(ex.get("lang", "en"))
        tokens, _, _ = tokenize_pair(
            model, ex["clean"], ex["good_verb"], ex["bad_verb"], multi_token=mt
        )

        # Cache only resid_post per layer (lighter than resid_pre + resid_mid)
        # resid_post[L-1] is resid_pre[L], and resid_mid ≈ resid_pre + attn_out
        hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for layer in range(n_layers):
            # Attention input ≈ previous layer's resid_post (or embedding for layer 0)
            if layer == 0:
                # For layer 0, approximate with resid_post[0] (close enough)
                attn_in = cache[f"blocks.0.hook_resid_post"][0, -1, :].detach().cpu().numpy()
            else:
                attn_in = cache[f"blocks.{layer-1}.hook_resid_post"][0, -1, :].detach().cpu().numpy()
            attn_norms[layer] += np.abs(attn_in)

            # MLP input ≈ resid_post (includes attention contribution)
            mlp_in = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].detach().cpu().numpy()
            mlp_norms[layer] += np.abs(mlp_in)

        count += 1

    if count > 0:
        attn_norms /= count
        mlp_norms /= count

    return {
        "attn_input_norms": attn_norms,
        "mlp_input_norms": mlp_norms,
    }


def compute_wanda_importance(
    model: HookedTransformer,
    activation_norms: dict,
) -> dict:
    """
    Compute Wanda-style importance: |W| × ||X|| for each weight.

    For W_V[i,j]: importance = |W_V[i,j]| × attn_input_norm[i]
    For W_O[j,k]: importance = |W_O[j,k]| × (head output norm, approximated)
    For MLP W_out[n,k]: importance = |W_out[n,k]| × mlp_input_norm (approximated)

    Returns:
        "wv_wanda": (n_layers, n_heads, d_model, d_head)
        "wo_wanda": (n_layers, n_heads, d_head, d_model)
        "mlp_wanda": (n_layers, d_mlp)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

    attn_norms = activation_norms["attn_input_norms"]  # (n_layers, d_model)
    mlp_norms = activation_norms["mlp_input_norms"]    # (n_layers, d_model)

    wv_wanda = np.zeros((n_layers, n_heads, d_model, d_head))
    wo_wanda = np.zeros((n_layers, n_heads, d_head, d_model))
    mlp_wanda = np.zeros((n_layers, model.cfg.d_mlp))

    with torch.no_grad():
        for layer in tqdm(range(n_layers), desc="Wanda importance"):
            act_norm = torch.tensor(attn_norms[layer], dtype=torch.float32)

            for head in range(n_heads):
                W_V = model.W_V[layer, head].float().cpu()  # (d_model, d_head)
                W_O = model.W_O[layer, head].float().cpu()  # (d_head, d_model)

                # W_V importance: |W_V[i,j]| × |activation[i]|
                wv_imp = W_V.abs() * act_norm.unsqueeze(1)  # (d_model, d_head)
                wv_wanda[layer, head] = wv_imp.numpy()

                # W_O importance: |W_O[j,k]| × (use uniform approximation for head output)
                wo_wanda[layer, head] = W_O.abs().numpy()

            # MLP importance: use W_out norm × input activation
            W_out = model.blocks[layer].mlp.W_out.float().cpu()  # (d_mlp, d_model)
            mlp_act = torch.tensor(mlp_norms[layer], dtype=torch.float32)
            # Per-neuron: sum of |W_out[n,:]| weighted by activation norms
            mlp_wanda[layer] = (W_out.abs() * mlp_act.unsqueeze(0)).sum(dim=1).numpy()

    return {
        "wv_wanda": wv_wanda,
        "wo_wanda": wo_wanda,
        "mlp_wanda": mlp_wanda,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Wanda-style activation × weight importance"
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

    print(f"Collecting activation norms ({args.lang}, {len(dataset)} examples)...")
    act_norms = collect_activation_norms(model, dataset, device=args.device)

    print("Computing Wanda importance...")
    results = compute_wanda_importance(model, act_norms)

    out_path = Path(args.out_dir) / f"wanda_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **results, **act_norms)
    print(f"Saved Wanda importance → {out_path}")

    # Summary stats
    wv = results["wv_wanda"]
    total = wv.size
    thresh = np.percentile(wv.ravel(), 90)
    critical = (wv.ravel() > thresh).sum()
    print(f"\nWanda W_V stats ({args.lang}):")
    print(f"  Total weights: {total:,}")
    print(f"  Top 10% threshold: {thresh:.6f}")
    print(f"  Critical weights: {critical:,}")


if __name__ == "__main__":
    main()
