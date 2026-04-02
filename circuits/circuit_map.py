"""
Weight-level circuit analysis: OV/QK decomposition, SVD, and task projection.

Decomposes every attention head's weight matrices into their functional
components, projects them onto the task-relevant unembedding direction,
and builds a full importance map across all layers and heads.

Usage:
    uv run python -m circuits.circuit_map --lang en --model gemma-2b
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from circuits.config import DATA_DIR, RESULTS_DIR
from circuits.model import load_model, get_token_id, get_first_token_id, is_multi_token_lang
from circuits.data import load_sva_dataset


def compute_ov_matrix(model: HookedTransformer, layer: int, head: int) -> torch.Tensor:
    """
    Compute the OV circuit matrix: W_V @ W_O for a given head.

    The OV matrix maps d_model -> d_model through the d_head bottleneck.
    It represents "what information this head writes to the residual stream"
    given what it reads from the residual stream.

    Returns: (d_model, d_model) tensor
    """
    W_V = model.W_V[layer, head].float()  # (d_model, d_head)
    W_O = model.W_O[layer, head].float()  # (d_head, d_model)
    return W_V @ W_O  # (d_model, d_model)


def compute_qk_matrix(model: HookedTransformer, layer: int, head: int) -> torch.Tensor:
    """
    Compute the QK circuit matrix: W_Q @ W_K^T for a given head.

    The QK matrix determines attention patterns — "what token positions
    attend to what other positions" based on residual stream content.

    Returns: (d_model, d_model) tensor
    """
    W_Q = model.W_Q[layer, head].float()  # (d_model, d_head)
    W_K = model.W_K[layer, head].float()  # (d_model, d_head)
    return W_Q @ W_K.T  # (d_model, d_model)


def svd_ov(ov_matrix: torch.Tensor, top_k: int = 10) -> tuple:
    """
    SVD decomposition of OV matrix to find dominant functional directions.

    Each singular value/vector pair represents an independent "channel"
    through the head. If the top singular value dominates, the head
    implements a rank-1 computation (clean, interpretable).

    Returns: (U[:, :top_k], S[:top_k], Vh[:top_k, :]) — truncated SVD
    """
    U, S, Vh = torch.linalg.svd(ov_matrix, full_matrices=False)
    k = min(top_k, S.shape[0])
    return U[:, :k], S[:k], Vh[:k, :]


def compute_task_projection(
    ov_matrix: torch.Tensor,
    unembed_dir: torch.Tensor,
) -> torch.Tensor:
    """
    Project OV matrix onto the task-relevant unembedding direction.

    task_weight = OV @ unembed_dir  (d_model,)

    This vector tells us: "what input directions does this head map
    into the verb-number decision direction?" Its norm measures how
    strongly the head's weights are aligned with the task.

    Args:
        ov_matrix: (d_model, d_model) — the head's OV circuit
        unembed_dir: (d_model,) — normalized verb-number direction

    Returns: (d_model,) task weight vector
    """
    return ov_matrix @ unembed_dir


def compute_mlp_task_projection(
    model: HookedTransformer,
    layer: int,
    unembed_dir: torch.Tensor,
) -> torch.Tensor:
    """
    Project MLP W_out onto the task-relevant unembedding direction.

    Returns: (d_mlp,) — per-neuron alignment with the task direction.
    Neurons with large absolute values here are structurally wired
    to influence the verb-number decision.
    """
    W_out = model.blocks[layer].mlp.W_out.float()  # (d_mlp, d_model)
    return W_out @ unembed_dir  # (d_mlp,)


def compute_unembed_direction(
    model: HookedTransformer,
    examples: list,
    multi_token: bool = False,
) -> torch.Tensor:
    """
    Compute the mean unembedding direction across all verb pairs in the dataset.

    unembed_dir = mean(W_U[:, good_id] - W_U[:, bad_id])

    This is the direction in residual stream space that separates the
    correct verb from the incorrect verb in the output vocabulary.

    Returns: (d_model,) normalized direction vector
    """
    W_U = model.W_U.float()  # (d_model, d_vocab)
    direction = torch.zeros(model.cfg.d_model, dtype=torch.float32, device=W_U.device)

    # Compute the mean unembedding direction.
    # The "clean" sentence always has the SINGULAR subject, so good_verb is
    # the verb that agrees with singular. We need a consistent direction:
    # always singular_verb - plural_verb. Since clean/corrupted can be swapped
    # in the dataset, we canonicalize by always pointing from plural to singular.
    id_fn = get_first_token_id if multi_token else get_token_id

    seen_pairs = set()
    for ex in examples:
        good_id = id_fn(model, ex["good_verb"])
        bad_id = id_fn(model, ex["bad_verb"])
        # Canonicalize: use frozenset so (a,b) and (b,a) are the same pair
        pair_key = frozenset((good_id, bad_id))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        # Determine which verb is singular. In the clean sentence the subject
        # is singular OR plural depending on the random swap. We can't tell
        # from one example. Instead, just pick a canonical direction per pair:
        # always subtract the smaller token ID from the larger.
        id_a, id_b = min(good_id, bad_id), max(good_id, bad_id)
        direction += W_U[:, id_a] - W_U[:, id_b]

    if len(seen_pairs) > 0:
        direction /= len(seen_pairs)
    direction = direction / (direction.norm() + 1e-10)
    return direction


def compute_weight_importance(
    model: HookedTransformer,
    layer: int,
    head: int,
    unembed_dir: torch.Tensor,
) -> dict:
    """
    Compute per-weight importance for W_V and W_O of a single head.

    The task projection is: output = (input @ W_V) @ W_O @ unembed_dir
    So the importance of W_V[i,j] is: |input_direction[i]| * |downstream[j]|
    where downstream[j] = (W_O @ unembed_dir)[j]

    Similarly for W_O[j,k]: importance = |upstream[j]| * |unembed_dir[k]|

    Returns dict with:
        "wv_importance": (d_model, d_head) — per-weight importance of W_V
        "wo_importance": (d_head, d_model) — per-weight importance of W_O
        "wv_stats": dict with summary stats (sparsity, top-k indices)
        "wo_stats": dict with summary stats
    """
    W_V = model.W_V[layer, head].float()  # (d_model, d_head)
    W_O = model.W_O[layer, head].float()  # (d_head, d_model)

    # For W_O: each weight W_O[j,k] contributes W_O[j,k] * unembed_dir[k]
    # scaled by how much activation flows through dimension j
    # Weight importance = |W_O[j,k]| * |unembed_dir[k]|
    wo_imp = (W_O.abs() * unembed_dir.abs().unsqueeze(0))  # (d_head, d_model)

    # For W_V: each weight W_V[i,j] feeds into the j-th dimension of d_head space.
    # The downstream importance of dimension j is ||W_O[j,:] @ unembed_dir||
    downstream = (W_O * unembed_dir.unsqueeze(0)).abs().sum(dim=1)  # (d_head,)
    wv_imp = W_V.abs() * downstream.unsqueeze(0)  # (d_model, d_head)

    return {
        "wv_importance": wv_imp.detach().cpu().numpy(),
        "wo_importance": wo_imp.detach().cpu().numpy(),
    }


def compute_cross_layer_connections(
    model: HookedTransformer,
    unembed_dir: torch.Tensor,
    top_k_heads: int = 10,
) -> dict:
    """
    Compute how information flows between heads across layers through
    the residual stream — the weight-level connection map.

    For head A in layer L and head B in layer L', the connection strength is:
        ||W_O_A @ W_V_B||_task = how much A's output feeds into B's task-relevant input

    Specifically: connection(A→B) = ||unembed_dir @ W_O_A^T @ W_V_B @ W_O_B @ unembed_dir||
    This measures: "if A writes to the residual stream, how much does B read from
    that and project onto the task direction?"

    Returns:
        "connection_matrix": (n_components, n_components) — connection strength
        "component_labels": list of "L{l}H{h}" strings
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # First, compute the task-relevant output direction for each head
    # and the task-relevant input direction for each head
    output_dirs = []  # what each head writes (d_model,)
    input_dirs = []   # what each head reads that matters for the task (d_model,)
    labels = []

    with torch.no_grad():
        for layer in range(n_layers):
            for head in range(n_heads):
                W_V = model.W_V[layer, head].float()
                W_O = model.W_O[layer, head].float()

                # Output direction: what this head writes that matters for the task
                # W_O^T @ (W_O @ unembed_dir) projects back to see what d_head dims
                # carry task info, then W_O projects those to residual stream
                ov = W_V @ W_O
                out_dir = ov @ unembed_dir  # (d_model,) — task-relevant output
                output_dirs.append(out_dir)

                # Input direction: what residual stream directions this head reads
                # that end up contributing to the task
                in_dir = ov.T @ unembed_dir  # (d_model,) — what input matters
                input_dirs.append(in_dir)

                labels.append(f"L{layer}H{head}")

    n_comp = len(labels)
    connection_matrix = np.zeros((n_comp, n_comp))

    # Connection strength: cosine similarity between output of A and input of B
    # (only A→B where A is in an earlier layer)
    for i in range(n_comp):
        for j in range(n_comp):
            li = int(labels[i][1:].split("H")[0])
            lj = int(labels[j][1:].split("H")[0])
            if li >= lj:
                continue  # only forward connections
            # How much does A's output align with B's input?
            out_a = output_dirs[i]
            in_b = input_dirs[j]
            norm_a = out_a.norm()
            norm_b = in_b.norm()
            if norm_a > 1e-10 and norm_b > 1e-10:
                connection_matrix[i, j] = (out_a @ in_b).abs().item() / (norm_a * norm_b).item()

    return {
        "connection_matrix": connection_matrix,
        "component_labels": np.array(labels),
    }


def build_importance_map(
    model: HookedTransformer,
    examples: list,
    svd_top_k: int = 10,
    multi_token: bool = False,
) -> dict:
    """
    Build per-weight importance map across the entire model.

    For each head, computes:
    - Per-weight importance for W_V and W_O (individual parameter sensitivity)
    - Head-level importance score (norm of task projection)
    - SVD spectrum of OV matrix
    - Cross-layer connection strengths (how heads wire to each other)

    Returns dict with numpy arrays:
        "head_importance": (n_layers, n_heads)
        "mlp_importance": (n_layers,)
        "head_task_weights": (n_layers, n_heads, d_model)
        "svd_spectra": (n_layers, n_heads, top_k)
        "wv_importance": (n_layers, n_heads, d_model, d_head) — per-weight W_V
        "wo_importance": (n_layers, n_heads, d_head, d_model) — per-weight W_O
        "mlp_neuron_importance": (n_layers, d_mlp) — per-neuron MLP importance
        "connection_matrix": (n_components, n_components) — cross-layer wiring
        "connection_labels": component labels for the connection matrix
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

    unembed_dir = compute_unembed_direction(model, examples, multi_token=multi_token)

    head_importance = np.zeros((n_layers, n_heads))
    mlp_importance = np.zeros(n_layers)
    head_task_weights = np.zeros((n_layers, n_heads, d_model))
    svd_spectra = np.zeros((n_layers, n_heads, svd_top_k))

    # Per-weight importance arrays
    wv_importance = np.zeros((n_layers, n_heads, d_model, d_head))
    wo_importance = np.zeros((n_layers, n_heads, d_head, d_model))
    mlp_neuron_importance = np.zeros((n_layers, model.cfg.d_mlp))

    with torch.no_grad():
        for layer in tqdm(range(n_layers), desc="Circuit map (weights)"):
            for head in range(n_heads):
                ov = compute_ov_matrix(model, layer, head)
                task_w = compute_task_projection(ov, unembed_dir)

                head_importance[layer, head] = task_w.norm().item()
                head_task_weights[layer, head] = task_w.detach().cpu().numpy()

                _, S, _ = svd_ov(ov, top_k=svd_top_k)
                s_np = S.detach().cpu().numpy()
                svd_spectra[layer, head, :len(s_np)] = s_np

                # Per-weight importance
                w_imp = compute_weight_importance(model, layer, head, unembed_dir)
                wv_importance[layer, head] = w_imp["wv_importance"]
                wo_importance[layer, head] = w_imp["wo_importance"]

            # Per-neuron MLP importance
            mlp_proj = compute_mlp_task_projection(model, layer, unembed_dir)
            mlp_importance[layer] = mlp_proj.norm().item()
            mlp_neuron_importance[layer] = mlp_proj.abs().detach().cpu().numpy()

    # Cross-layer connections
    print("Computing cross-layer connections...")
    connections = compute_cross_layer_connections(model, unembed_dir)

    return {
        "head_importance": head_importance,
        "mlp_importance": mlp_importance,
        "head_task_weights": head_task_weights,
        "svd_spectra": svd_spectra,
        "wv_importance": wv_importance,
        "wo_importance": wo_importance,
        "mlp_neuron_importance": mlp_neuron_importance,
        "connection_matrix": connections["connection_matrix"],
        "connection_labels": connections["component_labels"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Weight-level circuit analysis: OV/QK decomposition and task projection"
    )
    parser.add_argument("--lang", choices=["en", "es", "fr", "ru", "tr", "sw", "qu"], required=True)
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=256,
                        help="Max examples for computing unembed direction")
    parser.add_argument("--svd-top-k", type=int, default=10,
                        help="Number of top singular values to keep")
    args = parser.parse_args()

    dataset = load_sva_dataset(f"{args.data_dir}/{args.lang}_sva.jsonl")
    if args.max_examples and len(dataset) > args.max_examples:
        dataset = dataset[:args.max_examples]
        print(f"Using {len(dataset)} examples for unembed direction")

    model = load_model(args.model, device=args.device)
    mt = is_multi_token_lang(args.lang)
    results = build_importance_map(model, dataset, svd_top_k=args.svd_top_k, multi_token=mt)

    out_path = Path(args.out_dir) / f"circuit_map_{args.lang}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save per-weight importance separately (large files) and head-level together
    np.savez_compressed(out_path, **{
        k: v for k, v in results.items()
        if k not in ("wv_importance", "wo_importance")
    })
    print(f"Saved circuit map → {out_path}")

    # Save per-weight maps separately (these are large: ~150MB per language)
    weight_path = Path(args.out_dir) / f"weight_map_{args.lang}.npz"
    np.savez_compressed(weight_path,
        wv_importance=results["wv_importance"],
        wo_importance=results["wo_importance"],
        mlp_neuron_importance=results["mlp_neuron_importance"],
    )
    print(f"Saved per-weight map → {weight_path}")

    # Print top heads by weight importance
    head_imp = results["head_importance"]
    flat_idx = np.argsort(head_imp.ravel())[::-1][:10]
    print(f"\nTop 10 heads by weight importance ({args.lang}):")
    for rank, idx in enumerate(flat_idx):
        layer, head = divmod(idx, head_imp.shape[1])
        print(f"  #{rank+1:2d}  L{layer}H{head}  importance={head_imp[layer, head]:.4f}")

    # Per-weight sparsity stats
    wv = results["wv_importance"]
    wo = results["wo_importance"]
    total_weights = wv.size + wo.size
    wv_flat = wv.ravel()
    wo_flat = wo.ravel()
    all_flat = np.concatenate([wv_flat, wo_flat])
    thresh_90 = np.percentile(all_flat, 90)
    critical = (all_flat > thresh_90).sum()
    print(f"\nPer-weight stats:")
    print(f"  Total attention weights mapped: {total_weights:,}")
    print(f"  Top 10% threshold: {thresh_90:.6f}")
    print(f"  Weights above threshold: {critical:,} ({critical/total_weights*100:.1f}%)")
    print(f"  Weights safe for aggressive quantization (bottom 90%): {total_weights - critical:,}")

    # Top connections
    conn = results["connection_matrix"]
    labels = results["connection_labels"]
    conn_flat = conn.ravel()
    top_conn_idx = np.argsort(conn_flat)[::-1][:10]
    print(f"\nTop 10 cross-layer connections:")
    for rank, idx in enumerate(top_conn_idx):
        src, dst = divmod(idx, conn.shape[1])
        if conn_flat[idx] > 0:
            print(f"  #{rank+1:2d}  {labels[src]} → {labels[dst]}  strength={conn_flat[idx]:.4f}")


if __name__ == "__main__":
    main()
