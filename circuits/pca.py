"""
PCA on attention head outputs to extract subject-number direction.

Usage:
    uv run python -m circuits.pca --lang both --model gemma-2b
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformer_lens import HookedTransformer

from circuits.config import DATA_DIR, RESULTS_DIR
from circuits.model import load_model, tokenize_pair
from circuits.data import load_sva_dataset


def collect_head_outputs(
    model: HookedTransformer,
    examples: list,
    layer: int,
    head: int,
    device: str,
) -> tuple:
    """
    Returns:
        vectors : (2 * n_examples, d_head)
        labels  : (2 * n_examples,)  — 0 singular, 1 plural
        langs   : (2 * n_examples,)  — "en" or "es"
    """
    hook_name = f"blocks.{layer}.attn.hook_result"
    vectors, labels, langs = [], [], []

    for ex in tqdm(examples, desc=f"Collecting L{layer}H{head} outputs"):
        # Collect both clean (singular subject, label=0) and corrupted (plural subject,
        # label=1) outputs so PCA can find the direction that separates them
        for use_corrupted, label in [(False, 0), (True, 1)]:
            prompt = ex["corrupted"] if use_corrupted else ex["clean"]
            tokens, _, _ = tokenize_pair(model, prompt, ex["good_verb"], ex["bad_verb"])

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])

            # Extract the head's output vector at the final token position (d_head dims)
            h_out = cache[hook_name][0, -1, head, :].cpu().numpy()
            vectors.append(h_out)
            labels.append(label)
            langs.append(ex["lang"])

    return np.stack(vectors), np.array(labels), np.array(langs)


def fit_pca(vectors: np.ndarray) -> PCA:
    # If PC1 cleanly separates singular vs plural (and does so for both EN and ES
    # when trained on both), then the head uses a shared, language-independent
    # representation of subject number.
    pca = PCA(n_components=min(10, vectors.shape[1]))
    pca.fit(vectors)
    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.3f}")
    return pca


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "es", "both"], default="both")
    parser.add_argument("--model", default="gemma-2b")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--head", type=int, default=7)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--device", default="cuda")
    # Paper uses 256 examples for PCA (directions.ipynb, train split);
    # PC1 for steering was extracted from just 50 English examples
    parser.add_argument("--max-examples", type=int, default=256,
                        help="Max examples per language (default: 256, matching paper)")
    args = parser.parse_args()

    model = load_model(args.model, device=args.device)

    all_vectors, all_labels, all_langs = [], [], []

    langs_to_use = ["en", "es"] if args.lang == "both" else [args.lang]
    for lang in langs_to_use:
        dataset = load_sva_dataset(f"{args.data_dir}/{lang}_sva.jsonl")
        if args.max_examples and len(dataset) > args.max_examples:
            dataset = dataset[:args.max_examples]
            print(f"{lang.upper()}: using {len(dataset)} examples (subsampled)")
        vecs, lbls, lngs = collect_head_outputs(
            model, dataset, args.layer, args.head, device=args.device
        )
        all_vectors.append(vecs)
        all_labels.append(lbls)
        all_langs.append(lngs)

    vectors = np.concatenate(all_vectors)
    labels = np.concatenate(all_labels)
    langs = np.concatenate(all_langs)

    pca = fit_pca(vectors)
    # Project all vectors onto PC1 — the expected subject-number direction
    projections = pca.transform(vectors)[:, 0]

    out_path = Path(args.out_dir) / f"pca_L{args.layer}H{args.head}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save PC1 direction for use in steering experiments
    np.savez(
        out_path,
        pc1=pca.components_[0],
        projections=projections,
        labels=labels,
        langs=langs,
    )
    print(f"Saved PCA results → {out_path}")

    # Verify that PC1 separates singular vs plural — large separation = clean encoding
    sg_proj = projections[labels == 0].mean()
    pl_proj = projections[labels == 1].mean()
    print(f"Mean PC1 projection — singular: {sg_proj:.4f}, plural: {pl_proj:.4f}")
    print(f"Separation: {abs(sg_proj - pl_proj):.4f}")


if __name__ == "__main__":
    main()
