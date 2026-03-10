"""
Shared utilities for circuit analysis experiments.
"""
import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Tuple


# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "gemma-2b": {
        "hf_name": "google/gemma-2b",
        "tl_name": "gemma-2b",          # TransformerLens model name
        "n_layers": 18,
        "n_heads": 18,
        "d_model": 2048,
        "key_head": (13, 7),            # (layer, head) — L13H7
        "key_neurons": [(13, 2069), (17, 1138)],
    },
    "gemma-7b": {
        "hf_name": "google/gemma-7b",
        "tl_name": "gemma-7b",
        "n_layers": 28,
        "n_heads": 16,
        "d_model": 3072,
        "key_head": None,               # fill in after replicating
        "key_neurons": [],
    },
    "gemma-2-2b": {
        "hf_name": "google/gemma-2-2b",
        "tl_name": "gemma-2-2b",
        "n_layers": 26,
        "n_heads": 8,
        "d_model": 2304,
        "key_head": (19, 3),            # L19H3
        "key_neurons": [],
    },
}


def load_model(model_key: str = "gemma-2b", device: str = "cuda") -> HookedTransformer:
    """Load a HookedTransformer model for circuit analysis."""
    cfg = MODEL_CONFIGS[model_key]
    model = HookedTransformer.from_pretrained(
        cfg["tl_name"],
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=device,
    )
    model.eval()
    return model


# ── Logit difference ────────────────────────────────────────────────────────────
def logit_diff(
    logits: torch.Tensor,          # (batch, seq, vocab)
    good_token_ids: torch.Tensor,  # (batch,)
    bad_token_ids: torch.Tensor,   # (batch,)
    pos: int = -1,
) -> torch.Tensor:
    """
    Logit difference between the correct and incorrect verb continuation
    at position `pos` (default: last token).
    """
    last_logits = logits[:, pos, :]  # (batch, vocab)
    good = last_logits.gather(1, good_token_ids.unsqueeze(1)).squeeze(1)
    bad  = last_logits.gather(1, bad_token_ids.unsqueeze(1)).squeeze(1)
    return good - bad


def normalized_patch_effect(
    patched_ld: float,
    clean_ld: float,
    corrupted_ld: float,
) -> float:
    """
    Normalize patching effect to [0, 1]:
      0 = no recovery (corrupted baseline)
      1 = full recovery (clean baseline)
    """
    return (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)


# ── Token helpers ───────────────────────────────────────────────────────────────
def get_token_id(model: HookedTransformer, word: str) -> int:
    """Get the single-token ID for a word. Raises if tokenized into >1 tokens."""
    ids = model.to_tokens(word, prepend_bos=False)[0]
    if ids.shape[0] != 1:
        raise ValueError(f"'{word}' tokenizes into {ids.shape[0]} tokens, expected 1.")
    return ids[0].item()


def tokenize_pair(
    model: HookedTransformer,
    prompt: str,
    good_verb: str,
    bad_verb: str,
) -> Tuple[torch.Tensor, int, int]:
    """Tokenize a prompt and return (token_ids, good_verb_id, bad_verb_id)."""
    tokens = model.to_tokens(prompt)
    good_id = get_token_id(model, good_verb)
    bad_id  = get_token_id(model, bad_verb)
    return tokens, good_id, bad_id


# ── Dataset loading ─────────────────────────────────────────────────────────────
def load_sva_dataset(path: str) -> List[dict]:
    """
    Load a processed SVA dataset (JSONL).
    Each line: {"clean": str, "corrupted": str, "good_verb": str, "bad_verb": str, "lang": str}
    """
    import json
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples
