"""Model loading and token helpers."""
import torch
from transformer_lens import HookedTransformer
from typing import Tuple

from circuits.config import MODEL_CONFIGS


def load_model(model_key: str = "gemma-2b", device: str = "cuda") -> HookedTransformer:
    """Load a HookedTransformer model for circuit analysis."""
    cfg = MODEL_CONFIGS[model_key]
    # center_writing_weights=False, center_unembed=False, fold_ln=False:
    # Preserve the original model weights without TransformerLens post-processing.
    # Centering and LayerNorm folding simplify some analyses but alter the raw
    # activations, which would distort patching and DLA results.
    model = HookedTransformer.from_pretrained(
        cfg["tl_name"],
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=device,
    )
    model.eval()
    return model


def get_token_id(model: HookedTransformer, word: str) -> int:
    """Get the single-token ID for a word. Raises if tokenized into >1 tokens."""
    # Prepend space so SentencePiece treats word as a continuation token,
    # matching the filtering in data.py (_filter_word_pairs).
    ids = model.to_tokens(f" {word}", prepend_bos=False)[0]
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
    bad_id = get_token_id(model, bad_verb)
    return tokens, good_id, bad_id
