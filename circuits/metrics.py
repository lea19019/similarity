"""Logit difference and patching metrics."""
import torch


def logit_diff(
    logits: torch.Tensor,
    good_token_ids: torch.Tensor,
    bad_token_ids: torch.Tensor,
    pos: int = -1,
) -> torch.Tensor:
    """
    Logit difference between the correct and incorrect verb continuation
    at position `pos` (default: last token).
    """
    last_logits = logits[:, pos, :]
    good = last_logits.gather(1, good_token_ids.unsqueeze(1)).squeeze(1)
    bad = last_logits.gather(1, bad_token_ids.unsqueeze(1)).squeeze(1)
    # Positive means the model prefers the grammatically correct verb form
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
    # Maps the raw logit diff onto a [0,1] scale where 0 = corrupted baseline
    # and 1 = clean baseline, so values near 1 indicate the patched component
    # fully restores correct behavior
    return (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)
