"""Tests for circuits.steering — _top1_choice logic."""
import torch

from circuits.steering import _top1_choice


class TestTop1Choice:
    def test_good_wins(self):
        """When good logit > bad logit, returns 0."""
        logits = torch.tensor([[[0.0, 0.0, 10.0, 3.0]]])  # (1, 1, 4)
        good_id = 2  # logit=10
        bad_id = 3   # logit=3
        assert _top1_choice(logits, good_id, bad_id) == 0

    def test_bad_wins(self):
        """When bad logit > good logit, returns 1."""
        logits = torch.tensor([[[0.0, 0.0, 3.0, 10.0]]])
        good_id = 2
        bad_id = 3
        assert _top1_choice(logits, good_id, bad_id) == 1

    def test_tie_returns_one(self):
        """When equal, good > bad is False, so returns 1."""
        logits = torch.tensor([[[5.0, 5.0]]])
        assert _top1_choice(logits, 0, 1) == 1

    def test_uses_last_position(self):
        """_top1_choice reads logits[0, -1, :] so multi-position should use last."""
        logits = torch.tensor([[[10.0, 1.0], [1.0, 10.0]]])  # (1, 2, 2)
        # Last position: [1.0, 10.0], so good_id=0 (1.0) < bad_id=1 (10.0)
        assert _top1_choice(logits, 0, 1) == 1

    def test_negative_logits(self):
        """Works with negative values."""
        logits = torch.tensor([[[-5.0, -10.0]]])
        # -5 > -10, so good wins
        assert _top1_choice(logits, 0, 1) == 0
