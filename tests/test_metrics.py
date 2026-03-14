"""Tests for circuits.metrics — logit_diff and normalized_patch_effect."""
import pytest
import torch

from circuits.metrics import logit_diff, normalized_patch_effect


class TestLogitDiff:
    def test_basic_positive_diff(self):
        """good logit > bad logit => positive result."""
        logits = torch.tensor([[[1.0, 5.0, 3.0]]])  # (1, 1, 3)
        good_ids = torch.tensor([1])  # logit=5
        bad_ids = torch.tensor([2])   # logit=3
        result = logit_diff(logits, good_ids, bad_ids, pos=-1)
        assert result.item() == pytest.approx(2.0)

    def test_basic_negative_diff(self):
        """good logit < bad logit => negative result."""
        logits = torch.tensor([[[1.0, 2.0, 7.0]]])
        good_ids = torch.tensor([1])
        bad_ids = torch.tensor([2])
        result = logit_diff(logits, good_ids, bad_ids, pos=-1)
        assert result.item() == pytest.approx(-5.0)

    def test_zero_diff(self):
        """Equal logits => zero."""
        logits = torch.tensor([[[4.0, 4.0]]])
        good_ids = torch.tensor([0])
        bad_ids = torch.tensor([1])
        result = logit_diff(logits, good_ids, bad_ids)
        assert result.item() == pytest.approx(0.0)

    def test_pos_selects_token_position(self):
        """pos=0 should select the first position, not last."""
        logits = torch.tensor([[[10.0, 1.0], [1.0, 10.0]]])  # (1, 2, 2)
        good_ids = torch.tensor([0])
        bad_ids = torch.tensor([1])
        # pos=0: logits at position 0 are [10, 1], diff = 10-1 = 9
        result = logit_diff(logits, good_ids, bad_ids, pos=0)
        assert result.item() == pytest.approx(9.0)
        # pos=-1 (last): logits at position 1 are [1, 10], diff = 1-10 = -9
        result_last = logit_diff(logits, good_ids, bad_ids, pos=-1)
        assert result_last.item() == pytest.approx(-9.0)

    def test_batch_dimension(self):
        """Works with batch_size > 1."""
        logits = torch.tensor([
            [[3.0, 1.0]],
            [[1.0, 5.0]],
        ])  # (2, 1, 2)
        good_ids = torch.tensor([0, 0])
        bad_ids = torch.tensor([1, 1])
        result = logit_diff(logits, good_ids, bad_ids)
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(2.0)
        assert result[1].item() == pytest.approx(-4.0)


class TestNormalizedPatchEffect:
    def test_full_recovery(self):
        """patched == clean => 1.0."""
        result = normalized_patch_effect(
            patched_ld=5.0, clean_ld=5.0, corrupted_ld=-2.0
        )
        assert result == pytest.approx(1.0)

    def test_no_recovery(self):
        """patched == corrupted => 0.0."""
        result = normalized_patch_effect(
            patched_ld=-2.0, clean_ld=5.0, corrupted_ld=-2.0
        )
        assert result == pytest.approx(0.0)

    def test_partial_recovery(self):
        """Halfway between corrupted and clean => 0.5."""
        result = normalized_patch_effect(
            patched_ld=1.5, clean_ld=5.0, corrupted_ld=-2.0
        )
        assert result == pytest.approx(0.5)

    def test_overshoot(self):
        """patched > clean => > 1.0."""
        result = normalized_patch_effect(
            patched_ld=8.0, clean_ld=5.0, corrupted_ld=-2.0
        )
        assert result > 1.0

    def test_undershoot(self):
        """patched < corrupted => < 0.0."""
        result = normalized_patch_effect(
            patched_ld=-5.0, clean_ld=5.0, corrupted_ld=-2.0
        )
        assert result < 0.0

    def test_negative_clean_corrupted(self):
        """Works when both clean and corrupted are negative."""
        result = normalized_patch_effect(
            patched_ld=-3.0, clean_ld=-1.0, corrupted_ld=-5.0
        )
        # (-3 - (-5)) / (-1 - (-5)) = 2 / 4 = 0.5
        assert result == pytest.approx(0.5)
