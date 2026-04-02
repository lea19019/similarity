"""Tests for circuits.attention — attention pattern analysis."""
import numpy as np
import pytest

from circuits.attention import compute_subject_attention_score


class TestComputeSubjectAttentionScore:
    def test_output_shape(self):
        patterns = np.random.rand(4, 4, 20)
        result = compute_subject_attention_score(patterns)
        assert result.shape == (4, 4)

    def test_range(self):
        # Attention patterns are probabilities, so subject attention <= 1.0
        patterns = np.zeros((4, 4, 20))
        patterns[:, :, 1:4] = 1.0 / 3  # uniform on subject positions
        result = compute_subject_attention_score(patterns)
        assert (result >= 0).all()
        assert (result <= 1.0 + 1e-6).all()

    def test_zero_attention_to_subject(self):
        patterns = np.zeros((4, 4, 20))
        patterns[:, :, 10] = 1.0  # all attention on position 10
        result = compute_subject_attention_score(patterns)
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_full_attention_to_subject(self):
        patterns = np.zeros((2, 2, 10))
        # All attention on positions 1-3
        patterns[:, :, 1] = 0.4
        patterns[:, :, 2] = 0.3
        patterns[:, :, 3] = 0.3
        result = compute_subject_attention_score(patterns)
        np.testing.assert_array_almost_equal(result, 1.0)
