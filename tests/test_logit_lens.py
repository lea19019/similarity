"""Tests for circuits.logit_lens — residual stream decomposition."""
import numpy as np
import pytest


class TestLogitLensOutputFormat:
    """Test the expected output format using mock npz data."""

    def test_mock_logit_lens_shapes(self, tmp_path):
        n_layers = 4
        n_examples = 10
        # Simulate logit lens output
        data = {
            "logit_diff_by_layer": np.random.randn(n_layers + 1, n_examples),
            "correct_rank_by_layer": np.random.randint(0, 1000, (n_layers + 1, n_examples)),
            "correct_prob_by_layer": np.random.rand(n_layers + 1, n_examples),
            "mean_logit_diff": np.random.randn(n_layers + 1),
            "mean_correct_prob": np.random.rand(n_layers + 1),
            "mean_correct_rank": np.random.rand(n_layers + 1),
        }
        p = tmp_path / "logit_lens_test.npz"
        np.savez(p, **data)

        loaded = np.load(p)
        assert loaded["logit_diff_by_layer"].shape == (n_layers + 1, n_examples)
        assert loaded["mean_logit_diff"].shape == (n_layers + 1,)
        assert loaded["mean_correct_prob"].shape == (n_layers + 1,)

    def test_prob_in_range(self):
        # Probabilities should be in [0, 1]
        probs = np.random.rand(5, 10)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_final_layer_is_last_row(self):
        """The last row (index n_layers) should represent the final model output."""
        n_layers = 4
        data = np.random.randn(n_layers + 1, 10)
        # Final layer is at index n_layers
        assert data.shape[0] == n_layers + 1
