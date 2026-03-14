"""Tests for circuits.patching — save_results round-trip."""
import numpy as np
import pytest

from circuits.patching import save_results


class TestSaveResults:
    def test_round_trip_npz(self, tmp_path):
        """save_results should write arrays that can be loaded back."""
        n_layers, n_heads = 4, 4
        results = {
            "head_out": np.random.rand(n_layers, n_heads),
            "attn_out": np.random.rand(n_layers),
            "mlp_out": np.random.rand(n_layers),
            "n_examples": 10,  # non-array, should be skipped
        }
        out_path = tmp_path / "patching.npz"
        save_results(results, out_path)

        assert out_path.exists()
        loaded = np.load(out_path)
        np.testing.assert_array_almost_equal(loaded["head_out"], results["head_out"])
        np.testing.assert_array_almost_equal(loaded["attn_out"], results["attn_out"])
        np.testing.assert_array_almost_equal(loaded["mlp_out"], results["mlp_out"])

    def test_non_array_values_excluded(self, tmp_path):
        """Non-ndarray values (like int) should not appear in the npz."""
        results = {
            "head_out": np.zeros((2, 2)),
            "n_examples": 5,
        }
        out_path = tmp_path / "patching2.npz"
        save_results(results, out_path)
        loaded = np.load(out_path)
        assert "head_out" in loaded.files
        assert "n_examples" not in loaded.files

    def test_creates_parent_dirs(self, tmp_path):
        results = {"head_out": np.zeros((2, 2))}
        out_path = tmp_path / "nested" / "dir" / "results.npz"
        save_results(results, out_path)
        assert out_path.exists()
