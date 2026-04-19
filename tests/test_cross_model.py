"""Tests for circuits.cross_model — cross-model comparison utilities."""
import numpy as np
import pytest

from circuits.cross_model import (
    normalize_profile,
    interpolate_to_common_grid,
    compare_profiles,
    compare_flow_topology,
)


class TestNormalizeProfile:
    def test_output_range(self):
        values = np.array([1.0, 3.0, 5.0, 2.0])
        depth, norm = normalize_profile(values)
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)

    def test_depth_range(self):
        values = np.array([1.0, 3.0, 5.0])
        depth, _ = normalize_profile(values)
        assert depth[0] == pytest.approx(0.0)
        assert depth[-1] == pytest.approx(1.0)

    def test_length_preserved(self):
        values = np.random.randn(18)
        depth, norm = normalize_profile(values)
        assert len(depth) == 18
        assert len(norm) == 18

    def test_constant_input(self):
        """Constant values should normalize to all zeros."""
        values = np.ones(10) * 5.0
        _, norm = normalize_profile(values)
        np.testing.assert_array_almost_equal(norm, 0.0)

    def test_peak_preserved(self):
        """Peak position should be preserved after normalization."""
        values = np.array([0.0, 1.0, 5.0, 2.0, 0.0])
        _, norm = normalize_profile(values)
        assert np.argmax(norm) == 2


class TestInterpolateToCommonGrid:
    def test_output_length(self):
        depth_a = np.linspace(0, 1, 18)
        vals_a = np.random.randn(18)
        depth_b = np.linspace(0, 1, 30)
        vals_b = np.random.randn(30)

        common, interp_a, interp_b = interpolate_to_common_grid(
            depth_a, vals_a, depth_b, vals_b, n_points=50
        )
        assert len(common) == 50
        assert len(interp_a) == 50
        assert len(interp_b) == 50

    def test_endpoints_preserved(self):
        depth_a = np.linspace(0, 1, 10)
        vals_a = np.arange(10, dtype=float)
        depth_b = np.linspace(0, 1, 20)
        vals_b = np.arange(20, dtype=float)

        common, interp_a, interp_b = interpolate_to_common_grid(
            depth_a, vals_a, depth_b, vals_b
        )
        assert interp_a[0] == pytest.approx(vals_a[0], abs=1e-6)
        assert interp_a[-1] == pytest.approx(vals_a[-1], abs=1e-6)
        assert interp_b[0] == pytest.approx(vals_b[0], abs=1e-6)
        assert interp_b[-1] == pytest.approx(vals_b[-1], abs=1e-6)

    def test_same_length_inputs(self):
        depth = np.linspace(0, 1, 18)
        vals_a = np.random.randn(18)
        vals_b = np.random.randn(18)
        common, interp_a, interp_b = interpolate_to_common_grid(
            depth, vals_a, depth, vals_b
        )
        assert len(common) == 100  # default n_points


class TestCompareProfiles:
    def test_identical_profiles(self):
        profile = np.random.randn(100)
        result = compare_profiles(profile, profile)
        assert result["pearson"] == pytest.approx(1.0, abs=1e-6)
        assert result["spearman"] == pytest.approx(1.0, abs=1e-6)
        assert result["peak_shift"] == pytest.approx(0.0, abs=1e-6)
        assert result["l2_distance"] == pytest.approx(0.0, abs=1e-6)
        assert result["cosine"] == pytest.approx(1.0, abs=1e-6)

    def test_negated_profile(self):
        profile = np.random.randn(100) + 2  # shift to avoid zero-crossing issues
        result = compare_profiles(profile, -profile)
        assert result["pearson"] == pytest.approx(-1.0, abs=1e-6)
        assert result["cosine"] == pytest.approx(-1.0, abs=1e-6)

    def test_output_keys(self):
        a = np.random.randn(50)
        b = np.random.randn(50)
        result = compare_profiles(a, b)
        expected_keys = {
            "pearson", "pearson_p", "spearman", "spearman_p",
            "peak_shift", "peak_a", "peak_b",
            "l2_distance", "cosine",
        }
        assert set(result.keys()) == expected_keys

    def test_peak_shift_range(self):
        a = np.random.randn(100)
        b = np.random.randn(100)
        result = compare_profiles(a, b)
        assert 0.0 <= result["peak_shift"] <= 1.0

    def test_shifted_peak(self):
        """Profile with shifted peak should have non-zero peak_shift."""
        a = np.zeros(100)
        a[20] = 1.0  # peak at 20%
        b = np.zeros(100)
        b[80] = 1.0  # peak at 80%
        result = compare_profiles(a, b)
        assert result["peak_shift"] == pytest.approx(0.6, abs=0.02)

    def test_l2_distance_nonnegative(self):
        a = np.random.randn(100)
        b = np.random.randn(100)
        result = compare_profiles(a, b)
        assert result["l2_distance"] >= 0.0


class TestCompareFlowTopology:
    def test_missing_profiles_skipped(self, tmp_path):
        """Should skip languages with missing profiles without crashing."""
        results = compare_flow_topology(
            str(tmp_path), "gemma-2b", "bloom-3b", ["en", "es"]
        )
        assert results["model_a"] == "gemma-2b"
        assert results["model_b"] == "bloom-3b"
        # No profiles found, per_lang should be empty
        assert len(results["per_lang"]) == 0

    def test_with_synthetic_profiles(self, tmp_path):
        """Create synthetic RepE profiles and verify comparison works."""
        n_layers_a, n_layers_b, d_a, d_b = 18, 30, 2048, 2560

        for model, n_layers, d_model in [
            ("gemma_2b", n_layers_a, d_a),
            ("bloom_3b", n_layers_b, d_b),
        ]:
            np.savez(
                tmp_path / f"repe_{model}_en.npz",
                reading_vectors=np.random.randn(n_layers, d_model),
                explained_variance=np.random.rand(n_layers),
                accuracy=np.random.rand(n_layers) * 0.5 + 0.5,
                signal_magnitude=np.random.rand(n_layers),
                signal_std=np.random.rand(n_layers),
                signal_snr=np.random.rand(n_layers),
                diff_norms=np.random.rand(n_layers),
            )

        results = compare_flow_topology(
            str(tmp_path), "gemma-2b", "bloom-3b", ["en"]
        )
        assert "en" in results["per_lang"]
        assert "accuracy" in results["per_lang"]["en"]
        assert "pearson" in results["per_lang"]["en"]["accuracy"]

    def test_aggregate_computed(self, tmp_path):
        """Aggregate metrics should be computed when profiles exist."""
        for model, n_layers, d_model in [
            ("gemma_2b", 18, 2048),
            ("bloom_3b", 30, 2560),
        ]:
            for lang in ["en", "es"]:
                np.savez(
                    tmp_path / f"repe_{model}_{lang}.npz",
                    reading_vectors=np.random.randn(n_layers, d_model),
                    explained_variance=np.random.rand(n_layers),
                    accuracy=np.random.rand(n_layers) * 0.5 + 0.5,
                    signal_magnitude=np.random.rand(n_layers),
                    signal_std=np.random.rand(n_layers),
                    signal_snr=np.random.rand(n_layers),
                    diff_norms=np.random.rand(n_layers),
                )

        results = compare_flow_topology(
            str(tmp_path), "gemma-2b", "bloom-3b", ["en", "es"]
        )
        assert "aggregate" in results
        assert "accuracy" in results["aggregate"]
        assert "mean_pearson" in results["aggregate"]["accuracy"]
