"""Tests for circuits.repe — RepE layer scanning with synthetic data."""
import numpy as np
import pytest

from circuits.repe import (
    compute_reading_vectors,
    compute_signal_profile,
    compute_reading_vector_accuracy,
)


class TestComputeReadingVectors:
    def test_output_shapes(self):
        n_layers, n_examples, d_model = 6, 20, 64
        diffs = np.random.randn(n_layers, n_examples, d_model)
        rv, ev = compute_reading_vectors(diffs)
        assert rv.shape == (n_layers, d_model)
        assert ev.shape == (n_layers,)

    def test_explained_variance_in_range(self):
        diffs = np.random.randn(6, 30, 64)
        _, ev = compute_reading_vectors(diffs)
        assert np.all(ev >= 0.0)
        assert np.all(ev <= 1.0 + 1e-6)

    def test_reading_vectors_are_unit_directions(self):
        """PC1 from sklearn PCA are unit vectors."""
        diffs = np.random.randn(4, 20, 32)
        rv, _ = compute_reading_vectors(diffs)
        for layer in range(4):
            norm = np.linalg.norm(rv[layer])
            assert norm == pytest.approx(1.0, abs=1e-5)

    def test_structured_signal_captured(self):
        """If diffs have a dominant direction, PC1 should align with it."""
        rng = np.random.default_rng(42)
        n_examples, d_model = 50, 32

        # Layer 0: random noise (no structure)
        layer0 = rng.normal(0, 1, (n_examples, d_model))
        # Layer 1: strong signal along axis 0
        layer1 = rng.normal(0, 0.1, (n_examples, d_model))
        layer1[:, 0] += rng.normal(5, 0.5, n_examples)

        diffs = np.stack([layer0, layer1])
        rv, ev = compute_reading_vectors(diffs)

        # Layer 1 should have higher explained variance
        assert ev[1] > ev[0]
        # Layer 1's reading vector should point mostly along axis 0
        assert abs(rv[1, 0]) > 0.5

    def test_single_example_handled(self):
        """Should not crash with just 1 example (PCA needs >=2)."""
        diffs = np.random.randn(4, 1, 32)
        rv, ev = compute_reading_vectors(diffs)
        assert rv.shape == (4, 32)

    def test_mean_diff_output_shapes(self):
        diffs = np.random.randn(6, 20, 64)
        rv, ev = compute_reading_vectors(diffs, method="mean_diff")
        assert rv.shape == (6, 64)
        assert ev.shape == (6,)

    def test_mean_diff_produces_unit_vectors(self):
        diffs = np.random.randn(4, 30, 32)
        rv, _ = compute_reading_vectors(diffs, method="mean_diff")
        for layer in range(4):
            norm = np.linalg.norm(rv[layer])
            assert norm == pytest.approx(1.0, abs=1e-5)

    def test_mean_diff_consistency_in_range(self):
        diffs = np.random.randn(4, 30, 32)
        _, ev = compute_reading_vectors(diffs, method="mean_diff")
        assert np.all(ev >= 0.0)
        assert np.all(ev <= 1.0 + 1e-6)

    def test_mean_diff_structured_signal(self):
        """Mean diff should align with a consistent signal direction."""
        rng = np.random.default_rng(42)
        n_ex, d = 50, 32
        diffs = rng.normal(0, 0.1, (2, n_ex, d))
        # Layer 1: strong consistent signal along axis 0
        diffs[1, :, 0] += 5.0
        rv, ev = compute_reading_vectors(diffs, method="mean_diff")
        # Layer 1 should have high consistency
        assert ev[1] > ev[0]
        # Reading vector should point along axis 0
        assert abs(rv[1, 0]) > 0.8


class TestComputeSignalProfile:
    def test_output_keys(self):
        diffs = np.random.randn(6, 20, 64)
        rv, _ = compute_reading_vectors(diffs)
        profile = compute_signal_profile(diffs, rv)
        assert "signal_magnitude" in profile
        assert "signal_std" in profile
        assert "signal_snr" in profile
        assert "diff_norms" in profile

    def test_output_shapes(self):
        n_layers = 6
        diffs = np.random.randn(n_layers, 20, 64)
        rv, _ = compute_reading_vectors(diffs)
        profile = compute_signal_profile(diffs, rv)
        for key in ["signal_magnitude", "signal_std", "signal_snr", "diff_norms"]:
            assert profile[key].shape == (n_layers,)

    def test_magnitude_nonnegative(self):
        diffs = np.random.randn(4, 20, 32)
        rv, _ = compute_reading_vectors(diffs)
        profile = compute_signal_profile(diffs, rv)
        assert np.all(profile["signal_magnitude"] >= 0)

    def test_diff_norms_nonnegative(self):
        diffs = np.random.randn(4, 20, 32)
        rv, _ = compute_reading_vectors(diffs)
        profile = compute_signal_profile(diffs, rv)
        assert np.all(profile["diff_norms"] >= 0)

    def test_zero_diffs_give_zero_magnitude(self):
        diffs = np.zeros((4, 20, 32))
        rv = np.random.randn(4, 32)
        # Normalize rv so they're valid
        rv = rv / np.linalg.norm(rv, axis=1, keepdims=True)
        profile = compute_signal_profile(diffs, rv)
        np.testing.assert_array_almost_equal(profile["signal_magnitude"], 0.0)

    def test_structured_signal_has_higher_diff_norms(self):
        """Layer with structured signal should have higher diff norms."""
        rng = np.random.default_rng(42)
        n_ex, d = 50, 32

        # Layer 0: small noise
        layer0 = rng.normal(0, 0.1, (n_ex, d))
        # Layer 1: strong signal with variance along axis 0
        layer1 = rng.normal(0, 0.1, (n_ex, d))
        layer1[:, 0] += rng.normal(10, 1, n_ex)

        diffs = np.stack([layer0, layer1])
        rv, _ = compute_reading_vectors(diffs)
        profile = compute_signal_profile(diffs, rv)

        assert profile["diff_norms"][1] > profile["diff_norms"][0]


class TestComputeReadingVectorAccuracy:
    def test_output_shape(self):
        n_layers, n_ex, d = 4, 20, 32
        clean = np.random.randn(n_layers, n_ex, d)
        corrupted = np.random.randn(n_layers, n_ex, d)
        rv = np.random.randn(n_layers, d)
        acc = compute_reading_vector_accuracy(clean, corrupted, rv)
        assert acc.shape == (n_layers,)

    def test_accuracy_in_range(self):
        n_layers, n_ex, d = 4, 20, 32
        clean = np.random.randn(n_layers, n_ex, d)
        corrupted = np.random.randn(n_layers, n_ex, d)
        rv = np.random.randn(n_layers, d)
        acc = compute_reading_vector_accuracy(clean, corrupted, rv)
        assert np.all(acc >= 0.0)
        assert np.all(acc <= 1.0)

    def test_perfect_separation(self):
        """When clean and corrupted are perfectly separated along the
        reading vector, accuracy should be 1.0."""
        n_layers, n_ex, d = 2, 30, 16
        rng = np.random.default_rng(42)

        rv = np.zeros((n_layers, d))
        rv[:, 0] = 1.0  # reading vector along axis 0

        clean = rng.normal(0, 0.1, (n_layers, n_ex, d))
        corrupted = rng.normal(0, 0.1, (n_layers, n_ex, d))
        clean[:, :, 0] += 5.0   # clean projects high
        corrupted[:, :, 0] -= 5.0  # corrupted projects low

        acc = compute_reading_vector_accuracy(clean, corrupted, rv)
        assert np.all(acc > 0.95)

    def test_random_gives_near_chance(self):
        """With random data, accuracy should be near 0.5."""
        rng = np.random.default_rng(42)
        n_layers, n_ex, d = 4, 200, 32
        clean = rng.normal(0, 1, (n_layers, n_ex, d))
        corrupted = rng.normal(0, 1, (n_layers, n_ex, d))
        rv = rng.normal(0, 1, (n_layers, d))
        acc = compute_reading_vector_accuracy(clean, corrupted, rv)
        # With 200 examples, should be between 0.4 and 0.65
        assert np.all(acc > 0.35)
        assert np.all(acc < 0.70)

    def test_zero_reading_vector_gives_half(self):
        """Zero reading vector should give 0.5 accuracy."""
        n_layers, n_ex, d = 2, 20, 16
        clean = np.random.randn(n_layers, n_ex, d)
        corrupted = np.random.randn(n_layers, n_ex, d)
        rv = np.zeros((n_layers, d))
        acc = compute_reading_vector_accuracy(clean, corrupted, rv)
        np.testing.assert_array_almost_equal(acc, 0.5)
