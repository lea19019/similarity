"""Tests for circuits.geometry — cross-lingual geometry comparison metrics."""
import numpy as np
import pytest

from circuits.geometry import (
    linear_cka,
    svcca,
    rsa,
    procrustes_distance,
    cosine_task_projection_similarity,
    compute_pairwise_geometry,
)


# ── CKA ───────────────────────────────────────────────────────────────────


class TestLinearCKA:
    def test_identical_gives_one(self):
        X = np.random.randn(50, 64)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)

    def test_invariant_to_rotation(self):
        """CKA(X, X @ R) == 1.0 for orthogonal R."""
        X = np.random.randn(50, 10)
        R, _ = np.linalg.qr(np.random.randn(10, 10))
        assert linear_cka(X, X @ R) == pytest.approx(1.0, abs=1e-4)

    def test_invariant_to_isotropic_scaling(self):
        X = np.random.randn(50, 32)
        assert linear_cka(X, X * 3.7) == pytest.approx(1.0, abs=1e-6)

    def test_symmetric(self):
        X = np.random.randn(50, 32)
        Y = np.random.randn(50, 16)
        assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-10)

    def test_output_in_range(self):
        X = np.random.randn(50, 32)
        Y = np.random.randn(50, 16)
        score = linear_cka(X, Y)
        assert 0.0 <= score <= 1.0 + 1e-6

    def test_different_dimensions(self):
        """CKA works even when X and Y have different feature dimensions."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 20)
        score = linear_cka(X, Y)
        assert isinstance(score, float)

    def test_zero_matrix_gives_zero(self):
        X = np.zeros((50, 10))
        Y = np.random.randn(50, 10)
        assert linear_cka(X, Y) == pytest.approx(0.0, abs=1e-6)


# ── SVCCA ─────────────────────────────────────────────────────────────────


class TestSVCCA:
    def test_identical_gives_high_score(self):
        X = np.random.randn(50, 32)
        score = svcca(X, X)
        assert score > 0.95

    def test_output_in_range(self):
        X = np.random.randn(50, 32)
        Y = np.random.randn(50, 16)
        score = svcca(X, Y)
        assert 0.0 <= score <= 1.0 + 1e-6

    def test_symmetric(self):
        X = np.random.randn(50, 20)
        Y = np.random.randn(50, 15)
        assert svcca(X, Y) == pytest.approx(svcca(Y, X), abs=1e-4)


# ── RSA ───────────────────────────────────────────────────────────────────


class TestRSA:
    def test_identical_gives_one(self):
        X = np.random.randn(50, 32)
        assert rsa(X, X) == pytest.approx(1.0, abs=1e-6)

    def test_output_in_range(self):
        X = np.random.randn(50, 32)
        Y = np.random.randn(50, 16)
        score = rsa(X, Y)
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6

    def test_too_few_examples_gives_zero(self):
        X = np.random.randn(2, 10)
        Y = np.random.randn(2, 10)
        assert rsa(X, Y) == 0.0

    def test_symmetric(self):
        X = np.random.randn(50, 20)
        Y = np.random.randn(50, 15)
        assert rsa(X, Y) == pytest.approx(rsa(Y, X), abs=1e-6)


# ── Procrustes ────────────────────────────────────────────────────────────


class TestProcrustesDistance:
    def test_identical_gives_zero(self):
        X = np.random.randn(50, 10)
        assert procrustes_distance(X, X) == pytest.approx(0.0, abs=1e-6)

    def test_rotated_gives_near_zero(self):
        X = np.random.randn(50, 10)
        R, _ = np.linalg.qr(np.random.randn(10, 10))
        assert procrustes_distance(X, X @ R) < 1e-4

    def test_nonnegative(self):
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10)
        assert procrustes_distance(X, Y) >= 0.0

    def test_different_dimensions(self):
        """Should handle different feature dimensions by padding."""
        X = np.random.randn(50, 8)
        Y = np.random.randn(50, 12)
        d = procrustes_distance(X, Y)
        assert isinstance(d, float)
        assert d >= 0.0


# ── Cosine task projection similarity ─────────────────────────────────────


class TestCosineTaskProjectionSimilarity:
    def test_identical_gives_ones(self):
        tw = np.random.randn(4, 4, 64)
        result = cosine_task_projection_similarity(tw, tw)
        np.testing.assert_array_almost_equal(result, np.ones((4, 4)), decimal=5)

    def test_shape(self):
        tw_a = np.random.randn(4, 4, 64)
        tw_b = np.random.randn(4, 4, 64)
        result = cosine_task_projection_similarity(tw_a, tw_b)
        assert result.shape == (4, 4)

    def test_output_range(self):
        tw_a = np.random.randn(4, 4, 64)
        tw_b = np.random.randn(4, 4, 64)
        result = cosine_task_projection_similarity(tw_a, tw_b)
        assert np.all(result >= -1.0 - 1e-6)
        assert np.all(result <= 1.0 + 1e-6)

    def test_negated_gives_minus_one(self):
        tw = np.random.randn(4, 4, 64)
        result = cosine_task_projection_similarity(tw, -tw)
        np.testing.assert_array_almost_equal(result, -np.ones((4, 4)), decimal=5)


# ── Full pairwise computation ─────────────────────────────────────────────


class TestComputePairwiseGeometry:
    def test_output_keys(self):
        n_layers, n_ex, d_model = 4, 20, 32
        acts = {
            "en": np.random.randn(n_layers, n_ex, d_model),
            "es": np.random.randn(n_layers, n_ex, d_model),
        }
        results = compute_pairwise_geometry(acts, {}, ["en", "es"])
        assert "cka_per_layer" in results
        assert "svcca_per_layer" in results
        assert "rsa_per_layer" in results
        assert "procrustes_per_layer" in results
        assert "pair_labels" in results
        assert "convergence" in results

    def test_shapes_two_langs(self):
        n_layers, n_ex, d_model = 4, 20, 32
        acts = {
            "en": np.random.randn(n_layers, n_ex, d_model),
            "es": np.random.randn(n_layers, n_ex, d_model),
        }
        results = compute_pairwise_geometry(acts, {}, ["en", "es"])
        assert results["cka_per_layer"].shape == (4, 1)  # 1 pair
        assert results["convergence"].shape == (4,)
        assert list(results["pair_labels"]) == ["en-es"]

    def test_shapes_four_langs(self):
        n_layers, n_ex, d_model = 4, 20, 32
        langs = ["en", "es", "tr", "sw"]
        acts = {l: np.random.randn(n_layers, n_ex, d_model) for l in langs}
        results = compute_pairwise_geometry(acts, {}, langs)
        # C(4,2) = 6 pairs
        assert results["cka_per_layer"].shape == (4, 6)
        assert len(results["pair_labels"]) == 6

    def test_with_task_weights(self):
        n_layers, n_ex, d_model, n_heads = 4, 20, 32, 4
        langs = ["en", "es"]
        acts = {l: np.random.randn(n_layers, n_ex, d_model) for l in langs}
        tw = {l: np.random.randn(n_layers, n_heads, d_model) for l in langs}
        results = compute_pairwise_geometry(acts, tw, langs)
        assert "task_cosine" in results
        assert results["task_cosine"].shape == (1, n_layers, n_heads)

    def test_convergence_is_mean_cka(self):
        n_layers, n_ex, d_model = 4, 20, 32
        acts = {
            "en": np.random.randn(n_layers, n_ex, d_model),
            "es": np.random.randn(n_layers, n_ex, d_model),
        }
        results = compute_pairwise_geometry(acts, {}, ["en", "es"])
        expected = results["cka_per_layer"].mean(axis=1)
        np.testing.assert_array_almost_equal(results["convergence"], expected)

    def test_different_example_counts(self):
        """Should handle languages with different numbers of examples."""
        n_layers, d_model = 4, 32
        acts = {
            "en": np.random.randn(n_layers, 30, d_model),
            "es": np.random.randn(n_layers, 20, d_model),
        }
        results = compute_pairwise_geometry(acts, {}, ["en", "es"])
        # Should not crash — uses min(30, 20) = 20 examples
        assert results["cka_per_layer"].shape == (4, 1)
