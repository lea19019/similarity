"""Tests for circuits.pca — fit_pca with synthetic data."""
import numpy as np
import pytest
from sklearn.decomposition import PCA

from circuits.pca import fit_pca


class TestFitPca:
    def test_returns_pca_object(self):
        vectors = np.random.randn(50, 16)
        result = fit_pca(vectors)
        assert isinstance(result, PCA)

    def test_n_components_capped(self):
        """When d_head < 10, n_components should equal d_head."""
        vectors = np.random.randn(20, 5)
        pca = fit_pca(vectors)
        assert pca.n_components == 5

    def test_n_components_max_10(self):
        """When d_head >= 10, n_components should be 10."""
        vectors = np.random.randn(50, 64)
        pca = fit_pca(vectors)
        assert pca.n_components == 10

    def test_variance_ratio_sums_le_one(self):
        vectors = np.random.randn(100, 20)
        pca = fit_pca(vectors)
        assert sum(pca.explained_variance_ratio_) <= 1.0 + 1e-6

    def test_pc1_has_correct_shape(self):
        d_head = 32
        vectors = np.random.randn(50, d_head)
        pca = fit_pca(vectors)
        assert pca.components_[0].shape == (d_head,)

    def test_structured_data_separates(self):
        """PC1 should capture the dominant direction in structured data."""
        rng = np.random.default_rng(42)
        # Two clusters along axis 0
        group_a = rng.normal(loc=[5, 0, 0, 0], scale=0.1, size=(50, 4))
        group_b = rng.normal(loc=[-5, 0, 0, 0], scale=0.1, size=(50, 4))
        vectors = np.vstack([group_a, group_b])
        pca = fit_pca(vectors)
        # PC1 should explain most variance
        assert pca.explained_variance_ratio_[0] > 0.9
