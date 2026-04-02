"""Tests for circuits.edge_patching — Edge Attribution Patching."""
import numpy as np
import pytest

from circuits.edge_patching import _make_component_labels


class TestMakeComponentLabels:
    def test_length(self):
        """Should have n_layers * n_heads + n_layers entries."""
        labels = _make_component_labels(n_layers=4, n_heads=4)
        assert len(labels) == 4 * 4 + 4  # 20

    def test_head_labels_format(self):
        labels = _make_component_labels(n_layers=2, n_heads=3)
        # First n_layers * n_heads are head labels
        assert labels[0] == "L0H0"
        assert labels[1] == "L0H1"
        assert labels[2] == "L0H2"
        assert labels[3] == "L1H0"
        assert labels[4] == "L1H1"
        assert labels[5] == "L1H2"

    def test_mlp_labels_format(self):
        labels = _make_component_labels(n_layers=2, n_heads=3)
        # After heads come MLP labels
        assert labels[6] == "MLP0"
        assert labels[7] == "MLP1"

    def test_ordering(self):
        """Head labels come first, then MLP labels."""
        labels = _make_component_labels(n_layers=3, n_heads=2)
        # 6 head labels + 3 MLP labels = 9 total
        assert len(labels) == 9
        for i in range(6):
            assert labels[i].startswith("L")
        for i in range(6, 9):
            assert labels[i].startswith("MLP")


class TestEAPScoresShape:
    """Test the shape contracts of EAP results using the mock npz fixture."""

    def test_node_scores_shape(self, edge_patching_npz):
        path, data = edge_patching_npz
        loaded = np.load(path, allow_pickle=True)
        n_layers, n_heads = 4, 4
        expected = n_layers * n_heads + n_layers
        assert loaded["node_scores"].shape == (expected,)

    def test_edge_scores_shape(self, edge_patching_npz):
        path, data = edge_patching_npz
        loaded = np.load(path, allow_pickle=True)
        n = 4 * 4 + 4
        assert loaded["edge_scores"].shape == (n, n)

    def test_labels_match_scores(self, edge_patching_npz):
        path, data = edge_patching_npz
        loaded = np.load(path, allow_pickle=True)
        assert len(loaded["component_labels"]) == len(loaded["node_scores"])

    def test_scores_nonnegative(self, edge_patching_npz):
        path, data = edge_patching_npz
        loaded = np.load(path, allow_pickle=True)
        assert (loaded["node_scores"] >= 0).all()
