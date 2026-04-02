"""Tests for circuits.circuit_map — weight-level circuit analysis."""
import numpy as np
import pytest
import torch

from circuits.circuit_map import (
    compute_ov_matrix,
    compute_qk_matrix,
    compute_task_projection,
    compute_mlp_task_projection,
    compute_unembed_direction,
    compute_weight_importance,
    compute_cross_layer_connections,
    svd_ov,
    build_importance_map,
)


# ── OV / QK matrix computation ────────────────────────────────────────────


class TestComputeOVMatrix:
    def test_output_shape(self):
        """OV matrix should be (d_model, d_model)."""
        from unittest.mock import MagicMock

        d_model, d_head = 64, 16
        model = MagicMock()
        model.W_V = torch.randn(4, 4, d_model, d_head)
        model.W_O = torch.randn(4, 4, d_head, d_model)

        ov = compute_ov_matrix(model, layer=0, head=0)
        assert ov.shape == (d_model, d_model)

    def test_identity_like_weights(self):
        """With identity-like W_V and W_O, OV should be a projection matrix."""
        from unittest.mock import MagicMock

        d_model, d_head = 8, 4
        model = MagicMock()

        # W_V selects first d_head dimensions, W_O projects back
        W_V = torch.zeros(1, 1, d_model, d_head)
        W_V[0, 0, :d_head, :] = torch.eye(d_head)
        W_O = torch.zeros(1, 1, d_head, d_model)
        W_O[0, 0, :, :d_head] = torch.eye(d_head)

        model.W_V = W_V
        model.W_O = W_O

        ov = compute_ov_matrix(model, 0, 0)
        # OV should be a projection onto first d_head dimensions
        expected = torch.zeros(d_model, d_model)
        expected[:d_head, :d_head] = torch.eye(d_head)
        torch.testing.assert_close(ov, expected)


class TestComputeQKMatrix:
    def test_output_shape(self):
        from unittest.mock import MagicMock

        d_model, d_head = 64, 16
        model = MagicMock()
        model.W_Q = torch.randn(4, 4, d_model, d_head)
        model.W_K = torch.randn(4, 4, d_model, d_head)

        qk = compute_qk_matrix(model, layer=1, head=2)
        assert qk.shape == (d_model, d_model)

    def test_symmetric_when_q_equals_k(self):
        """When W_Q == W_K, QK matrix should be symmetric (W @ W^T)."""
        from unittest.mock import MagicMock

        d_model, d_head = 8, 4
        model = MagicMock()
        W = torch.randn(1, 1, d_model, d_head)
        model.W_Q = W
        model.W_K = W

        qk = compute_qk_matrix(model, 0, 0)
        torch.testing.assert_close(qk, qk.T, atol=1e-5, rtol=1e-5)


# ── SVD decomposition ─────────────────────────────────────────────────────


class TestSvdOV:
    def test_output_shapes(self):
        ov = torch.randn(64, 64)
        U, S, Vh = svd_ov(ov, top_k=5)
        assert U.shape == (64, 5)
        assert S.shape == (5,)
        assert Vh.shape == (5, 64)

    def test_singular_values_descending(self):
        ov = torch.randn(32, 32)
        _, S, _ = svd_ov(ov, top_k=10)
        for i in range(len(S) - 1):
            assert S[i] >= S[i + 1]

    def test_top_k_caps_at_matrix_size(self):
        ov = torch.randn(8, 8)
        _, S, _ = svd_ov(ov, top_k=100)
        assert S.shape[0] == 8

    def test_reconstruction(self):
        """U @ diag(S) @ Vh should approximate the original matrix."""
        ov = torch.randn(16, 16)
        U, S, Vh = svd_ov(ov, top_k=16)
        reconstructed = U @ torch.diag(S) @ Vh
        torch.testing.assert_close(reconstructed, ov, atol=1e-4, rtol=1e-4)


# ── Task projection ───────────────────────────────────────────────────────


class TestComputeTaskProjection:
    def test_output_shape(self):
        ov = torch.randn(64, 64)
        unembed_dir = torch.randn(64)
        unembed_dir = unembed_dir / unembed_dir.norm()
        result = compute_task_projection(ov, unembed_dir)
        assert result.shape == (64,)

    def test_identity_ov(self):
        """With identity OV, task projection should equal unembed_dir."""
        d = 16
        ov = torch.eye(d)
        unembed_dir = torch.randn(d)
        unembed_dir = unembed_dir / unembed_dir.norm()
        result = compute_task_projection(ov, unembed_dir)
        torch.testing.assert_close(result, unembed_dir, atol=1e-6, rtol=1e-6)

    def test_zero_ov(self):
        """With zero OV, task projection should be zero."""
        d = 16
        ov = torch.zeros(d, d)
        unembed_dir = torch.randn(d)
        result = compute_task_projection(ov, unembed_dir)
        assert result.norm().item() == pytest.approx(0.0, abs=1e-10)


class TestComputeMLPTaskProjection:
    def test_output_shape(self):
        from unittest.mock import MagicMock

        d_model, d_mlp = 64, 128
        model = MagicMock()
        model.blocks = [MagicMock()]
        model.blocks[0].mlp.W_out = torch.randn(d_mlp, d_model)

        unembed_dir = torch.randn(d_model)
        result = compute_mlp_task_projection(model, layer=0, unembed_dir=unembed_dir)
        assert result.shape == (d_mlp,)


# ── Unembed direction ─────────────────────────────────────────────────────


class TestComputeUnembedDirection:
    def test_output_is_normalized(self):
        from unittest.mock import MagicMock

        d_model, d_vocab = 64, 100
        model = MagicMock()
        model.cfg.d_model = d_model
        model.W_U = torch.randn(d_model, d_vocab)
        model.to_tokens = MagicMock(return_value=torch.tensor([[1, 42]]))

        # We need get_token_id to work — mock the tokenizer behavior
        def fake_to_tokens(text, prepend_bos=True):
            return torch.tensor([[1, 42]])

        model.to_tokens = MagicMock(side_effect=fake_to_tokens)

        examples = [
            {"good_verb": "is", "bad_verb": "are", "clean": "x", "corrupted": "y"},
        ]

        # Mock get_token_id to return deterministic IDs
        import circuits.circuit_map as cm
        original_get_token_id = cm.get_token_id
        cm.get_token_id = lambda model, word: 10 if word == "is" else 20

        try:
            result = compute_unembed_direction(model, examples)
            assert result.shape == (d_model,)
            assert abs(result.norm().item() - 1.0) < 1e-4
        finally:
            cm.get_token_id = original_get_token_id


# ── Per-weight importance ──────────────────────────────────────────────────


class TestComputeWeightImportance:
    def test_output_shapes(self):
        from unittest.mock import MagicMock

        d_model, d_head = 64, 16
        model = MagicMock()
        model.W_V = torch.randn(4, 4, d_model, d_head)
        model.W_O = torch.randn(4, 4, d_head, d_model)

        unembed_dir = torch.randn(d_model)
        unembed_dir = unembed_dir / unembed_dir.norm()

        result = compute_weight_importance(model, layer=0, head=0, unembed_dir=unembed_dir)
        assert result["wv_importance"].shape == (d_model, d_head)
        assert result["wo_importance"].shape == (d_head, d_model)

    def test_values_nonnegative(self):
        from unittest.mock import MagicMock

        d_model, d_head = 32, 8
        model = MagicMock()
        model.W_V = torch.randn(1, 1, d_model, d_head)
        model.W_O = torch.randn(1, 1, d_head, d_model)

        unembed_dir = torch.randn(d_model)
        result = compute_weight_importance(model, 0, 0, unembed_dir)
        assert (result["wv_importance"] >= 0).all()
        assert (result["wo_importance"] >= 0).all()

    def test_zero_unembed_gives_zero(self):
        from unittest.mock import MagicMock

        d_model, d_head = 16, 4
        model = MagicMock()
        model.W_V = torch.randn(1, 1, d_model, d_head)
        model.W_O = torch.randn(1, 1, d_head, d_model)

        unembed_dir = torch.zeros(d_model)
        result = compute_weight_importance(model, 0, 0, unembed_dir)
        assert result["wo_importance"].max() == pytest.approx(0.0, abs=1e-10)


class TestComputeCrossLayerConnections:
    def test_output_shapes(self):
        from unittest.mock import MagicMock

        n_layers, n_heads, d_model, d_head = 2, 2, 16, 8
        model = MagicMock()
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.W_V = torch.randn(n_layers, n_heads, d_model, d_head)
        model.W_O = torch.randn(n_layers, n_heads, d_head, d_model)

        unembed_dir = torch.randn(d_model)
        result = compute_cross_layer_connections(model, unembed_dir)
        n_comp = n_layers * n_heads  # 4
        assert result["connection_matrix"].shape == (n_comp, n_comp)
        assert len(result["component_labels"]) == n_comp

    def test_no_self_or_backward_connections(self):
        from unittest.mock import MagicMock

        n_layers, n_heads, d_model, d_head = 3, 2, 16, 8
        model = MagicMock()
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.W_V = torch.randn(n_layers, n_heads, d_model, d_head)
        model.W_O = torch.randn(n_layers, n_heads, d_head, d_model)

        unembed_dir = torch.randn(d_model)
        result = compute_cross_layer_connections(model, unembed_dir)
        conn = result["connection_matrix"]
        labels = result["component_labels"]

        # No self-connections or backward connections
        for i in range(len(labels)):
            li = int(str(labels[i])[1:].split("H")[0])
            for j in range(len(labels)):
                lj = int(str(labels[j])[1:].split("H")[0])
                if li >= lj:
                    assert conn[i, j] == 0.0, f"Expected 0 for {labels[i]}→{labels[j]}"

    def test_connection_values_in_range(self):
        from unittest.mock import MagicMock

        n_layers, n_heads, d_model, d_head = 2, 2, 16, 8
        model = MagicMock()
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.W_V = torch.randn(n_layers, n_heads, d_model, d_head)
        model.W_O = torch.randn(n_layers, n_heads, d_head, d_model)

        unembed_dir = torch.randn(d_model)
        result = compute_cross_layer_connections(model, unembed_dir)
        # Cosine-based, should be in [0, 1]
        assert (result["connection_matrix"] >= -0.01).all()
        assert (result["connection_matrix"] <= 1.01).all()


# ── Full importance map ────────────────────────────────────────────────────


class TestBuildImportanceMap:
    def test_output_keys(self):
        from unittest.mock import MagicMock

        n_layers, n_heads, d_model, d_head, d_mlp = 2, 2, 16, 8, 32

        model = MagicMock()
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.cfg.d_model = d_model
        model.cfg.d_head = d_head
        model.cfg.d_mlp = d_mlp
        model.W_U = torch.randn(d_model, 100)
        model.W_V = torch.randn(n_layers, n_heads, d_model, d_head)
        model.W_O = torch.randn(n_layers, n_heads, d_head, d_model)

        for layer in range(n_layers):
            model.blocks[layer].mlp.W_out = torch.randn(d_mlp, d_model)

        import circuits.circuit_map as cm
        original = cm.get_token_id
        cm.get_token_id = lambda m, w: hash(w) % 100

        examples = [
            {"good_verb": "is", "bad_verb": "are", "clean": "x", "corrupted": "y"},
        ]

        try:
            results = build_importance_map(model, examples, svd_top_k=5)
        finally:
            cm.get_token_id = original

        assert "head_importance" in results
        assert "mlp_importance" in results
        assert "head_task_weights" in results
        assert "svd_spectra" in results
        assert "wv_importance" in results
        assert "wo_importance" in results
        assert "mlp_neuron_importance" in results
        assert "connection_matrix" in results

    def test_shapes(self):
        from unittest.mock import MagicMock

        n_layers, n_heads, d_model, d_head, d_mlp = 2, 2, 16, 8, 32
        svd_top_k = 5

        model = MagicMock()
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.cfg.d_model = d_model
        model.cfg.d_head = d_head
        model.cfg.d_mlp = d_mlp
        model.W_U = torch.randn(d_model, 100)
        model.W_V = torch.randn(n_layers, n_heads, d_model, d_head)
        model.W_O = torch.randn(n_layers, n_heads, d_head, d_model)

        for layer in range(n_layers):
            model.blocks[layer].mlp.W_out = torch.randn(d_mlp, d_model)

        import circuits.circuit_map as cm
        original = cm.get_token_id
        cm.get_token_id = lambda m, w: hash(w) % 100

        examples = [
            {"good_verb": "is", "bad_verb": "are", "clean": "x", "corrupted": "y"},
        ]

        try:
            results = build_importance_map(model, examples, svd_top_k=svd_top_k)
        finally:
            cm.get_token_id = original

        assert results["head_importance"].shape == (n_layers, n_heads)
        assert results["mlp_importance"].shape == (n_layers,)
        assert results["head_task_weights"].shape == (n_layers, n_heads, d_model)
        assert results["svd_spectra"].shape == (n_layers, n_heads, svd_top_k)
        assert results["wv_importance"].shape == (n_layers, n_heads, d_model, d_head)
        assert results["wo_importance"].shape == (n_layers, n_heads, d_head, d_model)
        assert results["mlp_neuron_importance"].shape == (n_layers, d_mlp)
        assert results["connection_matrix"].shape == (n_layers * n_heads, n_layers * n_heads)

    def test_importance_nonnegative(self):
        from unittest.mock import MagicMock

        n_layers, n_heads, d_model, d_head, d_mlp = 2, 2, 16, 8, 32

        model = MagicMock()
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.cfg.d_model = d_model
        model.cfg.d_head = d_head
        model.cfg.d_mlp = d_mlp
        model.W_U = torch.randn(d_model, 100)
        model.W_V = torch.randn(n_layers, n_heads, d_model, d_head)
        model.W_O = torch.randn(n_layers, n_heads, d_head, d_model)

        for layer in range(n_layers):
            model.blocks[layer].mlp.W_out = torch.randn(d_mlp, d_model)

        import circuits.circuit_map as cm
        original = cm.get_token_id
        cm.get_token_id = lambda m, w: hash(w) % 100

        examples = [
            {"good_verb": "is", "bad_verb": "are", "clean": "x", "corrupted": "y"},
        ]

        try:
            results = build_importance_map(model, examples, svd_top_k=5)
        finally:
            cm.get_token_id = original

        assert (results["head_importance"] >= 0).all()
        assert (results["mlp_importance"] >= 0).all()
        assert (results["svd_spectra"] >= 0).all()
