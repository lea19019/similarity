"""Tests for circuits.wanda — Wanda-style activation × weight importance."""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from circuits.wanda import compute_wanda_importance


class TestComputeWandaImportance:
    def test_output_keys(self):
        n_layers, n_heads, d_model, d_head, d_mlp = 2, 2, 16, 8, 32
        model = MagicMock()
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.cfg.d_model = d_model
        model.cfg.d_head = d_head
        model.cfg.d_mlp = d_mlp
        model.W_V = torch.randn(n_layers, n_heads, d_model, d_head)
        model.W_O = torch.randn(n_layers, n_heads, d_head, d_model)
        for layer in range(n_layers):
            model.blocks[layer].mlp.W_out = torch.randn(d_mlp, d_model)

        act_norms = {
            "attn_input_norms": np.random.rand(n_layers, d_model),
            "mlp_input_norms": np.random.rand(n_layers, d_model),
        }

        results = compute_wanda_importance(model, act_norms)
        assert "wv_wanda" in results
        assert "wo_wanda" in results
        assert "mlp_wanda" in results

    def test_shapes(self):
        n_layers, n_heads, d_model, d_head, d_mlp = 2, 2, 16, 8, 32
        model = MagicMock()
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.cfg.d_model = d_model
        model.cfg.d_head = d_head
        model.cfg.d_mlp = d_mlp
        model.W_V = torch.randn(n_layers, n_heads, d_model, d_head)
        model.W_O = torch.randn(n_layers, n_heads, d_head, d_model)
        for layer in range(n_layers):
            model.blocks[layer].mlp.W_out = torch.randn(d_mlp, d_model)

        act_norms = {
            "attn_input_norms": np.random.rand(n_layers, d_model),
            "mlp_input_norms": np.random.rand(n_layers, d_model),
        }

        results = compute_wanda_importance(model, act_norms)
        assert results["wv_wanda"].shape == (n_layers, n_heads, d_model, d_head)
        assert results["wo_wanda"].shape == (n_layers, n_heads, d_head, d_model)
        assert results["mlp_wanda"].shape == (n_layers, d_mlp)

    def test_nonnegative(self):
        n_layers, n_heads, d_model, d_head, d_mlp = 2, 2, 16, 8, 32
        model = MagicMock()
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.cfg.d_model = d_model
        model.cfg.d_head = d_head
        model.cfg.d_mlp = d_mlp
        model.W_V = torch.randn(n_layers, n_heads, d_model, d_head)
        model.W_O = torch.randn(n_layers, n_heads, d_head, d_model)
        for layer in range(n_layers):
            model.blocks[layer].mlp.W_out = torch.randn(d_mlp, d_model)

        act_norms = {
            "attn_input_norms": np.abs(np.random.rand(n_layers, d_model)),
            "mlp_input_norms": np.abs(np.random.rand(n_layers, d_model)),
        }

        results = compute_wanda_importance(model, act_norms)
        assert (results["wv_wanda"] >= 0).all()
        assert (results["mlp_wanda"] >= 0).all()
