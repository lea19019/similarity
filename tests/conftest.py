"""Shared fixtures for the circuits test suite."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_examples():
    """A small list of SVA examples for testing."""
    return [
        {
            "clean": "The executive that embarrassed the manager has",
            "corrupted": "The executives that embarrassed the manager have",
            "good_verb": "has",
            "bad_verb": "have",
            "lang": "en",
        },
        {
            "clean": "The doctor that helped the teacher is",
            "corrupted": "The doctors that helped the teacher are",
            "good_verb": "is",
            "bad_verb": "are",
            "lang": "en",
        },
    ]


@pytest.fixture
def sample_es_examples():
    """A small list of Spanish SVA examples."""
    return [
        {
            "clean": "El ingeniero que ayudo al maestro era",
            "corrupted": "Los ingenieros que ayudaron al maestro eran",
            "good_verb": "era",
            "bad_verb": "eran",
            "lang": "es",
        },
    ]


@pytest.fixture
def tmp_jsonl(tmp_path, sample_examples):
    """Write sample examples to a JSONL file and return the path."""
    p = tmp_path / "test_sva.jsonl"
    with open(p, "w") as f:
        for ex in sample_examples:
            f.write(json.dumps(ex) + "\n")
    return p


@pytest.fixture
def sample_fr_examples():
    """A small list of French SVA examples."""
    return [
        {
            "clean": "Le professeur qui voit le client est",
            "corrupted": "Les professeurs qui voient le client sont",
            "good_verb": "est",
            "bad_verb": "sont",
            "lang": "fr",
        },
    ]


@pytest.fixture
def sample_ru_examples():
    """A small list of Russian SVA examples."""
    return [
        {
            "clean": "Учитель который видел директора был",
            "corrupted": "Учителя которые видели директора были",
            "good_verb": "был",
            "bad_verb": "были",
            "lang": "ru",
        },
    ]


@pytest.fixture
def mock_model():
    """
    A mock HookedTransformer that supports to_tokens and basic config.
    Does NOT load any real weights.
    """
    model = MagicMock()
    model.cfg.n_layers = 4
    model.cfg.n_heads = 4
    model.cfg.d_model = 64
    model.cfg.d_mlp = 128

    # Default: to_tokens returns a single token
    def _to_tokens(text, prepend_bos=True):
        # Return a (1, seq_len) tensor; for single-word inputs return (1,1)
        if prepend_bos:
            return torch.tensor([[1, 42]])  # BOS + one token
        else:
            return torch.tensor([[42]])  # single token

    model.to_tokens = MagicMock(side_effect=_to_tokens)
    return model


@pytest.fixture
def patching_npz(tmp_path):
    """Create a mock patching results .npz file and return its path."""
    n_layers, n_heads = 4, 4
    data = {
        "head_out": np.random.rand(n_layers, n_heads).astype(np.float64),
        "attn_out": np.random.rand(n_layers).astype(np.float64),
        "mlp_out": np.random.rand(n_layers).astype(np.float64),
    }
    p = tmp_path / "patching_test.npz"
    np.savez(p, **data)
    return p, data


@pytest.fixture
def dla_npz(tmp_path):
    """Create a mock DLA results .npz file and return its path."""
    n_layers, n_heads = 4, 4
    data = {
        "head_dla": np.random.randn(n_layers, n_heads).astype(np.float64),
        "mlp_dla": np.random.randn(n_layers).astype(np.float64),
    }
    p = tmp_path / "dla_test.npz"
    np.savez(p, **data)
    return p, data


@pytest.fixture
def pca_npz(tmp_path):
    """Create a mock PCA results .npz file and return its path."""
    n = 20
    data = {
        "pc1": np.random.randn(64).astype(np.float64),
        "projections": np.random.randn(n).astype(np.float64),
        "labels": np.array([0, 1] * (n // 2)),
        "langs": np.array(["en"] * (n // 2) + ["es"] * (n // 2)),
    }
    p = tmp_path / "pca_test.npz"
    np.savez(p, **data)
    return p, data


@pytest.fixture
def steering_npz(tmp_path):
    """Create a mock steering results .npz file and return its path."""
    data = {
        "alphas": np.array([0.0, 5.0, 10.0, 20.0]),
        "flip_rate_pos": np.array([0.0, 0.1, 0.3, 0.5]),
        "flip_rate_neg": np.array([0.0, 0.05, 0.15, 0.35]),
    }
    p = tmp_path / "steering_test.npz"
    np.savez(p, **data)
    return p, data


@pytest.fixture
def circuit_map_npz(tmp_path):
    """Create a mock circuit_map results .npz file."""
    n_layers, n_heads, d_model = 4, 4, 64
    data = {
        "head_importance": np.random.rand(n_layers, n_heads).astype(np.float64),
        "mlp_importance": np.random.rand(n_layers).astype(np.float64),
        "head_task_weights": np.random.randn(n_layers, n_heads, d_model).astype(np.float64),
        "svd_spectra": np.sort(np.random.rand(n_layers, n_heads, 10))[:, :, ::-1].copy().astype(np.float64),
    }
    p = tmp_path / "circuit_map_test.npz"
    np.savez(p, **data)
    return p, data


@pytest.fixture
def edge_patching_npz(tmp_path):
    """Create a mock edge patching results .npz file."""
    n_components = 4 * 4 + 4  # heads + MLPs
    labels = [f"L{l}H{h}" for l in range(4) for h in range(4)] + [f"MLP{l}" for l in range(4)]
    data = {
        "node_scores": np.random.rand(n_components).astype(np.float64),
        "edge_scores": np.random.rand(n_components, n_components).astype(np.float64),
        "component_labels": np.array(labels),
    }
    p = tmp_path / "edge_patching_test.npz"
    np.savez(p, **data)
    return p, data


@pytest.fixture
def geometry_npz(tmp_path):
    """Create a mock geometry results .npz file."""
    n_layers, n_langs = 4, 4
    data = {
        "cka_per_layer": np.random.rand(n_layers, n_langs, n_langs).astype(np.float64),
        "convergence": np.random.rand(n_layers).astype(np.float64),
    }
    p = tmp_path / "geometry_test.npz"
    np.savez(p, **data)
    return p, data
