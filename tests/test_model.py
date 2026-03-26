"""Tests for circuits.model — get_token_id, tokenize_pair, load_model (mocked)."""
from unittest.mock import MagicMock, patch

import pytest
import torch

from circuits.model import get_token_id, load_model, tokenize_pair


class TestGetTokenId:
    def test_single_token_returns_id(self, mock_model):
        """A word that tokenizes to exactly one token should return its int ID."""
        mock_model.to_tokens = MagicMock(
            return_value=torch.tensor([[42]])
        )
        result = get_token_id(mock_model, "has")
        assert result == 42
        mock_model.to_tokens.assert_called_once_with(" has", prepend_bos=False)

    def test_multi_token_raises(self, mock_model):
        """A word tokenized into >1 tokens should raise ValueError."""
        mock_model.to_tokens = MagicMock(
            return_value=torch.tensor([[42, 43]])
        )
        with pytest.raises(ValueError, match="tokenizes into 2 tokens"):
            get_token_id(mock_model, "unfamiliar")

    def test_empty_token_raises(self, mock_model):
        """An empty token sequence should raise ValueError."""
        mock_model.to_tokens = MagicMock(
            return_value=torch.tensor([[]]).reshape(1, 0)
        )
        with pytest.raises(ValueError, match="tokenizes into 0 tokens"):
            get_token_id(mock_model, "")


class TestTokenizePair:
    def test_returns_three_items(self, mock_model):
        mock_model.to_tokens = MagicMock(
            side_effect=lambda text, **kw: (
                torch.tensor([[1, 2, 3, 4]])
                if kw.get("prepend_bos", True)
                else torch.tensor([[99]])
            )
        )
        tokens, good_id, bad_id = tokenize_pair(
            mock_model, "The cat is", "is", "are"
        )
        assert isinstance(tokens, torch.Tensor)
        assert isinstance(good_id, int)
        assert isinstance(bad_id, int)

    def test_good_and_bad_ids_differ(self, mock_model):
        call_count = [0]

        def _to_tokens(text, prepend_bos=True):
            if prepend_bos:
                return torch.tensor([[1, 2, 3]])
            call_count[0] += 1
            return torch.tensor([[10 + call_count[0]]])

        mock_model.to_tokens = MagicMock(side_effect=_to_tokens)
        _, good_id, bad_id = tokenize_pair(mock_model, "The cat", "is", "are")
        assert good_id != bad_id


class TestLoadModel:
    @patch("circuits.model.HookedTransformer")
    def test_load_model_calls_from_pretrained(self, mock_ht):
        mock_instance = MagicMock()
        mock_ht.from_pretrained.return_value = mock_instance

        model = load_model("gemma-2b", device="cpu")

        mock_ht.from_pretrained.assert_called_once_with(
            "gemma-2b",
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device="cpu",
        )
        mock_instance.eval.assert_called_once()
        assert model is mock_instance

    @patch("circuits.model.HookedTransformer")
    def test_load_model_invalid_key(self, mock_ht):
        with pytest.raises(KeyError):
            load_model("nonexistent-model", device="cpu")
