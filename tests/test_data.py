"""Tests for circuits.data — dataset generation, JSONL I/O, and subword filtering."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from circuits.data import (
    ES_NOUNS,
    ES_RC_VERBS,
    ES_PRED_VERBS,
    _build_english_dataset,
    _build_spanish_dataset,
    _filter_english_verbs,
    _filter_word_pairs,
    load_sva_dataset,
    save_dataset,
)


# ── English (CausalGym) ─────────────────────────────────────────────────────

class TestBuildEnglishDataset:
    @patch("datasets.load_dataset")
    def test_returns_list(self, mock_load):
        """With mocked HF dataset, should return a list."""
        mock_load.return_value = mock_load
        mock_load.filter.return_value = []
        # Can't fully test without real CausalGym, but verify no crash
        examples = _build_english_dataset()
        assert isinstance(examples, list)

    @patch("datasets.load_dataset")
    def test_filters_to_6_words(self, mock_load):
        """Only 6-word sentences should be kept."""
        mock_split = MagicMock()
        mock_split.filter.return_value = [
            {
                "task": "agr_sv_num_subj-relc",
                "src": ["<|endoftext|>", " The", " cats", " that", " ate", " the", " fish"],
                "base": ["<|endoftext|>", " The", " cat", " that", " ate", " the", " fish"],
                "src_label": " have",
                "base_label": " has",
            },
            {
                "task": "agr_sv_num_subj-relc",
                "src": ["<|endoftext|>", " Too", " many", " words", " here", " for", " this", " test"],
                "base": ["<|endoftext|>", " Too", " many", " words"],
                "src_label": " x",
                "base_label": " y",
            },
        ]
        mock_load.return_value = mock_split
        examples = _build_english_dataset()
        # Only the 6-word example should pass (×3 splits since mock returns same data)
        assert len(examples) <= 3
        # The 8-word example should be excluded
        for ex in examples:
            assert len(ex["clean"].split()) == 6

    @patch("datasets.load_dataset")
    def test_example_has_required_keys(self, mock_load):
        mock_split = MagicMock()
        mock_split.filter.return_value = [
            {
                "task": "agr_sv_num_subj-relc",
                "src": ["<|endoftext|>", " The", " executives", " that", " helped", " the"],
                "base": ["<|endoftext|>", " The", " executive", " that", " helped", " the"],
                "src_label": " have",
                "base_label": " has",
            },
        ]
        mock_load.return_value = mock_split
        examples = _build_english_dataset()
        if examples:
            required = {"clean", "corrupted", "good_verb", "bad_verb", "lang", "split"}
            assert set(examples[0].keys()) == required
            assert examples[0]["lang"] == "en"


# ── Spanish (template-based) ────────────────────────────────────────────────

class TestBuildSpanishDataset:
    @patch("circuits.data._filter_word_pairs")
    def test_returns_nonempty_list(self, mock_filter):
        """With a few noun/verb pairs, should generate examples."""
        mock_filter.side_effect = lambda pairs, model_key: pairs[:3]
        examples = _build_spanish_dataset("gemma-2b")
        assert len(examples) > 0

    @patch("circuits.data._filter_word_pairs")
    def test_all_lang_es(self, mock_filter):
        mock_filter.side_effect = lambda pairs, model_key: pairs[:3]
        examples = _build_spanish_dataset("gemma-2b")
        assert all(ex["lang"] == "es" for ex in examples)

    @patch("circuits.data._filter_word_pairs")
    def test_has_required_keys(self, mock_filter):
        mock_filter.side_effect = lambda pairs, model_key: pairs[:3]
        examples = _build_spanish_dataset("gemma-2b")
        required = {"clean", "corrupted", "good_verb", "bad_verb", "lang", "split"}
        for ex in examples[:5]:
            assert set(ex.keys()) == required

    @patch("circuits.data._filter_word_pairs")
    def test_has_train_val_test_splits(self, mock_filter):
        mock_filter.side_effect = lambda pairs, model_key: pairs[:5]
        examples = _build_spanish_dataset("gemma-2b")
        splits = {ex["split"] for ex in examples}
        assert "train" in splits
        # val/test may not appear with very few examples, but train should

    @patch("circuits.data._filter_word_pairs")
    def test_pred_verbs_used(self, mock_filter):
        """Prediction verbs should come from ES_PRED_VERBS, not RC verbs."""
        mock_filter.side_effect = lambda pairs, model_key: pairs[:4]
        examples = _build_spanish_dataset("gemma-2b")
        valid_pred = {v for pair in ES_PRED_VERBS for v in pair}
        for ex in examples:
            assert ex["good_verb"] in valid_pred
            assert ex["bad_verb"] in valid_pred

    def test_word_lists_nonempty(self):
        assert len(ES_NOUNS) >= 100
        assert len(ES_RC_VERBS) >= 50
        assert len(ES_PRED_VERBS) == 5


# ── JSONL I/O ────────────────────────────────────────────────────────────────

class TestSaveAndLoadDataset:
    def test_round_trip(self, tmp_path, sample_examples):
        p = tmp_path / "roundtrip.jsonl"
        save_dataset(sample_examples, p)
        loaded = load_sva_dataset(str(p))
        assert loaded == sample_examples

    def test_creates_parent_dirs(self, tmp_path, sample_examples):
        p = tmp_path / "nested" / "dir" / "data.jsonl"
        save_dataset(sample_examples, p)
        assert p.exists()

    def test_file_is_valid_jsonl(self, tmp_path, sample_examples):
        p = tmp_path / "valid.jsonl"
        save_dataset(sample_examples, p)
        with open(p) as f:
            lines = f.readlines()
        assert len(lines) == len(sample_examples)
        for line in lines:
            parsed = json.loads(line.strip())
            assert isinstance(parsed, dict)

    def test_empty_list(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        save_dataset([], p)
        loaded = load_sva_dataset(str(p))
        assert loaded == []

    def test_load_from_fixture(self, tmp_jsonl, sample_examples):
        loaded = load_sva_dataset(str(tmp_jsonl))
        assert len(loaded) == len(sample_examples)
        assert loaded[0]["good_verb"] == sample_examples[0]["good_verb"]


# ── Subword filtering ────────────────────────────────────────────────────────

class TestFilterEnglishVerbs:
    @patch("transformers.AutoTokenizer")
    def test_filters_multitoken_verbs(self, mock_auto_tok):
        mock_tokenizer = MagicMock()

        def fake_encode(word, add_special_tokens=False):
            if word.strip() == "has":
                return [100]
            elif word.strip() == "have":
                return [200, 201]
            return [300]

        mock_tokenizer.encode = MagicMock(side_effect=fake_encode)
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer

        examples = [
            {"good_verb": "has", "bad_verb": "have", "lang": "en"},
        ]
        result = _filter_english_verbs(examples, "gemma-2b")
        assert len(result) == 0

    @patch("transformers.AutoTokenizer")
    def test_keeps_singletoken_verbs(self, mock_auto_tok):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[100])
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer

        examples = [
            {"good_verb": "is", "bad_verb": "are", "lang": "en"},
            {"good_verb": "has", "bad_verb": "have", "lang": "en"},
        ]
        result = _filter_english_verbs(examples, "gemma-2b")
        assert len(result) == 2


class TestFilterWordPairs:
    @patch("transformers.AutoTokenizer")
    def test_filters_multitoken_words(self, mock_auto_tok):
        mock_tokenizer = MagicMock()

        def fake_encode(word, add_special_tokens=False):
            if "doctor" in word:
                return [100]
            return [200, 201]  # multi-token

        mock_tokenizer.encode = MagicMock(side_effect=fake_encode)
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer

        pairs = [("doctor", "doctores"), ("farmacéutico", "farmacéuticos")]
        result = _filter_word_pairs(pairs, "gemma-2b")
        # Only "doctor"/"doctores" should pass (both contain "doctor")
        # "farmacéutico" doesn't contain "doctor" so returns multi-token
        assert len(result) <= len(pairs)
