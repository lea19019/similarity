"""Tests for circuits.data — template generation, JSONL I/O, and subword filtering."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from circuits.data import (
    EN_MAIN_VERBS,
    EN_OBJECTS,
    EN_RC_VERBS,
    EN_SUBJECTS,
    ES_MAIN_VERBS,
    ES_OBJECTS,
    ES_RC_VERBS,
    ES_SUBJECTS,
    _build_english_examples,
    _build_spanish_examples,
    _filter_single_token,
    load_sva_dataset,
    save_dataset,
)


class TestBuildEnglishExamples:
    def test_returns_nonempty_list(self):
        examples = _build_english_examples()
        assert len(examples) > 0

    def test_example_has_required_keys(self):
        examples = _build_english_examples()
        required = {"clean", "corrupted", "good_verb", "bad_verb", "lang"}
        for ex in examples[:5]:
            assert set(ex.keys()) == required

    def test_all_lang_en(self):
        examples = _build_english_examples()
        assert all(ex["lang"] == "en" for ex in examples)

    def test_clean_contains_singular_subject(self):
        examples = _build_english_examples()
        ex = examples[0]
        assert ex["clean"].startswith("The ")

    def test_no_self_reference(self):
        """Subject and object should never be the same word."""
        examples = _build_english_examples()
        for ex in examples:
            # The clean prompt looks like "The {sg_subj} that {rc_verb} the {obj} {sg_verb}"
            parts = ex["clean"].split()
            subject = parts[1]
            obj = parts[parts.index("the", 2) + 1]  # second "the"
            assert subject != obj

    def test_count_upper_bound(self):
        """Number of examples should be <= full cartesian product (some filtered by self-ref)."""
        full = len(EN_SUBJECTS) * len(EN_RC_VERBS) * len(EN_OBJECTS) * len(EN_MAIN_VERBS)
        examples = _build_english_examples()
        assert len(examples) <= full
        assert len(examples) > 0

    def test_good_bad_verbs_from_templates(self):
        examples = _build_english_examples()
        valid_sg = {v[0] for v in EN_MAIN_VERBS}
        valid_pl = {v[1] for v in EN_MAIN_VERBS}
        for ex in examples:
            assert ex["good_verb"] in valid_sg
            assert ex["bad_verb"] in valid_pl


class TestBuildSpanishExamples:
    def test_returns_nonempty_list(self):
        examples = _build_spanish_examples()
        assert len(examples) > 0

    def test_all_lang_es(self):
        examples = _build_spanish_examples()
        assert all(ex["lang"] == "es" for ex in examples)

    def test_example_has_required_keys(self):
        examples = _build_spanish_examples()
        required = {"clean", "corrupted", "good_verb", "bad_verb", "lang"}
        for ex in examples[:5]:
            assert set(ex.keys()) == required

    def test_clean_starts_with_el(self):
        examples = _build_spanish_examples()
        assert all(ex["clean"].startswith("El ") for ex in examples)

    def test_corrupted_starts_with_los(self):
        examples = _build_spanish_examples()
        assert all(ex["corrupted"].startswith("Los ") for ex in examples)

    def test_count_upper_bound(self):
        full = len(ES_SUBJECTS) * len(ES_RC_VERBS) * len(ES_OBJECTS) * len(ES_MAIN_VERBS)
        examples = _build_spanish_examples()
        assert len(examples) <= full
        assert len(examples) > 0


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


class TestFilterSingleToken:
    @patch("transformers.AutoTokenizer")
    def test_filters_multitoken_verbs(self, mock_auto_tok):
        """Verbs that tokenize to >1 subwords should be removed."""
        mock_tokenizer = MagicMock()

        def fake_encode(word, add_special_tokens=False):
            # " has" -> single token, " have" -> multi token
            if word.strip() == "has":
                return [100]
            elif word.strip() == "have":
                return [200, 201]  # multi-token
            return [300]

        mock_tokenizer.encode = MagicMock(side_effect=fake_encode)
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer

        examples = [
            {"good_verb": "has", "bad_verb": "have", "lang": "en"},
        ]
        result = _filter_single_token(examples, "gemma-2b")
        # "have" is multi-token, so this example should be filtered out
        assert len(result) == 0

    @patch("transformers.AutoTokenizer")
    def test_keeps_singletoken_verbs(self, mock_auto_tok):
        """Both verbs single-token => kept."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[100])
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer

        examples = [
            {"good_verb": "is", "bad_verb": "are", "lang": "en"},
            {"good_verb": "has", "bad_verb": "have", "lang": "en"},
        ]
        result = _filter_single_token(examples, "gemma-2b")
        assert len(result) == 2
