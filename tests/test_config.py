"""Tests for circuits.config — path constants and model registry."""
from pathlib import Path

from circuits.config import (
    DATA_DIR,
    FIGURES_DIR,
    MODEL_CONFIGS,
    PROJECT_ROOT,
    RESULTS_DIR,
)


class TestPathConstants:
    def test_project_root_is_absolute(self):
        assert PROJECT_ROOT.is_absolute()

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_data_dir_under_project(self):
        assert str(DATA_DIR).startswith(str(PROJECT_ROOT))
        assert DATA_DIR == PROJECT_ROOT / "data" / "processed"

    def test_results_dir_under_project(self):
        assert RESULTS_DIR == PROJECT_ROOT / "results"

    def test_figures_dir_under_results(self):
        assert FIGURES_DIR == RESULTS_DIR / "figures"

    def test_all_paths_are_path_objects(self):
        for p in (PROJECT_ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR):
            assert isinstance(p, Path)


class TestModelConfigs:
    EXPECTED_KEYS = {"gemma-2b", "gemma-7b", "gemma-2-2b"}
    REQUIRED_FIELDS = {
        "hf_name", "tl_name", "n_layers", "n_heads", "d_model",
        "key_head", "key_neurons",
    }

    def test_all_expected_keys_present(self):
        assert set(MODEL_CONFIGS.keys()) == self.EXPECTED_KEYS

    def test_each_config_has_required_fields(self):
        for key, cfg in MODEL_CONFIGS.items():
            for field in self.REQUIRED_FIELDS:
                assert field in cfg, f"{key} missing field '{field}'"

    def test_n_layers_positive(self):
        for key, cfg in MODEL_CONFIGS.items():
            assert cfg["n_layers"] > 0, f"{key}: n_layers must be positive"

    def test_n_heads_positive(self):
        for key, cfg in MODEL_CONFIGS.items():
            assert cfg["n_heads"] > 0, f"{key}: n_heads must be positive"

    def test_d_model_positive(self):
        for key, cfg in MODEL_CONFIGS.items():
            assert cfg["d_model"] > 0, f"{key}: d_model must be positive"

    def test_key_head_is_tuple_or_none(self):
        for key, cfg in MODEL_CONFIGS.items():
            kh = cfg["key_head"]
            assert kh is None or (isinstance(kh, tuple) and len(kh) == 2), (
                f"{key}: key_head should be None or a 2-tuple"
            )

    def test_key_neurons_is_list(self):
        for key, cfg in MODEL_CONFIGS.items():
            assert isinstance(cfg["key_neurons"], list), (
                f"{key}: key_neurons should be a list"
            )

    def test_gemma_2b_specific_values(self):
        cfg = MODEL_CONFIGS["gemma-2b"]
        assert cfg["n_layers"] == 18
        assert cfg["n_heads"] == 18
        assert cfg["d_model"] == 2048
        assert cfg["key_head"] == (13, 7)
        assert len(cfg["key_neurons"]) == 2
