"""Model registry and path defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Language registry for cross-lingual experiments
# Single-token languages (EN/ES/FR/RU) use exact verb token matching.
# Multi-token languages (TR/SW/QU) use first-subword token matching.
LANG_CONFIGS = {
    "en": {"name": "English", "family": "Germanic", "order": "SVO", "multi_token": False},
    "es": {"name": "Spanish", "family": "Romance", "order": "SVO", "multi_token": False},
    "fr": {"name": "French", "family": "Romance", "order": "SVO", "multi_token": False},
    "ru": {"name": "Russian", "family": "Slavic", "order": "SVO", "multi_token": False},
    "tr": {"name": "Turkish", "family": "Turkic", "order": "SOV", "multi_token": True},
    "sw": {"name": "Swahili", "family": "Bantu", "order": "SVO", "multi_token": True},
    "qu": {"name": "Quechua", "family": "Quechuan", "order": "SOV", "multi_token": True},
}

ALL_LANGS = list(LANG_CONFIGS.keys())

MODEL_CONFIGS = {
    "gemma-2b": {
        "hf_name": "google/gemma-2b",
        "tl_name": "gemma-2b",
        "n_layers": 18,
        "n_heads": 18,
        "d_model": 2048,
        # key_head: the attention head most causally important for SVA (found via patching)
        "key_head": (13, 7),
        # key_neurons: MLP neurons with highest DLA for verb-number signal
        # (13, 2069) = neuron 2069 in MLP layer 13; (17, 1138) = neuron 1138 in MLP layer 17
        "key_neurons": [(13, 2069), (17, 1138)],
    },
    "gemma-7b": {
        "hf_name": "google/gemma-7b",
        "tl_name": "gemma-7b",
        "n_layers": 28,
        "n_heads": 16,
        "d_model": 3072,
        # Different architecture means different key heads — not yet identified for 7b
        "key_head": None,
        "key_neurons": [],
    },
    "gemma-2-2b": {
        "hf_name": "google/gemma-2-2b",
        "tl_name": "gemma-2-2b",
        "n_layers": 26,
        "n_heads": 8,
        "d_model": 2304,
        # Gemma-2 has a different architecture, so the key head shifts to a later layer
        "key_head": (19, 3),
        "key_neurons": [],
    },
    "bloom-3b": {
        "hf_name": "bigscience/bloom-3b",
        "tl_name": "bloom-3b",
        "n_layers": 30,
        "n_heads": 32,
        "d_model": 2560,
        # key_head: to be identified via patching — run patching first
        "key_head": None,
        "key_neurons": [],
    },
}
