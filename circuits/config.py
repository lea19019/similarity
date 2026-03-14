"""Model registry and path defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_CONFIGS = {
    "gemma-2b": {
        "hf_name": "google/gemma-2b",
        "tl_name": "gemma-2b",
        "n_layers": 18,
        "n_heads": 18,
        "d_model": 2048,
        "key_head": (13, 7),
        "key_neurons": [(13, 2069), (17, 1138)],
    },
    "gemma-7b": {
        "hf_name": "google/gemma-7b",
        "tl_name": "gemma-7b",
        "n_layers": 28,
        "n_heads": 16,
        "d_model": 3072,
        "key_head": None,
        "key_neurons": [],
    },
    "gemma-2-2b": {
        "hf_name": "google/gemma-2-2b",
        "tl_name": "gemma-2-2b",
        "n_layers": 26,
        "n_heads": 8,
        "d_model": 2304,
        "key_head": (19, 3),
        "key_neurons": [],
    },
}
