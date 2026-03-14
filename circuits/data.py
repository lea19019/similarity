"""
Dataset creation and loading for SVA contrastive pairs.

English and Spanish datasets are built from curated template lists.
Examples are filtered to ensure verbs tokenize to a single Gemma subword.

Usage:
    uv run python -m circuits.data --lang both --out-dir data/processed --model gemma-2b
"""
import argparse
import json
from itertools import product
from pathlib import Path
from typing import List

from circuits.config import DATA_DIR, MODEL_CONFIGS


# ── English templates ────────────────────────────────────────────────────────

EN_SUBJECTS = [
    ("executive", "executives"),
    ("officer", "officers"),
    ("senator", "senators"),
    ("doctor", "doctors"),
    ("lawyer", "lawyers"),
    ("farmer", "farmers"),
    ("driver", "drivers"),
    ("singer", "singers"),
    ("painter", "painters"),
    ("worker", "workers"),
    ("leader", "leaders"),
    ("writer", "writers"),
    ("hunter", "hunters"),
    ("banker", "bankers"),
    ("dealer", "dealers"),
]

EN_RC_VERBS = [
    "embarrassed", "helped", "praised", "blamed", "fired",
    "hired", "ignored", "thanked", "tricked", "visited",
]

EN_OBJECTS = [
    "manager", "director", "captain", "teacher", "pilot",
    "guard", "clerk", "coach", "agent", "vendor",
]

EN_MAIN_VERBS = [
    ("has", "have"),
    ("was", "were"),
    ("is", "are"),
    ("does", "do"),
]


def _build_english_examples() -> List[dict]:
    """Generate English SVA contrastive pairs from templates."""
    examples = []
    for (sg_subj, pl_subj), rc_verb, obj, (sg_verb, pl_verb) in product(
        EN_SUBJECTS, EN_RC_VERBS, EN_OBJECTS, EN_MAIN_VERBS
    ):
        if sg_subj == obj or pl_subj == obj:
            continue
        examples.append({
            "clean": f"The {sg_subj} that {rc_verb} the {obj} {sg_verb}",
            "corrupted": f"The {pl_subj} that {rc_verb} the {obj} {pl_verb}",
            "good_verb": sg_verb,
            "bad_verb": pl_verb,
            "lang": "en",
        })
    return examples


# ── Spanish templates ────────────────────────────────────────────────────────

ES_SUBJECTS = [
    ("ingeniero", "ingenieros"),
    ("abogado", "abogados"),
    ("soldado", "soldados"),
    ("pintor", "pintores"),
    ("doctor", "doctores"),
    ("escritor", "escritores"),
    ("jugador", "jugadores"),
    ("profesor", "profesores"),
    ("pastor", "pastores"),
    ("inventor", "inventores"),
    ("senador", "senadores"),
    ("director", "directores"),
    ("conductor", "conductores"),
    ("cantante", "cantantes"),
    ("piloto", "pilotos"),
]

# RC verb must agree: (sg form, pl form)
ES_RC_VERBS = [
    ("ayudó", "ayudaron"),
    ("culpó", "culparon"),
    ("llamó", "llamaron"),
    ("visitó", "visitaron"),
    ("ignoró", "ignoraron"),
    ("saludó", "saludaron"),
    ("criticó", "criticaron"),
    ("admiró", "admiraron"),
    ("buscó", "buscaron"),
    ("salvó", "salvaron"),
]

ES_OBJECTS = [
    "cantante", "maestro", "piloto", "guardia", "alcalde",
    "soldado", "agente", "poeta", "pintor", "pastor",
]

ES_MAIN_VERBS = [
    ("era", "eran"),
    ("fue", "fueron"),
    ("tiene", "tienen"),
    ("hace", "hacen"),
]


def _build_spanish_examples() -> List[dict]:
    """Generate Spanish SVA contrastive pairs from templates."""
    examples = []
    for (sg_subj, pl_subj), (rc_sg, rc_pl), obj, (sg_verb, pl_verb) in product(
        ES_SUBJECTS, ES_RC_VERBS, ES_OBJECTS, ES_MAIN_VERBS
    ):
        if sg_subj == obj or pl_subj == obj:
            continue
        examples.append({
            "clean": f"El {sg_subj} que {rc_sg} al {obj} {sg_verb}",
            "corrupted": f"Los {pl_subj} que {rc_pl} al {obj} {pl_verb}",
            "good_verb": sg_verb,
            "bad_verb": pl_verb,
            "lang": "es",
        })
    return examples


# ── Subword filtering ────────────────────────────────────────────────────────

def _filter_single_token(examples: List[dict], model_key: str) -> List[dict]:
    """Keep only examples where both good_verb and bad_verb are single Gemma subwords."""
    import os
    from transformers import AutoTokenizer

    hf_name = MODEL_CONFIGS[model_key]["hf_name"]
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, token=token)

    def is_single_token(word: str) -> bool:
        ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        return len(ids) == 1

    filtered = [
        ex for ex in examples
        if is_single_token(ex["good_verb"]) and is_single_token(ex["bad_verb"])
    ]
    return filtered


# ── I/O ──────────────────────────────────────────────────────────────────────

def save_dataset(examples: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples → {path}")


def load_sva_dataset(path: str) -> List[dict]:
    """
    Load a processed SVA dataset (JSONL).
    Each line: {"clean": str, "corrupted": str, "good_verb": str, "bad_verb": str, "lang": str}
    """
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate SVA contrastive-pair datasets")
    parser.add_argument("--lang", choices=["en", "es", "both"], default="both")
    parser.add_argument("--out-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--model", default="gemma-2b",
                        help="Model key for subword filtering (default: gemma-2b)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip subword filtering (faster, no model download needed)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.lang in ("en", "both"):
        examples = _build_english_examples()
        print(f"English: {len(examples)} candidates from templates")
        if not args.no_filter:
            examples = _filter_single_token(examples, args.model)
            print(f"English: {len(examples)} after single-token filtering")
        save_dataset(examples, out_dir / "en_sva.jsonl")

    if args.lang in ("es", "both"):
        examples = _build_spanish_examples()
        print(f"Spanish: {len(examples)} candidates from templates")
        if not args.no_filter:
            examples = _filter_single_token(examples, args.model)
            print(f"Spanish: {len(examples)} after single-token filtering")
        save_dataset(examples, out_dir / "es_sva.jsonl")


if __name__ == "__main__":
    main()
