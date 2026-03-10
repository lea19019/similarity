"""
01_create_dataset.py

Build English and Spanish SVA contrastive-pair datasets.

English: SyntaxGym agr_sv_num_subj-relc (Arora et al., 2024).
Spanish: Parallel dataset constructed using GPT-4 (see paper §3.2).

Output (JSONL, one example per line):
    data/processed/en_sva.jsonl
    data/processed/es_sva.jsonl

Each line:
    {
        "clean":     str,   # prefix where subject agrees with verb
        "corrupted": str,   # same prefix with subject number flipped
        "good_verb": str,   # correct continuation token
        "bad_verb":  str,   # incorrect continuation token
        "lang":      str,   # "en" or "es"
    }
"""
import json
import argparse
from pathlib import Path

# ── English ─────────────────────────────────────────────────────────────────────
def build_english_dataset(out_path: Path) -> None:
    """
    Download and process the SyntaxGym agr_sv_num_subj-relc subset.

    TODO: Figure out the exact loading API for Arora et al. (2024).
    The SyntaxGym hub: https://huggingface.co/datasets/cpllab/syntaxgym
    Subset name may be: agr_sv_num_subj-relc
    """
    # TODO: replace with actual dataset loading
    # from datasets import load_dataset
    # raw = load_dataset("cpllab/syntaxgym", "agr_sv_num_subj-relc")

    # Placeholder structure — iterate over raw and emit one dict per pair
    examples = []

    # Example (from paper):
    examples.append({
        "clean":     "The executive that embarrassed the manager has",
        "corrupted": "The executives that embarrassed the manager have",
        "good_verb": "has",     # singular — agrees with "executive"
        "bad_verb":  "have",
        "lang": "en",
    })

    # TODO: filter out examples where good_verb or bad_verb tokenize
    # into more than one subword (important for clean logit analysis)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} English examples → {out_path}")


# ── Spanish ─────────────────────────────────────────────────────────────────────
def build_spanish_dataset(out_path: Path, use_gpt4: bool = False) -> None:
    """
    Build the Spanish SVA dataset.

    Option A (use_gpt4=True):
        Generate noun/verb pairs with GPT-4, then template them into sentences
        mirroring the English relative-clause structure.

    Option B (use_gpt4=False):
        Load from a hand-curated file in data/raw/es_sva_raw.json.
    """
    examples = []

    if use_gpt4:
        examples = _generate_spanish_with_gpt4()
    else:
        # TODO: load hand-curated data/raw/es_sva_raw.json
        # Example (from paper):
        examples.append({
            "clean":     "El ingeniero que ayudó al cantante era",
            "corrupted": "Los ingenieros que ayudaron al cantante eran",
            "good_verb": "era",     # singular
            "bad_verb":  "eran",
            "lang": "es",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} Spanish examples → {out_path}")


def _generate_spanish_with_gpt4() -> list:
    """
    Use GPT-4 to generate Spanish noun/verb pairs and build contrastive sentences.

    TODO:
    1. Prompt GPT-4 for a list of (singular_noun, plural_noun, sg_verb, pl_verb) tuples.
    2. Template them: "El {sg_noun} que {sg_verb} al {filler} {copula_sg}"
                      "Los {pl_noun} que {pl_verb} al {filler} {copula_pl}"
    3. Filter multi-subword tokens (use Gemma tokenizer).
    """
    from openai import OpenAI
    client = OpenAI()

    # TODO: craft the prompt and parse the response
    raise NotImplementedError("Implement GPT-4 generation here")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "es", "both"], default="both")
    parser.add_argument("--use-gpt4", action="store_true",
                        help="Generate Spanish data with GPT-4 (requires OPENAI_API_KEY)")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.lang in ("en", "both"):
        build_english_dataset(out_dir / "en_sva.jsonl")

    if args.lang in ("es", "both"):
        build_spanish_dataset(out_dir / "es_sva.jsonl", use_gpt4=args.use_gpt4)


if __name__ == "__main__":
    main()
