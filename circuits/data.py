"""
Dataset creation and loading for SVA contrastive pairs.

Matches the original paper (Ferrando & Costa-jussà, 2024) exactly:
  - English: CausalGym (aryaman/causalgym), subset agr_sv_num_subj-relc, 6-word filter
  - Spanish: GPT-4-curated noun/verb lists from the paper's repo, same template + seed

Usage (run on login node — needs internet for CausalGym download):
    uv run python -m circuits.data --lang both --out-dir data/processed --model gemma-2b
"""
import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np

from circuits.config import DATA_DIR, MODEL_CONFIGS

# Match the paper's random seed for reproducibility
SEED = 10


# ── Spanish word lists (from paper's repo) ──────────────────────────────────
# Source: https://github.com/javiferran/circuits_languages/tree/main/datasets
#
# 100 noun pairs (singular/plural), GPT-4-generated, manually curated.
# At runtime, pairs whose singular or plural form tokenizes to >1 subword
# are filtered out (matching the paper's read_files() function).
ES_NOUNS = [
    ("doctor", "doctores"), ("abogado", "abogados"), ("maestro", "maestros"),
    ("empleado", "empleados"), ("presidente", "presidentes"), ("chef", "chefs"),
    ("policía", "policías"), ("bombero", "bomberos"), ("agricultor", "agricultores"),
    ("pintor", "pintores"), ("músico", "músicos"), ("bailarín", "bailarines"),
    ("escritor", "escritores"), ("actor", "actores"), ("mecánico", "mecánicos"),
    ("jardinero", "jardineros"), ("fotógrafo", "fotógrafos"), ("editor", "editores"),
    ("director", "directores"), ("arquitecto", "arquitectos"), ("sacerdote", "sacerdotes"),
    ("alcalde", "alcaldes"), ("diputado", "diputados"), ("senador", "senadores"),
    ("rey", "reyes"), ("príncipe", "príncipes"), ("vendedor", "vendedores"),
    ("gerente", "gerentes"), ("economista", "economistas"), ("contable", "contables"),
    ("científico", "científicos"), ("investigador", "investigadores"),
    ("psicólogo", "psicólogos"), ("terapeuta", "terapeutas"), ("enfermero", "enfermeros"),
    ("dentista", "dentistas"), ("cirujano", "cirujanos"), ("anestesista", "anestesistas"),
    ("farmacéutico", "farmacéuticos"), ("nutricionista", "nutricionistas"),
    ("entrenador", "entrenadores"), ("árbitro", "árbitros"), ("futbolista", "futbolistas"),
    ("basquetbolista", "basquetbolistas"), ("tenista", "tenistas"),
    ("nadador", "nadadores"), ("ciclista", "ciclistas"), ("esquiador", "esquiadores"),
    ("corredor", "corredores"), ("boxeador", "boxeadores"), ("periodista", "periodistas"),
    ("locutor", "locutores"), ("camarógrafo", "camarógrafos"),
    ("productor", "productores"), ("guionista", "guionistas"), ("escultor", "escultores"),
    ("dibujante", "dibujantes"), ("diseñador", "diseñadores"), ("tatuador", "tatuadores"),
    ("zapatero", "zapateros"), ("sastre", "sastres"), ("plomero", "plomeros"),
    ("electricista", "electricistas"), ("carpintero", "carpinteros"),
    ("albañil", "albañiles"), ("bibliotecario", "bibliotecarios"),
    ("criminólogo", "criminólogos"), ("sociólogo", "sociólogos"),
    ("historiador", "historiadores"), ("filósofo", "filósofos"),
    ("traductor", "traductores"), ("interprete", "intérpretes"),
    ("embajador", "embajadores"), ("cónsul", "cónsules"), ("técnico", "técnicos"),
    ("programador", "programadores"), ("analista", "analistas"),
    ("consultor", "consultores"), ("asistente", "asistentes"), ("socio", "socios"),
    ("comerciante", "comerciantes"), ("campesino", "campesinos"),
    ("minero", "mineros"), ("pescador", "pescadores"), ("marinero", "marineros"),
    ("piloto", "pilotos"), ("astronauta", "astronautas"),
    ("meteorólogo", "meteorólogos"), ("geólogo", "geólogos"), ("óptico", "ópticos"),
    ("kinésico", "kinésicos"), ("psiquiatra", "psiquiatras"),
    ("ortodoncista", "ortodoncistas"), ("cadete", "cadetes"),
    ("vigilante", "vigilantes"), ("guardia", "guardias"), ("escolta", "escoltas"),
    ("agente", "agentes"), ("detective", "detectives"), ("espía", "espías"),
    # Hardcoded extras from the paper's code
    ("cantante", "cantantes"), ("ingeniero", "ingenieros"),
    ("ministro", "ministros"),
]

# 52 preterite RC verb pairs (plausible subset curated from 100).
# These go inside the relative clause: "que {verb_sg} al ..."
ES_RC_VERBS = [
    ("habló", "hablaron"), ("leyó", "leyeron"), ("vendió", "vendieron"),
    ("escribió", "escribieron"), ("pagó", "pagaron"), ("dijo", "dijeron"),
    ("trajo", "trajeron"), ("vio", "vieron"), ("puso", "pusieron"),
    ("dio", "dieron"), ("envió", "enviaron"), ("conoció", "conocieron"),
    ("tuvo", "tuvieron"), ("encontró", "encontraron"), ("escuchó", "escucharon"),
    ("cantó", "cantaron"), ("estudió", "estudiaron"), ("explicó", "explicaron"),
    ("miró", "miraron"), ("olvidó", "olvidaron"), ("pasó", "pasaron"),
    ("permitió", "permitieron"), ("probó", "probaron"), ("recordó", "recordaron"),
    ("sacó", "sacaron"), ("usó", "usaron"), ("visitó", "visitaron"),
    ("cerró", "cerraron"), ("mostró", "mostraron"), ("ofreció", "ofrecieron"),
    ("preparó", "prepararon"), ("tocó", "tocaron"), ("tosió", "tosieron"),
    ("esperó", "esperaron"), ("limpió", "limpiaron"), ("negó", "negaron"),
    ("respondió", "respondieron"), ("siguió", "siguieron"), ("cambió", "cambiaron"),
    ("lavó", "lavaron"), ("mandó", "mandaron"), ("mató", "mataron"),
    ("pidió", "pidieron"), ("vendió", "vendieron"), ("ganó", "ganaron"),
    ("incluyó", "incluyeron"), ("insistió", "insistieron"),
    ("introdujo", "introdujeron"), ("mendigó", "mendigaron"),
    ("mostró", "mostraron"), ("movió", "movieron"), ("negó", "negaron"),
    ("operó", "operaron"), ("roció", "regaron"), ("trajo", "trajeron"),
    ("votó", "votaron"),
    # Hardcoded extra from paper's code
    ("tuvo", "tuvieron"),
]

# 5 prediction verb pairs — these are the verbs the model must predict at the
# end of the sentence. They are SEPARATE from the RC verbs above.
ES_PRED_VERBS = [
    ("es", "son"),
    ("era", "eran"),
    ("fue", "fueron"),
    ("tuvo", "tuvieron"),
    ("tiene", "tienen"),
]


# ── English dataset (CausalGym) ─────────────────────────────────────────────

def _build_english_dataset() -> List[dict]:
    """
    Load English SVA examples from CausalGym (aryaman/causalgym).

    Matches the paper: filter to agr_sv_num_subj-relc task, keep only 6-word
    sentences, skip compound words. Returns contrastive pairs in our format.
    """
    from datasets import load_dataset

    examples = []
    for split in ["train", "validation", "test"]:
        hf_dataset = load_dataset("aryaman/causalgym", split=split)
        hf_dataset = hf_dataset.filter(
            lambda ex: ex["task"] == "agr_sv_num_subj-relc"
        )

        for item in hf_dataset:
            # Skip examples with compound words (multi-word tokens)
            has_compound = False
            for sent_type in ["src", "base"]:
                for word in item[sent_type]:
                    if len(word.split()) > 1:
                        has_compound = True
                        break
            if has_compound:
                continue

            # Join words, remove BOS token
            src = "".join(item["src"]).replace("<|endoftext|>", "")
            base = "".join(item["base"]).replace("<|endoftext|>", "")

            # Paper only keeps 6-word sentences
            if len(src.split()) != 6 or len(base.split()) != 6:
                continue

            # Determine which is singular vs plural based on second word
            # (the subject noun). Paper's heuristic: ends in 's' = plural.
            base_is_plural = base.split()[1].endswith("s")

            if base_is_plural:
                # base = plural sentence, src = singular sentence
                examples.append({
                    "clean": src.strip(),
                    "corrupted": base.strip(),
                    "good_verb": item["src_label"].strip(),
                    "bad_verb": item["base_label"].strip(),
                    "lang": "en",
                    "split": split,
                })
            else:
                # base = singular sentence (clean)
                examples.append({
                    "clean": base.strip(),
                    "corrupted": src.strip(),
                    "good_verb": item["base_label"].strip(),
                    "bad_verb": item["src_label"].strip(),
                    "lang": "en",
                    "split": split,
                })

    return examples


# ── Spanish dataset (template-based, matching paper) ────────────────────────

def _filter_word_pairs(pairs: list, model_key: str) -> list:
    """
    Filter noun/verb pairs to those where both forms are single subwords.

    Matches the paper's read_files() — both singular and plural must tokenize
    to exactly 2 tokens (BOS + content) via model.to_tokens(f' {word}').
    We use the HF tokenizer directly (1 token with add_special_tokens=False).
    """
    import os
    from transformers import AutoTokenizer

    hf_name = MODEL_CONFIGS[model_key]["hf_name"]
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, token=token)

    def is_single_token(word: str) -> bool:
        ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        return len(ids) == 1

    return [
        (sg, pl) for sg, pl in pairs
        if is_single_token(sg) and is_single_token(pl)
    ]


def _build_spanish_dataset(model_key: str = "gemma-2b") -> List[dict]:
    """
    Generate Spanish SVA contrastive pairs matching the paper exactly.

    Template: "El {noun_sg} que {rc_verb_sg} al {noun2_sg} {pred_verb_sg}"
    vs:       "Los {noun_pl} que {rc_verb_pl} al {noun2_sg} {pred_verb_pl}"

    Key detail: the RC verb and prediction verb are DIFFERENT. The RC verb
    is from the preterite list (52 pairs); the prediction verb is from a
    separate set of 5 copula/auxiliary pairs (es/son, era/eran, etc.).

    Uses seed=10 and 70/15/15 train/val/test split, matching the paper.
    """
    # Filter word pairs by single-token constraint
    noun_pairs = _filter_word_pairs(
        list(set(ES_NOUNS)), model_key
    )
    rc_verb_pairs = _filter_word_pairs(
        list(set(ES_RC_VERBS)), model_key
    )
    pred_verb_pairs = _filter_word_pairs(
        list(set(ES_PRED_VERBS)), model_key
    )

    noun_sg = [f" {n[0]}" for n in noun_pairs]
    noun_pl = [f" {n[1]}" for n in noun_pairs]
    rc_sg = [f" {v[0]}" for v in rc_verb_pairs]
    rc_pl = [f" {v[1]}" for v in rc_verb_pairs]
    pred_sg = [f" {v[0]}" for v in pred_verb_pairs]
    pred_pl = [f" {v[1]}" for v in pred_verb_pairs]

    # All permutations where subject noun != object noun (matching paper: k != i)
    perms = [
        [i, j, k]
        for i in range(len(noun_sg))
        for j in range(len(rc_sg))
        for k in range(len(noun_sg))
        if k != i
    ]
    perm_array = np.array(perms)

    np.random.seed(SEED)
    np.random.shuffle(perm_array)
    random.seed(SEED)

    examples = []
    for i, j, k in perm_array:
        # Build both sentence forms
        sent_pl = f"Los{noun_pl[k]} que{rc_pl[j]} al{noun_sg[i]}"
        sent_sg = f"El{noun_sg[k]} que{rc_sg[j]} al{noun_sg[i]}"

        # Pick a prediction verb different from the RC verb (matching paper)
        verb_indices = list(range(len(pred_sg)))
        available = [idx for idx in verb_indices if idx != j % len(verb_indices)]
        pred_idx = np.random.choice(available)

        # Randomly assign which is base vs src (matching paper's 50/50 swap)
        rdm = int(round(random.uniform(0, 1), 0))

        if rdm == 0:
            # base = singular, src = plural
            examples.append({
                "clean": sent_sg.strip(),
                "corrupted": sent_pl.strip(),
                "good_verb": pred_sg[pred_idx].strip(),
                "bad_verb": pred_pl[pred_idx].strip(),
                "lang": "es",
            })
        else:
            # base = plural, src = singular
            examples.append({
                "clean": sent_pl.strip(),
                "corrupted": sent_sg.strip(),
                "good_verb": pred_pl[pred_idx].strip(),
                "bad_verb": pred_sg[pred_idx].strip(),
                "lang": "es",
            })

    # 70/15/15 train/val/test split (matching paper)
    n = len(examples)
    train_end = n * 70 // 100
    val_end = n * 85 // 100

    for ex in examples[:train_end]:
        ex["split"] = "train"
    for ex in examples[train_end:val_end]:
        ex["split"] = "validation"
    for ex in examples[val_end:]:
        ex["split"] = "test"

    return examples


# ── Subword filtering (English verbs) ───────────────────────────────────────

def _filter_english_verbs(examples: List[dict], model_key: str) -> List[dict]:
    """Filter English examples to those where both verbs are single subwords."""
    import os
    from transformers import AutoTokenizer

    hf_name = MODEL_CONFIGS[model_key]["hf_name"]
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, token=token)

    def is_single_token(word: str) -> bool:
        ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        return len(ids) == 1

    return [
        ex for ex in examples
        if is_single_token(ex["good_verb"]) and is_single_token(ex["bad_verb"])
    ]


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
    Each line: {"clean": str, "corrupted": str, "good_verb": str,
                "bad_verb": str, "lang": str, "split": str}
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
    parser.add_argument("--split", choices=["train", "validation", "test", "all"],
                        default="all", help="Which split to save (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.lang in ("en", "both"):
        examples = _build_english_dataset()
        print(f"English: {len(examples)} examples from CausalGym (after 6-word filter)")
        examples = _filter_english_verbs(examples, args.model)
        print(f"English: {len(examples)} after single-token verb filtering")
        if args.split != "all":
            examples = [ex for ex in examples if ex["split"] == args.split]
            print(f"English: {len(examples)} in '{args.split}' split")
        save_dataset(examples, out_dir / "en_sva.jsonl")

    if args.lang in ("es", "both"):
        examples = _build_spanish_dataset(model_key=args.model)
        print(f"Spanish: {len(examples)} examples (after single-token noun/verb filtering)")
        if args.split != "all":
            examples = [ex for ex in examples if ex["split"] == args.split]
            print(f"Spanish: {len(examples)} in '{args.split}' split")
        save_dataset(examples, out_dir / "es_sva.jsonl")


if __name__ == "__main__":
    main()
