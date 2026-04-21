import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import json
import logging
import sys

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MEDQA_HF_DATASET = "GBaker/MedQA-USMLE-4-options"
MEDQA_DIR = os.path.join(config.RAW_DIR, "medqa_hf")
MEDQA_TEST_CSV = os.path.join(config.PROCESSED_DIR, "medqa_test_queries.csv")
MEDQA_TRAINING_CSV = os.path.join(
    config.BASE_DIR, "training", "medqa", "medqa_training_pool.csv"
)
MEDQA_TEST_SIZE = config.MEDQA_TEST_SIZE


def _normalize_text(value):
    return str(value or "").strip()


def _format_query(question, options):
    lines = [f"Question: {_normalize_text(question)}", "Options:"]
    for key in sorted((options or {}).keys()):
        lines.append(f"({key}) {_normalize_text(options[key])}")
    return "\n".join(lines)


def _to_dataframe(split, split_name):
    df = split.to_pandas().copy()
    option_letters = sorted(df["options"].iloc[0].keys()) if len(df) > 0 else ["A", "B", "C", "D"]

    df["query"] = [
        _format_query(question, options)
        for question, options in zip(df["question"], df["options"])
    ]
    df["reference"] = df["answer"].fillna("").astype(str).str.strip()
    df["answer_idx"] = df["answer_idx"].fillna("").astype(str).str.strip()
    df["options_json"] = [
        json.dumps({key: _normalize_text(value) for key, value in sorted((options or {}).items())}, ensure_ascii=False)
        for options in df["options"]
    ]
    for letter in option_letters:
        df[f"option_{letter.lower()}"] = [
            _normalize_text((options or {}).get(letter, ""))
            for options in df["options"]
        ]
    df["meta_info"] = df["meta_info"].fillna("").astype(str).str.strip()
    df["user_type"] = "expert"
    df = df[(df["query"] != "") & (df["reference"] != "") & (df["answer_idx"] != "")].reset_index(drop=True)
    df.insert(0, "id", [f"medqa_{split_name}_{i:05d}" for i in range(len(df))])

    keep_cols = [
        "id",
        "query",
        "reference",
        "answer_idx",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "options_json",
        "meta_info",
        "user_type",
    ]
    return df[keep_cols]


def load_medqa(force=False):
    if not force and os.path.exists(MEDQA_TEST_CSV) and os.path.exists(MEDQA_TRAINING_CSV):
        test_df = pd.read_csv(MEDQA_TEST_CSV)
        train_df = pd.read_csv(MEDQA_TRAINING_CSV)
        if len(test_df) == MEDQA_TEST_SIZE and len(train_df) > 0:
            logger.info("MedQA processed files already exist. Skipping.")
            return

    os.makedirs(MEDQA_DIR, exist_ok=True)
    logger.info("Downloading MedQA from Hugging Face dataset %s", MEDQA_HF_DATASET)
    ds = load_dataset(MEDQA_HF_DATASET, cache_dir=config.HF_HOME)

    train_df = _to_dataframe(ds["train"], "train")
    test_df = _to_dataframe(ds["test"], "test")
    test_df["id"] = [f"medqa_test_{i:05d}" for i in range(len(test_df))]

    os.makedirs(os.path.dirname(MEDQA_TEST_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MEDQA_TRAINING_CSV), exist_ok=True)
    test_df.to_csv(MEDQA_TEST_CSV, index=False)
    train_df.to_csv(MEDQA_TRAINING_CSV, index=False)
    logger.info("Saved %s MedQA test rows -> %s", len(test_df), MEDQA_TEST_CSV)
    logger.info("Saved %s MedQA training rows -> %s", len(train_df), MEDQA_TRAINING_CSV)


def sanity_check():
    if not os.path.exists(MEDQA_TEST_CSV):
        logger.error("SANITY CHECK FAILED: medqa_test_queries.csv not found")
        return
    df = pd.read_csv(MEDQA_TEST_CSV)
    assert len(df) == MEDQA_TEST_SIZE, f"Expected {MEDQA_TEST_SIZE} test rows, found {len(df)}"
    assert "query" in df.columns, "Missing query column"
    assert "reference" in df.columns, "Missing reference column"
    assert "answer_idx" in df.columns, "Missing answer_idx column"
    assert all(df["user_type"] == "expert"), "user_type not all expert"
    logger.info("Loaded %s MedQA test rows. SANITY CHECK PASSED", len(df))


if __name__ == "__main__":
    load_medqa()
    sanity_check()
