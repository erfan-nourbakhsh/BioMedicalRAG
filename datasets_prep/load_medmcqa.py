import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import logging
import sys

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MEDMCQA_HF_DATASET = "openlifescienceai/medmcqa"
MEDMCQA_DIR = os.path.join(config.RAW_DIR, "medmcqa_hf")
MEDMCQA_TEST_CSV = os.path.join(config.PROCESSED_DIR, "medmcqa_test_queries.csv")
MEDMCQA_TRAINING_CSV = os.path.join(
    config.BASE_DIR, "training", "medmcqa", "medmcqa_training_pool.csv"
)
MEDMCQA_TEST_SIZE = config.MEDMCQA_TEST_SIZE


def _normalize_text(value):
    return str(value or "").strip()


def _format_query(question, options):
    lines = [f"Question: {_normalize_text(question)}", "Options:"]
    for key in ["A", "B", "C", "D"]:
        lines.append(f"({key}) {_normalize_text(options.get(key, ''))}")
    return "\n".join(lines)


def _cop_to_label(value):
    if pd.isna(value):
        return ""
    try:
        idx = int(float(str(value).strip()))
    except ValueError:
        text = str(value).strip().lower()
        if text in {"a", "b", "c", "d"}:
            return text.upper()
        return ""
    labels = {0: "A", 1: "B", 2: "C", 3: "D"}
    return labels.get(idx, "")


def _to_dataframe(split, split_name):
    df = split.to_pandas().copy()
    df["choice_type"] = df["choice_type"].fillna("").astype(str).str.strip().str.lower()
    df = df[df["choice_type"] == "single"].copy()

    df["answer_idx"] = df["cop"].apply(_cop_to_label)
    df["option_a"] = df["opa"].fillna("").astype(str).str.strip()
    df["option_b"] = df["opb"].fillna("").astype(str).str.strip()
    df["option_c"] = df["opc"].fillna("").astype(str).str.strip()
    df["option_d"] = df["opd"].fillna("").astype(str).str.strip()
    df["query"] = [
        _format_query(
            question,
            {"A": opa, "B": opb, "C": opc, "D": opd},
        )
        for question, opa, opb, opc, opd in zip(
            df["question"], df["option_a"], df["option_b"], df["option_c"], df["option_d"]
        )
    ]
    df["reference"] = [
        {"A": a, "B": b, "C": c, "D": d}.get(label, "")
        for label, a, b, c, d in zip(
            df["answer_idx"], df["option_a"], df["option_b"], df["option_c"], df["option_d"]
        )
    ]
    df["user_type"] = "expert"
    df["choice_type"] = df["choice_type"].fillna("").astype(str).str.strip()
    df["exp"] = df["exp"].fillna("").astype(str).str.strip()
    df["subject_name"] = df["subject_name"].fillna("").astype(str).str.strip()
    df["topic_name"] = df["topic_name"].fillna("").astype(str).str.strip()
    df = df[(df["query"] != "") & (df["reference"] != "") & (df["answer_idx"] != "")].reset_index(drop=True)
    df["id"] = [f"medmcqa_{split_name}_{i:05d}" for i in range(len(df))]

    keep_cols = [
        "id",
        "query",
        "reference",
        "answer_idx",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "choice_type",
        "exp",
        "subject_name",
        "topic_name",
        "user_type",
    ]
    return df[keep_cols]


def load_medmcqa(force=False, seed=config.SEED):
    if not force and os.path.exists(MEDMCQA_TEST_CSV) and os.path.exists(MEDMCQA_TRAINING_CSV):
        test_df = pd.read_csv(MEDMCQA_TEST_CSV)
        train_df = pd.read_csv(MEDMCQA_TRAINING_CSV)
        if len(test_df) == MEDMCQA_TEST_SIZE and len(train_df) > 0:
            logger.info("MedMCQA processed files already exist. Skipping.")
            return

    os.makedirs(MEDMCQA_DIR, exist_ok=True)
    logger.info("Downloading MedMCQA from Hugging Face dataset %s", MEDMCQA_HF_DATASET)
    ds = load_dataset(MEDMCQA_HF_DATASET, cache_dir=config.HF_HOME)

    train_df = _to_dataframe(ds["train"], "train")

    eval_df_full = _to_dataframe(ds["validation"], "validation")
    test_df = eval_df_full.sample(n=MEDMCQA_TEST_SIZE, random_state=seed).reset_index(drop=True).copy()
    test_df["id"] = [f"medmcqa_test_{i:05d}" for i in range(len(test_df))]

    os.makedirs(os.path.dirname(MEDMCQA_TEST_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MEDMCQA_TRAINING_CSV), exist_ok=True)
    test_df.to_csv(MEDMCQA_TEST_CSV, index=False)
    train_df.to_csv(MEDMCQA_TRAINING_CSV, index=False)
    logger.info(
        "Saved %s MedMCQA eval rows sampled from the labeled validation split -> %s",
        len(test_df),
        MEDMCQA_TEST_CSV,
    )
    logger.info("Saved %s MedMCQA training rows -> %s", len(train_df), MEDMCQA_TRAINING_CSV)


def sanity_check():
    if not os.path.exists(MEDMCQA_TEST_CSV):
        logger.error("SANITY CHECK FAILED: medmcqa_test_queries.csv not found")
        return
    df = pd.read_csv(MEDMCQA_TEST_CSV)
    assert len(df) == MEDMCQA_TEST_SIZE, f"Expected {MEDMCQA_TEST_SIZE} test rows, found {len(df)}"
    assert "query" in df.columns, "Missing query column"
    assert "reference" in df.columns, "Missing reference column"
    assert "answer_idx" in df.columns, "Missing answer_idx column"
    assert all(df["user_type"] == "expert"), "user_type not all expert"
    logger.info("Loaded %s MedMCQA test rows. SANITY CHECK PASSED", len(df))


if __name__ == "__main__":
    load_medmcqa()
    sanity_check()
