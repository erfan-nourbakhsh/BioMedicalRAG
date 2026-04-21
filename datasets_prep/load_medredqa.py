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

MEDREDQA_HF_DATASET = "varun500/medredqa"
MEDREDQA_DIR = os.path.join(config.RAW_DIR, "medredqa_hf")
MEDREDQA_TEST_CSV = os.path.join(config.PROCESSED_DIR, "medredqa_test_queries.csv")
MEDREDQA_TRAINING_CSV = os.path.join(
    config.BASE_DIR, "training", "medredqa", "medredqa_training_pool.csv"
)
MEDREDQA_TEST_SIZE = config.MEDREDQA_TEST_SIZE


def _normalize_text(value):
    return str(value or "").strip()


def _combine_query(title, body):
    title_s = _normalize_text(title)
    body_s = _normalize_text(body)
    return " ".join(part for part in [title_s, body_s] if part).strip()


def _to_dataframe(split, split_name):
    df = split.to_pandas().copy()
    df["query"] = [
        _combine_query(title, body)
        for title, body in zip(df["Title"], df["Body"])
    ]
    df["reference"] = df["Response"].fillna("").astype(str).str.strip()
    df["title"] = df["Title"].fillna("").astype(str).str.strip()
    df["body"] = df["Body"].fillna("").astype(str).str.strip()
    df["occupation"] = df["Occupation"].fillna("").astype(str).str.strip()
    df["response_score"] = df["Response Score"]
    df["pmcids"] = df["PMCID(s)"].fillna("").astype(str).str.strip()
    df["user_type"] = "layperson"
    df = df[(df["query"] != "") & (df["reference"] != "")].reset_index(drop=True)
    df.insert(0, "id", [f"medredqa_{split_name}_{i:05d}" for i in range(len(df))])
    return df[
        [
            "id",
            "query",
            "reference",
            "title",
            "body",
            "occupation",
            "response_score",
            "pmcids",
            "user_type",
        ]
    ]


def load_medredqa(force=False, seed=config.SEED):
    if not force and os.path.exists(MEDREDQA_TEST_CSV) and os.path.exists(MEDREDQA_TRAINING_CSV):
        test_df = pd.read_csv(MEDREDQA_TEST_CSV)
        train_df = pd.read_csv(MEDREDQA_TRAINING_CSV)
        if len(test_df) == MEDREDQA_TEST_SIZE and len(train_df) > 0:
            logger.info("MedRedQA processed files already exist. Skipping.")
            return

    os.makedirs(MEDREDQA_DIR, exist_ok=True)
    logger.info("Downloading MedRedQA from Hugging Face dataset %s", MEDREDQA_HF_DATASET)
    ds = load_dataset(MEDREDQA_HF_DATASET, cache_dir=config.HF_HOME)

    test_df_full = _to_dataframe(ds["test"], "test")
    test_df = (
        test_df_full.sample(n=MEDREDQA_TEST_SIZE, random_state=seed)
        .reset_index(drop=True)
        .copy()
    )
    test_df["id"] = [f"medredqa_test_{i:05d}" for i in range(len(test_df))]
    train_df = pd.concat(
        [_to_dataframe(ds["train"], "train"), _to_dataframe(ds["validation"], "validation")],
        ignore_index=True,
    )

    os.makedirs(os.path.dirname(MEDREDQA_TEST_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MEDREDQA_TRAINING_CSV), exist_ok=True)
    test_df.to_csv(MEDREDQA_TEST_CSV, index=False)
    train_df.to_csv(MEDREDQA_TRAINING_CSV, index=False)
    logger.info(
        "Saved %s MedRedQA test rows sampled from the official test split -> %s",
        len(test_df),
        MEDREDQA_TEST_CSV,
    )
    logger.info("Saved %s MedRedQA training rows -> %s", len(train_df), MEDREDQA_TRAINING_CSV)


def sanity_check():
    if not os.path.exists(MEDREDQA_TEST_CSV):
        logger.error("SANITY CHECK FAILED: medredqa_test_queries.csv not found")
        return
    df = pd.read_csv(MEDREDQA_TEST_CSV)
    assert len(df) == MEDREDQA_TEST_SIZE, f"Expected {MEDREDQA_TEST_SIZE} test rows, found {len(df)}"
    assert "query" in df.columns, "Missing query column"
    assert "reference" in df.columns, "Missing reference column"
    assert all(df["user_type"] == "layperson"), "user_type not all layperson"
    logger.info("Loaded %s MedRedQA test rows. SANITY CHECK PASSED", len(df))


if __name__ == "__main__":
    load_medredqa()
    sanity_check()
