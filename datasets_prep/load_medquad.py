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

MEDQUAD_HF_DATASET = "lavita/MedQuAD"
MEDQUAD_DIR = os.path.join(config.RAW_DIR, "medquad_hf")
MEDQUAD_TEST_CSV = os.path.join(config.PROCESSED_DIR, "medquad_test_queries.csv")
MEDQUAD_TRAINING_CSV = os.path.join(
    config.BASE_DIR, "training", "medquad", "medquad_training_pool.csv"
)
MEDQUAD_TEST_SIZE = 1000


def load_medquad(force=False, seed=config.SEED):
    if not force and os.path.exists(MEDQUAD_TEST_CSV) and os.path.exists(MEDQUAD_TRAINING_CSV):
        test_df = pd.read_csv(MEDQUAD_TEST_CSV)
        train_df = pd.read_csv(MEDQUAD_TRAINING_CSV)
        if len(test_df) == MEDQUAD_TEST_SIZE and len(train_df) > 0:
            logger.info("MedQuAD processed files already exist. Skipping.")
            return

    os.makedirs(MEDQUAD_DIR, exist_ok=True)
    logger.info("Downloading MedQuAD from Hugging Face dataset %s", MEDQUAD_HF_DATASET)
    ds = load_dataset(MEDQUAD_HF_DATASET, cache_dir=config.HF_HOME)
    df = ds["train"].to_pandas().copy()

    df["query"] = df["question"].fillna("").astype(str).str.strip()
    df["reference"] = df["answer"].fillna("").astype(str).str.strip()
    df["question_focus"] = df["question_focus"].fillna("").astype(str).str.strip()
    df["question_type"] = df["question_type"].fillna("").astype(str).str.strip()
    df["document_source"] = df["document_source"].fillna("").astype(str).str.strip()
    df["document_url"] = df["document_url"].fillna("").astype(str).str.strip()
    df["user_type"] = "expert"
    df = df[(df["query"] != "") & (df["reference"] != "")].reset_index(drop=True)

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = shuffled.iloc[:MEDQUAD_TEST_SIZE].copy().reset_index(drop=True)
    train_df = shuffled.iloc[MEDQUAD_TEST_SIZE:].copy().reset_index(drop=True)

    test_df.insert(0, "id", [f"medquad_{i:05d}" for i in range(len(test_df))])
    train_df.insert(0, "id", [f"medquad_train_{i:05d}" for i in range(len(train_df))])

    keep_cols = [
        "id",
        "query",
        "reference",
        "question_focus",
        "question_type",
        "document_source",
        "document_url",
        "user_type",
    ]
    test_df = test_df[keep_cols]
    train_df = train_df[keep_cols]

    os.makedirs(os.path.dirname(MEDQUAD_TEST_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MEDQUAD_TRAINING_CSV), exist_ok=True)
    test_df.to_csv(MEDQUAD_TEST_CSV, index=False)
    train_df.to_csv(MEDQUAD_TRAINING_CSV, index=False)
    logger.info("Saved %s MedQuAD test rows -> %s", len(test_df), MEDQUAD_TEST_CSV)
    logger.info("Saved %s MedQuAD training rows -> %s", len(train_df), MEDQUAD_TRAINING_CSV)


def sanity_check():
    if not os.path.exists(MEDQUAD_TEST_CSV):
        logger.error("SANITY CHECK FAILED: medquad_test_queries.csv not found")
        return
    df = pd.read_csv(MEDQUAD_TEST_CSV)
    assert len(df) == MEDQUAD_TEST_SIZE, f"Expected {MEDQUAD_TEST_SIZE} test rows, found {len(df)}"
    assert "query" in df.columns, "Missing query column"
    assert "reference" in df.columns, "Missing reference column"
    assert all(df["user_type"] == "expert"), "user_type not all expert"
    logger.info("Loaded %s MedQuAD test rows. SANITY CHECK PASSED", len(df))


if __name__ == "__main__":
    load_medquad()
    sanity_check()
