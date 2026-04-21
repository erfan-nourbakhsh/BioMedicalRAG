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

MEDICATIONQA_HF_DATASET = "truehealth/medicationqa"
MEDICATIONQA_DIR = os.path.join(config.RAW_DIR, "medicationqa_hf")
MEDICATIONQA_TEST_CSV = os.path.join(config.PROCESSED_DIR, "medicationqa_test_queries.csv")
MEDICATIONQA_TRAINING_CSV = os.path.join(
    config.BASE_DIR, "training", "medicationqa", "medicationqa_training_pool.csv"
)
MEDICATIONQA_TEST_SIZE = config.MEDICATIONQA_TEST_SIZE


def load_medicationqa(force=False, seed=config.SEED):
    if not force and os.path.exists(MEDICATIONQA_TEST_CSV) and os.path.exists(MEDICATIONQA_TRAINING_CSV):
        test_df = pd.read_csv(MEDICATIONQA_TEST_CSV)
        train_df = pd.read_csv(MEDICATIONQA_TRAINING_CSV)
        if len(test_df) == MEDICATIONQA_TEST_SIZE and len(train_df) > 0:
            logger.info("MedicationQA processed files already exist. Skipping.")
            return

    os.makedirs(MEDICATIONQA_DIR, exist_ok=True)
    logger.info("Downloading MedicationQA from Hugging Face dataset %s", MEDICATIONQA_HF_DATASET)
    ds = load_dataset(MEDICATIONQA_HF_DATASET, cache_dir=config.HF_HOME)
    df = ds["train"].to_pandas().copy()

    df["query"] = df["Question"].fillna("").astype(str).str.strip()
    df["reference"] = df["Answer"].fillna("").astype(str).str.strip()
    df["focus_drug"] = df["Focus (Drug)"].fillna("").astype(str).str.strip()
    df["question_type"] = df["Question Type"].fillna("").astype(str).str.strip()
    df["section_title"] = df["Section Title"].fillna("").astype(str).str.strip()
    df["url"] = df["URL"].fillna("").astype(str).str.strip()
    df["user_type"] = "layperson"
    df = df[(df["query"] != "") & (df["reference"] != "")].reset_index(drop=True)

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = shuffled.iloc[:MEDICATIONQA_TEST_SIZE].copy().reset_index(drop=True)
    train_df = shuffled.iloc[MEDICATIONQA_TEST_SIZE:].copy().reset_index(drop=True)

    test_df.insert(0, "id", [f"medicationqa_{i:05d}" for i in range(len(test_df))])
    train_df.insert(0, "id", [f"medicationqa_train_{i:05d}" for i in range(len(train_df))])

    keep_cols = [
        "id",
        "query",
        "reference",
        "focus_drug",
        "question_type",
        "section_title",
        "url",
        "user_type",
    ]
    test_df = test_df[keep_cols]
    train_df = train_df[keep_cols]

    os.makedirs(os.path.dirname(MEDICATIONQA_TEST_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MEDICATIONQA_TRAINING_CSV), exist_ok=True)
    test_df.to_csv(MEDICATIONQA_TEST_CSV, index=False)
    train_df.to_csv(MEDICATIONQA_TRAINING_CSV, index=False)
    logger.info("Saved %s MedicationQA test rows -> %s", len(test_df), MEDICATIONQA_TEST_CSV)
    logger.info("Saved %s MedicationQA training rows -> %s", len(train_df), MEDICATIONQA_TRAINING_CSV)


def sanity_check():
    if not os.path.exists(MEDICATIONQA_TEST_CSV):
        logger.error("SANITY CHECK FAILED: medicationqa_test_queries.csv not found")
        return
    df = pd.read_csv(MEDICATIONQA_TEST_CSV)
    assert len(df) == MEDICATIONQA_TEST_SIZE, f"Expected {MEDICATIONQA_TEST_SIZE} test rows, found {len(df)}"
    assert "query" in df.columns, "Missing query column"
    assert "reference" in df.columns, "Missing reference column"
    assert all(df["user_type"] == "layperson"), "user_type not all layperson"
    logger.info("Loaded %s MedicationQA test rows. SANITY CHECK PASSED", len(df))


if __name__ == "__main__":
    load_medicationqa()
    sanity_check()
