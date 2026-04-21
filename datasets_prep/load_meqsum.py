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

MEQSUM_HF_DATASET = "albertvillanova/meqsum"
MEQSUM_DIR = os.path.join(config.RAW_DIR, "meqsum_hf")
MEQSUM_PARQUET = os.path.join(MEQSUM_DIR, "train.parquet")
MEQSUM_JSONL = os.path.join(MEQSUM_DIR, "train.jsonl")
MEQSUM_TEST_CSV = os.path.join(config.PROCESSED_DIR, "meqsum_test_queries.csv")
MEQSUM_TRAINING_CSV = os.path.join(config.BASE_DIR, "training", "meqsum", "meqsum_training_pool.csv")


def download_meqsum():
    if os.path.exists(MEQSUM_PARQUET) and os.path.exists(MEQSUM_JSONL):
        logger.info("MeQSum HF files already exist at %s", MEQSUM_DIR)
        return

    os.makedirs(MEQSUM_DIR, exist_ok=True)
    logger.info("Downloading MeQSum from Hugging Face dataset %s", MEQSUM_HF_DATASET)
    ds = load_dataset(MEQSUM_HF_DATASET)
    train = ds["train"]
    train.to_parquet(MEQSUM_PARQUET)
    train.to_json(MEQSUM_JSONL)
    logger.info("Saved %s rows to %s", len(train), MEQSUM_DIR)


def load_full_meqsum():
    download_meqsum()
    df = pd.read_parquet(MEQSUM_PARQUET)
    df = df.rename(columns={"File": "source_file", "CHQ": "query", "Summary": "reference"})
    df["query"] = df["query"].fillna("").astype(str).str.strip()
    df["reference"] = df["reference"].fillna("").astype(str).str.strip()
    df = df[(df["query"] != "") & (df["reference"] != "")].reset_index(drop=True)
    logger.info("Loaded %s valid MeQSum rows", len(df))
    return df


def build_splits(seed=config.SEED):
    df = load_full_meqsum()
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = shuffled.iloc[:500].copy().reset_index(drop=True)
    train_df = shuffled.iloc[500:1000].copy().reset_index(drop=True)

    test_df.insert(0, "id", [f"meqsum_{i:04d}" for i in range(len(test_df))])
    train_df.insert(0, "id", [f"meqsum_train_{i:04d}" for i in range(len(train_df))])

    for part in (test_df, train_df):
        part["user_type"] = "layperson"

    test_df = test_df[["id", "query", "reference", "source_file", "user_type"]]
    train_df = train_df[["id", "query", "reference", "source_file", "user_type"]]
    return test_df, train_df


def save_and_report(test_df, train_df):
    os.makedirs(os.path.dirname(MEQSUM_TEST_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MEQSUM_TRAINING_CSV), exist_ok=True)
    test_df.to_csv(MEQSUM_TEST_CSV, index=False)
    train_df.to_csv(MEQSUM_TRAINING_CSV, index=False)
    logger.info("Saved %s test rows -> %s", len(test_df), MEQSUM_TEST_CSV)
    logger.info("Saved %s training rows -> %s", len(train_df), MEQSUM_TRAINING_CSV)


def load_meqsum(force=False):
    if not force and os.path.exists(MEQSUM_TEST_CSV) and os.path.exists(MEQSUM_TRAINING_CSV):
        test_df = pd.read_csv(MEQSUM_TEST_CSV)
        train_df = pd.read_csv(MEQSUM_TRAINING_CSV)
        if len(test_df) == 500 and len(train_df) == 500:
            logger.info("MeQSum splits already exist (500/500). Skipping.")
            return

    test_df, train_df = build_splits(seed=config.SEED)
    save_and_report(test_df, train_df)


def sanity_check():
    if not os.path.exists(MEQSUM_TEST_CSV):
        logger.error("SANITY CHECK FAILED: meqsum_test_queries.csv not found")
        return
    df = pd.read_csv(MEQSUM_TEST_CSV)
    assert len(df) == 500, f"Expected 500 test rows, found {len(df)}"
    assert "query" in df.columns, "Missing query column"
    assert "reference" in df.columns, "Missing reference column"
    assert all(df["user_type"] == "layperson"), "user_type not all layperson"
    logger.info("Loaded %s MeQSum test rows. SANITY CHECK PASSED", len(df))


if __name__ == "__main__":
    load_meqsum()
    sanity_check()
