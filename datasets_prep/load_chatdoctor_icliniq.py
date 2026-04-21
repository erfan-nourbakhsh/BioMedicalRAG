import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import json
import logging
import sys

import gdown
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHATDOCTOR_ICLINIQ_FILE_ID = "1ZKbqgYqWc7DJHs3N9TQYQVPdDQmZaClA"
CHATDOCTOR_ICLINIQ_URL = f"https://drive.google.com/uc?id={CHATDOCTOR_ICLINIQ_FILE_ID}"
CHATDOCTOR_ICLINIQ_DIR = os.path.join(config.RAW_DIR, "chatdoctor_icliniq")
CHATDOCTOR_ICLINIQ_JSON = os.path.join(CHATDOCTOR_ICLINIQ_DIR, "chatdoctor_icliniq.json")
CHATDOCTOR_ICLINIQ_TEST_CSV = os.path.join(config.PROCESSED_DIR, "chatdoctor_icliniq_test_queries.csv")
CHATDOCTOR_ICLINIQ_TRAINING_CSV = os.path.join(
    config.BASE_DIR, "training", "chatdoctor_icliniq", "chatdoctor_icliniq_training_pool.csv"
)
CHATDOCTOR_ICLINIQ_TEST_SIZE = config.CHATDOCTOR_ICLINIQ_TEST_SIZE


def _normalize_text(value):
    return str(value or "").strip()


def _ensure_downloaded():
    os.makedirs(CHATDOCTOR_ICLINIQ_DIR, exist_ok=True)
    if not os.path.exists(CHATDOCTOR_ICLINIQ_JSON):
        logger.info("Downloading ChatDoctor-iCliniq JSON from Google Drive")
        gdown.download(CHATDOCTOR_ICLINIQ_URL, CHATDOCTOR_ICLINIQ_JSON, quiet=False)


def load_chatdoctor_icliniq(force=False, seed=config.SEED):
    if not force and os.path.exists(CHATDOCTOR_ICLINIQ_TEST_CSV) and os.path.exists(CHATDOCTOR_ICLINIQ_TRAINING_CSV):
        test_df = pd.read_csv(CHATDOCTOR_ICLINIQ_TEST_CSV)
        train_df = pd.read_csv(CHATDOCTOR_ICLINIQ_TRAINING_CSV)
        if len(test_df) == CHATDOCTOR_ICLINIQ_TEST_SIZE and len(train_df) > 0:
            logger.info("ChatDoctor-iCliniq processed files already exist. Skipping.")
            return

    _ensure_downloaded()
    with open(CHATDOCTOR_ICLINIQ_JSON, encoding="utf-8") as f:
        payload = json.load(f)

    df = pd.DataFrame(payload)
    df["query"] = df["input"].fillna("").astype(str).str.strip()
    df["reference"] = df["answer_icliniq"].fillna("").astype(str).str.strip()
    df["answer_chatgpt"] = df.get("answer_chatgpt", "").fillna("").astype(str).str.strip()
    df["answer_chatdoctor"] = df.get("answer_chatdoctor", "").fillna("").astype(str).str.strip()
    df["user_type"] = "layperson"
    df = df[(df["query"] != "") & (df["reference"] != "")].reset_index(drop=True)

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = shuffled.iloc[:CHATDOCTOR_ICLINIQ_TEST_SIZE].copy().reset_index(drop=True)
    train_df = shuffled.iloc[CHATDOCTOR_ICLINIQ_TEST_SIZE:].copy().reset_index(drop=True)

    test_df.insert(0, "id", [f"chatdoctor_icliniq_test_{i:05d}" for i in range(len(test_df))])
    train_df.insert(0, "id", [f"chatdoctor_icliniq_train_{i:05d}" for i in range(len(train_df))])

    keep_cols = [
        "id",
        "query",
        "reference",
        "answer_chatgpt",
        "answer_chatdoctor",
        "user_type",
    ]
    test_df = test_df[keep_cols]
    train_df = train_df[keep_cols]

    os.makedirs(os.path.dirname(CHATDOCTOR_ICLINIQ_TEST_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(CHATDOCTOR_ICLINIQ_TRAINING_CSV), exist_ok=True)
    test_df.to_csv(CHATDOCTOR_ICLINIQ_TEST_CSV, index=False)
    train_df.to_csv(CHATDOCTOR_ICLINIQ_TRAINING_CSV, index=False)
    logger.info("Saved %s ChatDoctor-iCliniq test rows -> %s", len(test_df), CHATDOCTOR_ICLINIQ_TEST_CSV)
    logger.info("Saved %s ChatDoctor-iCliniq training rows -> %s", len(train_df), CHATDOCTOR_ICLINIQ_TRAINING_CSV)


def sanity_check():
    if not os.path.exists(CHATDOCTOR_ICLINIQ_TEST_CSV):
        logger.error("SANITY CHECK FAILED: chatdoctor_icliniq_test_queries.csv not found")
        return
    df = pd.read_csv(CHATDOCTOR_ICLINIQ_TEST_CSV)
    assert len(df) == CHATDOCTOR_ICLINIQ_TEST_SIZE, f"Expected {CHATDOCTOR_ICLINIQ_TEST_SIZE} test rows, found {len(df)}"
    assert "query" in df.columns, "Missing query column"
    assert "reference" in df.columns, "Missing reference column"
    assert all(df["user_type"] == "layperson"), "user_type not all layperson"
    logger.info("Loaded %s ChatDoctor-iCliniq test rows. SANITY CHECK PASSED", len(df))


if __name__ == "__main__":
    load_chatdoctor_icliniq()
    sanity_check()
