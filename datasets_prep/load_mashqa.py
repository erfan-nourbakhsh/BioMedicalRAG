import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import json
import logging
import sys
import zipfile

import gdown
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MASHQA_FILE_ID = "1RY_gWB4gaUPkW3w9WhIZAwxg5dzNFliK"
MASHQA_DOWNLOAD_URL = f"https://drive.google.com/uc?id={MASHQA_FILE_ID}"
MASHQA_DIR = os.path.join(config.RAW_DIR, "mashqa")
MASHQA_ZIP_PATH = os.path.join(MASHQA_DIR, "mashqa_data.zip")
MASHQA_EXTRACT_DIR = os.path.join(MASHQA_DIR, "mashqa_data")
MASHQA_TRAIN_JSON = os.path.join(MASHQA_EXTRACT_DIR, "train_webmd_squad_v2_consec.json")
MASHQA_TEST_JSON = os.path.join(MASHQA_EXTRACT_DIR, "test_webmd_squad_v2_consec.json")
MASHQA_TEST_CSV = os.path.join(config.PROCESSED_DIR, "mashqa_test_queries.csv")
MASHQA_TRAINING_CSV = os.path.join(
    config.BASE_DIR, "training", "mashqa", "mashqa_training_pool.csv"
)
MASHQA_TEST_SIZE = config.MASHQA_TEST_SIZE


def _normalize_text(value):
    return str(value or "").strip()


def _ensure_downloaded():
    os.makedirs(MASHQA_DIR, exist_ok=True)
    if not os.path.exists(MASHQA_ZIP_PATH):
        logger.info("Downloading MASH-QA archive from Google Drive")
        gdown.download(MASHQA_DOWNLOAD_URL, MASHQA_ZIP_PATH, quiet=False, fuzzy=True)

    if os.path.exists(MASHQA_TRAIN_JSON) and os.path.exists(MASHQA_TEST_JSON):
        return

    logger.info("Extracting MASH-QA archive -> %s", MASHQA_DIR)
    with zipfile.ZipFile(MASHQA_ZIP_PATH) as zf:
        zf.extractall(MASHQA_DIR)


def _flatten_split(json_path, split_name):
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for article in payload.get("data", []):
        title = _normalize_text(article.get("title", ""))
        for paragraph in article.get("paragraphs", []):
            context = _normalize_text(paragraph.get("context", ""))
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers") or []
                first_answer = answers[0] if answers else {}
                reference = _normalize_text(first_answer.get("text", ""))
                query = _normalize_text(qa.get("question", ""))
                if not query or not reference:
                    continue
                rows.append(
                    {
                        "id": _normalize_text(qa.get("id", "")),
                        "query": query,
                        "reference": reference,
                        "title": title,
                        "url": _normalize_text(qa.get("url", "")),
                        "context": context,
                        "answer_start": first_answer.get("answer_start", ""),
                        "is_impossible": bool(qa.get("is_impossible", False)),
                        "user_type": "layperson",
                    }
                )

    df = pd.DataFrame(rows)
    df.insert(0, "source_split", split_name)
    return df


def load_mashqa(force=False, seed=config.SEED):
    if not force and os.path.exists(MASHQA_TEST_CSV) and os.path.exists(MASHQA_TRAINING_CSV):
        test_df = pd.read_csv(MASHQA_TEST_CSV)
        train_df = pd.read_csv(MASHQA_TRAINING_CSV)
        if len(test_df) == MASHQA_TEST_SIZE and len(train_df) > 0:
            logger.info("MASH-QA processed files already exist. Skipping.")
            return

    _ensure_downloaded()

    train_df = _flatten_split(MASHQA_TRAIN_JSON, "train")
    test_df_full = _flatten_split(MASHQA_TEST_JSON, "test")
    test_df = test_df_full.sample(n=MASHQA_TEST_SIZE, random_state=seed).reset_index(drop=True).copy()

    train_df["id"] = [f"mashqa_train_{i:05d}" for i in range(len(train_df))]
    test_df["id"] = [f"mashqa_test_{i:05d}" for i in range(len(test_df))]

    os.makedirs(os.path.dirname(MASHQA_TEST_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MASHQA_TRAINING_CSV), exist_ok=True)
    test_df.to_csv(MASHQA_TEST_CSV, index=False)
    train_df.to_csv(MASHQA_TRAINING_CSV, index=False)
    logger.info(
        "Saved %s MASH-QA test rows sampled from the official test split -> %s",
        len(test_df),
        MASHQA_TEST_CSV,
    )
    logger.info("Saved %s MASH-QA training rows -> %s", len(train_df), MASHQA_TRAINING_CSV)


def sanity_check():
    if not os.path.exists(MASHQA_TEST_CSV):
        logger.error("SANITY CHECK FAILED: mashqa_test_queries.csv not found")
        return
    df = pd.read_csv(MASHQA_TEST_CSV)
    assert len(df) == MASHQA_TEST_SIZE, f"Expected {MASHQA_TEST_SIZE} test rows, found {len(df)}"
    assert "query" in df.columns, "Missing query column"
    assert "reference" in df.columns, "Missing reference column"
    assert all(df["user_type"] == "layperson"), "user_type not all layperson"
    logger.info("Loaded %s MASH-QA test rows. SANITY CHECK PASSED", len(df))


if __name__ == "__main__":
    load_mashqa()
    sanity_check()
