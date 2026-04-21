import gzip
import json
import logging
import os
import sys
from glob import glob
import zipfile

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BIOASQ_TASKB_DIR = os.path.join(config.RAW_DIR, "bioasq_taskb")
BIOASQ_TASKB_TRAIN_URL = "https://participants-area.bioasq.org/Tasks/13b/trainingDataset/"
BIOASQ_TASKB_GOLD_URL = "https://participants-area.bioasq.org/Tasks/13b/goldenDataset/"
BIOASQ_TASKB_TEST_CSV = os.path.join(config.PROCESSED_DIR, "bioasq_taskb_test_queries.csv")
BIOASQ_TASKB_TRAINING_CSV = os.path.join(
    config.BASE_DIR, "training", "bioasq_taskb", "bioasq_taskb_training_pool.csv"
)
LOCAL_TRAIN_ZIP = os.path.join(config.BASE_DIR, "BioASQ-training13b.zip")
LOCAL_GOLD_ZIP = os.path.join(config.BASE_DIR, "Task13BGoldenEnriched.zip")


def _raw_candidates(kind):
    patterns = {
        "train": [
            "*13b*train*.json", "*13b*train*.json.gz",
            "*train*13b*.json", "*train*13b*.json.gz",
            "*training*.json", "*training*.json.gz",
        ],
        "gold": [
            "*13b*gold*.json", "*13b*gold*.json.gz",
            "*gold*13b*.json", "*gold*13b*.json.gz",
            "*golden*.json", "*golden*.json.gz",
        ],
    }
    matches = []
    for pattern in patterns[kind]:
        matches.extend(glob(os.path.join(BIOASQ_TASKB_DIR, "**", pattern), recursive=True))
    return sorted(set(matches))


def _extract_local_zip(zip_path):
    if not os.path.exists(zip_path):
        return
    os.makedirs(BIOASQ_TASKB_DIR, exist_ok=True)
    logger.info("Extracting %s into %s", zip_path, BIOASQ_TASKB_DIR)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(BIOASQ_TASKB_DIR)


def _download_if_possible(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    logger.info("Attempting download from %s", url)
    response = requests.get(url, timeout=60)
    content_type = response.headers.get("content-type", "")
    text_head = response.text[:2000] if "text" in content_type else ""

    if "text/html" in content_type or "BioASQ Participants Area" in text_head:
        raise RuntimeError(
            "BioASQ Task 13b download requires participant-area login. "
            f"Please download the dataset manually from {url} and place the JSON file in {BIOASQ_TASKB_DIR}."
        )

    with open(target_path, "wb") as f:
        f.write(response.content)
    logger.info("Saved downloaded dataset to %s", target_path)


def _ensure_raw_file(kind):
    zip_path = LOCAL_TRAIN_ZIP if kind == "train" else LOCAL_GOLD_ZIP
    if os.path.exists(zip_path):
        _extract_local_zip(zip_path)

    candidates = _raw_candidates(kind)
    if candidates:
        logger.info("Using existing BioASQ Task B %s file(s): %s", kind, candidates)
        return candidates

    default_name = "trainingDataset.json" if kind == "train" else "goldenDataset.json"
    url = BIOASQ_TASKB_TRAIN_URL if kind == "train" else BIOASQ_TASKB_GOLD_URL
    target_path = os.path.join(BIOASQ_TASKB_DIR, default_name)
    _download_if_possible(url, target_path)
    return [target_path]


def _load_json(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _load_questions(paths):
    all_questions = []
    for path in paths:
        payload = _load_json(path)
        questions = payload.get("questions", payload if isinstance(payload, list) else [])
        all_questions.extend(questions)
    return {"questions": all_questions}


def _normalize_ideal_answers(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_summary_rows(payload, split_name):
    questions = payload.get("questions", payload if isinstance(payload, list) else [])
    rows = []
    for idx, question in enumerate(questions):
        if question.get("type") != "summary":
            continue
        body = str(question.get("body", "")).strip()
        if not body:
            continue
        ideal_answers = _normalize_ideal_answers(question.get("ideal_answer"))
        rows.append({
            "id": question.get("id") or f"bioasq13b_{split_name}_{idx:05d}",
            "query": body,
            "reference": json.dumps(ideal_answers, ensure_ascii=False) if split_name == "test" else (ideal_answers[0] if ideal_answers else ""),
            "reference_list": json.dumps(ideal_answers, ensure_ascii=False),
            "question_type": question.get("type", ""),
            "user_type": "expert",
        })
    return pd.DataFrame(rows)


def save_and_report(test_df, train_df):
    os.makedirs(os.path.dirname(BIOASQ_TASKB_TEST_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(BIOASQ_TASKB_TRAINING_CSV), exist_ok=True)
    test_df.to_csv(BIOASQ_TASKB_TEST_CSV, index=False)
    train_df.to_csv(BIOASQ_TASKB_TRAINING_CSV, index=False)
    logger.info("Saved %s BioASQ Task B test rows -> %s", len(test_df), BIOASQ_TASKB_TEST_CSV)
    logger.info("Saved %s BioASQ Task B training rows -> %s", len(train_df), BIOASQ_TASKB_TRAINING_CSV)


def load_bioasq_taskb(force=False):
    if not force and os.path.exists(BIOASQ_TASKB_TEST_CSV) and os.path.exists(BIOASQ_TASKB_TRAINING_CSV):
        test_df = pd.read_csv(BIOASQ_TASKB_TEST_CSV)
        train_df = pd.read_csv(BIOASQ_TASKB_TRAINING_CSV)
        if len(test_df) > 0 and len(train_df) > 0:
            logger.info("BioASQ Task B processed files already exist. Skipping.")
            return

    train_paths = _ensure_raw_file("train")
    gold_paths = _ensure_raw_file("gold")
    train_payload = _load_questions(train_paths)
    gold_payload = _load_questions(gold_paths)

    train_df = _extract_summary_rows(train_payload, "train")
    test_df = _extract_summary_rows(gold_payload, "test")

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError("No summary questions extracted from BioASQ Task 13b.")

    save_and_report(test_df, train_df)


def sanity_check():
    if not os.path.exists(BIOASQ_TASKB_TEST_CSV):
        logger.error("SANITY CHECK FAILED: bioasq_taskb_test_queries.csv not found")
        return
    df = pd.read_csv(BIOASQ_TASKB_TEST_CSV)
    assert len(df) > 0, "Empty BioASQ Task B test dataframe"
    assert "query" in df.columns, "Missing query column"
    assert "reference" in df.columns, "Missing reference column"
    assert "reference_list" in df.columns, "Missing reference_list column"
    assert all(df["user_type"] == "expert"), "user_type not all expert"
    logger.info("Loaded %s BioASQ Task B test rows. SANITY CHECK PASSED", len(df))


if __name__ == "__main__":
    load_bioasq_taskb()
    sanity_check()
