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

MMLU_HF_DATASET = "cais/mmlu"
MMLU_MEDICAL_SUBSETS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]
MMLU_MEDICAL_DIR = os.path.join(config.RAW_DIR, "mmlu_medical_hf")
MMLU_MEDICAL_TEST_CSV = os.path.join(config.PROCESSED_DIR, "mmlu_medical_test_queries.csv")
MMLU_MEDICAL_TRAINING_CSV = os.path.join(
    config.BASE_DIR, "training", "mmlu_medical", "mmlu_medical_training_pool.csv"
)
MMLU_MEDICAL_PER_SUBSET_TEST_SIZE = 100


def _normalize_text(value):
    return str(value or "").strip()


def _format_query(question, choices):
    labels = ["A", "B", "C", "D"]
    lines = [f"Question: {_normalize_text(question)}", "Options:"]
    for label, choice in zip(labels, choices):
        lines.append(f"({label}) {_normalize_text(choice)}")
    return "\n".join(lines)


def _answer_to_label(value):
    if pd.isna(value):
        return ""
    try:
        idx = int(value)
    except Exception:
        text = str(value).strip().upper()
        return text if text in {"A", "B", "C", "D"} else ""
    return {0: "A", 1: "B", 2: "C", 3: "D"}.get(idx, "")


def _subset_dataframe(subset_name):
    ds = load_dataset(MMLU_HF_DATASET, subset_name, cache_dir=config.HF_HOME)
    frames = []
    for split_name in ["test", "validation", "dev"]:
        split_df = ds[split_name].to_pandas().copy()
        split_df["source_split"] = split_name
        frames.append(split_df)
    df = pd.concat(frames, ignore_index=True)
    df["subject"] = df["subject"].fillna("").astype(str).str.strip()
    df["answer_idx"] = df["answer"].apply(_answer_to_label)
    df["option_a"] = df["choices"].apply(lambda xs: _normalize_text(xs[0]) if len(xs) > 0 else "")
    df["option_b"] = df["choices"].apply(lambda xs: _normalize_text(xs[1]) if len(xs) > 1 else "")
    df["option_c"] = df["choices"].apply(lambda xs: _normalize_text(xs[2]) if len(xs) > 2 else "")
    df["option_d"] = df["choices"].apply(lambda xs: _normalize_text(xs[3]) if len(xs) > 3 else "")
    df["query"] = [
        _format_query(question, choices)
        for question, choices in zip(df["question"], df["choices"])
    ]
    df["reference"] = [
        {"A": a, "B": b, "C": c, "D": d}.get(label, "")
        for label, a, b, c, d in zip(
            df["answer_idx"], df["option_a"], df["option_b"], df["option_c"], df["option_d"]
        )
    ]
    df["user_type"] = "expert"
    df["subset_name"] = subset_name
    df = df[(df["query"] != "") & (df["reference"] != "") & (df["answer_idx"] != "")].reset_index(drop=True)
    return df[
        [
            "subset_name",
            "source_split",
            "subject",
            "query",
            "reference",
            "answer_idx",
            "option_a",
            "option_b",
            "option_c",
            "option_d",
            "user_type",
        ]
    ]


def load_mmlu_medical(force=False, seed=config.SEED):
    if not force and os.path.exists(MMLU_MEDICAL_TEST_CSV) and os.path.exists(MMLU_MEDICAL_TRAINING_CSV):
        test_df = pd.read_csv(MMLU_MEDICAL_TEST_CSV)
        train_df = pd.read_csv(MMLU_MEDICAL_TRAINING_CSV)
        if len(test_df) == config.MMLU_MEDICAL_TEST_SIZE and len(train_df) > 0:
            logger.info("MMLU medical processed files already exist. Skipping.")
            return

    os.makedirs(MMLU_MEDICAL_DIR, exist_ok=True)
    test_parts = []
    train_parts = []
    for idx, subset in enumerate(MMLU_MEDICAL_SUBSETS):
        subset_seed = seed + idx
        df = _subset_dataframe(subset)
        if len(df) < MMLU_MEDICAL_PER_SUBSET_TEST_SIZE:
            raise ValueError(
                f"Subset {subset} has only {len(df)} rows, cannot sample "
                f"{MMLU_MEDICAL_PER_SUBSET_TEST_SIZE} test examples."
            )
        sampled_test = df.sample(n=MMLU_MEDICAL_PER_SUBSET_TEST_SIZE, random_state=subset_seed)
        sampled_train = df.drop(sampled_test.index).reset_index(drop=True)
        sampled_test = sampled_test.reset_index(drop=True)
        test_parts.append(sampled_test)
        train_parts.append(sampled_train)

    test_df = pd.concat(test_parts, ignore_index=True)
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df.insert(0, "id", [f"mmlu_medical_test_{i:05d}" for i in range(len(test_df))])
    train_df.insert(0, "id", [f"mmlu_medical_train_{i:05d}" for i in range(len(train_df))])

    os.makedirs(os.path.dirname(MMLU_MEDICAL_TEST_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MMLU_MEDICAL_TRAINING_CSV), exist_ok=True)
    test_df.to_csv(MMLU_MEDICAL_TEST_CSV, index=False)
    train_df.to_csv(MMLU_MEDICAL_TRAINING_CSV, index=False)
    logger.info("Saved %s MMLU medical test rows -> %s", len(test_df), MMLU_MEDICAL_TEST_CSV)
    logger.info("Saved %s MMLU medical training rows -> %s", len(train_df), MMLU_MEDICAL_TRAINING_CSV)


def sanity_check():
    if not os.path.exists(MMLU_MEDICAL_TEST_CSV):
        logger.error("SANITY CHECK FAILED: mmlu_medical_test_queries.csv not found")
        return
    df = pd.read_csv(MMLU_MEDICAL_TEST_CSV)
    assert len(df) == config.MMLU_MEDICAL_TEST_SIZE, f"Expected {config.MMLU_MEDICAL_TEST_SIZE} test rows, found {len(df)}"
    assert "query" in df.columns, "Missing query column"
    assert "reference" in df.columns, "Missing reference column"
    assert "answer_idx" in df.columns, "Missing answer_idx column"
    assert all(df["user_type"] == "expert"), "user_type not all expert"
    per_subset = df["subset_name"].value_counts().to_dict()
    logger.info("Loaded %s MMLU medical test rows. Per subset: %s. SANITY CHECK PASSED", len(df), per_subset)


if __name__ == "__main__":
    load_mmlu_medical()
    sanity_check()
