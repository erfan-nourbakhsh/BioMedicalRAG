import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import csv
import json
import logging
import random
import re
import sys
import time
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from bs4.exceptions import ParserRejectedMarkup
from tqdm import tqdm

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import config_extended as ext_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

YAHOO_TRAIN_CSV = os.path.join(config.RAW_DIR, "yahoo", "train.csv")


def clean_html(text):
    if not text:
        return ""
    try:
        soup = BeautifulSoup(text, "html.parser")
        cleaned = soup.get_text(separator=" ")
    except ParserRejectedMarkup:
        cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_all_yahoo_posts():
    """Load the full Yahoo Answers train.csv across all categories."""
    if not os.path.exists(YAHOO_TRAIN_CSV):
        raise FileNotFoundError(
            f"{YAHOO_TRAIN_CSV} not found. Download from Kaggle first:\n"
            f"  kaggle datasets download -d jarupula/yahoo-answers-dataset "
            f"-p {os.path.join(config.RAW_DIR, 'yahoo')} --unzip"
        )

    logger.info(f"Loading Yahoo Answers from {YAHOO_TRAIN_CSV}")
    logger.info("Source: https://www.kaggle.com/datasets/jarupula/yahoo-answers-dataset")

    total_rows = 0
    skipped_malformed = 0
    skipped_quality = 0
    docs = []
    min_answer_words = ext_config.YAHOO_MIN_ANSWER_WORDS
    min_total_words = ext_config.YAHOO_MIN_TOTAL_WORDS

    with open(YAHOO_TRAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in tqdm(reader, total=1400000, desc="Reading Yahoo train.csv"):
            total_rows += 1
            if len(row) < 4:
                skipped_malformed += 1
                continue

            class_index = row[0]
            q_title = clean_html(row[1])
            q_content = clean_html(row[2])
            best_answer = clean_html(row[3])

            answer_wc = len(best_answer.split()) if best_answer else 0
            if answer_wc < min_answer_words:
                skipped_quality += 1
                continue

            question_text = f"{q_title} {q_content}".strip()
            doc_text = f"{q_title} {q_content} {best_answer}".strip()
            if len(doc_text.split()) < min_total_words:
                skipped_quality += 1
                continue

            wc = len(doc_text.split())
            docs.append({
                "chunk_id": f"yahoo_{len(docs) + 1}",
                "source_id": f"yahoo_{len(docs) + 1}",
                "class_index": class_index,
                "question_title": q_title,
                "question_text": question_text,
                "answer_text": best_answer,
                "text": doc_text,
                "word_count": wc,
            })

    logger.info(f"Total rows in train.csv: {total_rows}")
    logger.info(f"Skipped malformed rows: {skipped_malformed}")
    logger.info(f"Skipped (quality filter): {skipped_quality}")
    logger.info(f"Documents kept: {len(docs)}")
    return docs, total_rows


def save_results(docs):
    proc_path = os.path.join(config.PROCESSED_DIR, "yahoo_corpus.jsonl")
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    with open(proc_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    logger.info(f"Saved {len(docs)} documents to {proc_path}")


def write_statistics(docs, total_rows, elapsed):
    import numpy as np
    wc = [d["word_count"] for d in docs]
    wc_arr = np.array(wc)

    stats_path = os.path.join(config.BASE_DIR, "statistics.txt")
    mode = "a" if os.path.exists(stats_path) else "w"
    with open(stats_path, mode) as f:
        f.write("=" * 60 + "\n")
        f.write("YAHOO ANSWERS CORPUS STATISTICS\n")
        f.write(f"Source: https://www.kaggle.com/datasets/jarupula/yahoo-answers-dataset\n")
        f.write(f"Date built: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total rows in train.csv:    {total_rows}\n")
        f.write(f"Documents in corpus:        {len(docs)}\n")
        f.write(f"Chunking:                    None (whole documents)\n")
        f.write(f"Filtering:                   All categories kept; minimum answer and total length filters applied\n")
        f.write(f"---\n")
        f.write(f"Word count — mean:           {wc_arr.mean():.1f}\n")
        f.write(f"Word count — median:         {np.median(wc_arr):.1f}\n")
        f.write(f"Word count — std:            {wc_arr.std():.1f}\n")
        f.write(f"Word count — min:            {wc_arr.min()}\n")
        f.write(f"Word count — max:            {wc_arr.max()}\n")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            f.write(f"Word count — {p}th pctl:     {np.percentile(wc_arr, p):.0f}\n")
        f.write(f"Build time:                  {elapsed:.1f}s\n")
        f.write("\n")

    logger.info(f"Statistics appended to {stats_path}")


def sanity_check():
    proc_path = os.path.join(config.PROCESSED_DIR, "yahoo_corpus.jsonl")
    if not os.path.exists(proc_path):
        logger.error("SANITY CHECK FAILED: yahoo_corpus.jsonl not found")
        return
    docs = []
    with open(proc_path) as f:
        for line in f:
            docs.append(json.loads(line))
    required_keys = {"chunk_id", "source_id", "question_title", "text", "word_count"}
    random.seed(config.SEED)
    samples = random.sample(docs, min(5, len(docs)))
    for s in samples:
        if not required_keys.issubset(s.keys()):
            logger.error(f"SANITY CHECK FAILED: missing keys in {s['chunk_id']}")
            return
    logger.info(f"Total documents loaded: {len(docs)}")
    logger.info("SANITY CHECK PASSED")


def build_yahoo_corpus():
    proc_path = os.path.join(config.PROCESSED_DIR, "yahoo_corpus.jsonl")
    if os.path.exists(proc_path):
        with open(proc_path) as f:
            n = sum(1 for _ in f)
        if n > 0:
            logger.info(f"Yahoo corpus already exists with {n} docs. Skipping.")
            return

    t0 = time.time()
    docs, total_rows = load_all_yahoo_posts()
    save_results(docs)
    elapsed = time.time() - t0
    write_statistics(docs, total_rows, elapsed)
    logger.info(f"Yahoo corpus build completed in {elapsed:.1f}s")


if __name__ == "__main__":
    build_yahoo_corpus()
    sanity_check()
