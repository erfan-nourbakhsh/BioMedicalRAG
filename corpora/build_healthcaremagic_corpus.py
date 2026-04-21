import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import warnings

import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from bs4.exceptions import ParserRejectedMarkup
from tqdm import tqdm

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import config_extended as ext_config

os.environ["HF_HOME"] = config.HF_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HF_REPO = "wangrongsheng/HealthCareMagic-100k-en"
OUTPUT_FILENAME = "healthcaremagic_corpus.jsonl"


def clean_html(text) -> str:
    if not text:
        return ""
    try:
        soup = BeautifulSoup(str(text), "html.parser")
        cleaned = soup.get_text(separator=" ")
    except ParserRejectedMarkup:
        cleaned = re.sub(r"<[^>]+>", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _first_present(row: dict, names: list[str]) -> str:
    for name in names:
        if name in row and row[name] is not None:
            value = clean_html(row[name])
            if value:
                return value
    return ""


def load_all_healthcaremagic_posts():
    """Load the full HealthCareMagic-100k-en dataset from Hugging Face."""
    from datasets import load_dataset

    logger.info(f"Loading full Hugging Face dataset: {HF_REPO}")
    ds = load_dataset(HF_REPO, split="train", cache_dir=config.HF_HOME)

    docs = []
    skipped_quality = 0
    min_answer_words = ext_config.YAHOO_MIN_ANSWER_WORDS
    min_total_words = ext_config.YAHOO_MIN_TOTAL_WORDS

    question_fields = [
        "question",
        "input",
        "query",
        "Question",
        "Description",
        "instruction",
        "prompt",
    ]
    answer_fields = [
        "answer",
        "output",
        "response",
        "Answer",
        "Doctor",
        "doctor_answer",
        "completion",
    ]
    title_fields = ["title", "question_title", "Title"]

    for i, row in enumerate(tqdm(ds, desc="Processing HealthCareMagic", unit="doc")):
        question_text = _first_present(row, question_fields)
        answer_text = _first_present(row, answer_fields)
        title = _first_present(row, title_fields)

        if not question_text and not answer_text:
            text = clean_html(row.get("text", ""))
        else:
            text = f"{title} {question_text} {answer_text}".strip()

        answer_wc = len(answer_text.split()) if answer_text else 0
        if answer_wc < min_answer_words or len(text.split()) < min_total_words:
            skipped_quality += 1
            continue

        source_id = clean_html(row.get("id", "")) or f"healthcaremagic_{i}"
        docs.append(
            {
                "chunk_id": f"healthcaremagic_{len(docs)}",
                "source_id": source_id,
                "source_type": "healthcaremagic_qa",
                "question_title": title[:200],
                "question_text": question_text,
                "answer_text": answer_text,
                "text": text,
                "word_count": len(text.split()),
            }
        )

    logger.info(f"Rows loaded: {len(ds):,}")
    logger.info(f"Skipped (quality filter): {skipped_quality:,}")
    logger.info(f"Documents kept: {len(docs):,}")
    return docs, len(ds), skipped_quality


def save_results(docs):
    proc_path = os.path.join(config.PROCESSED_DIR, OUTPUT_FILENAME)
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    with open(proc_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(docs):,} documents to {proc_path}")


def write_statistics(docs, total_rows, skipped_quality, elapsed):
    wc = np.array([d["word_count"] for d in docs], dtype=np.float32)
    stats_path = os.path.join(config.BASE_DIR, "statistics.txt")
    mode = "a" if os.path.exists(stats_path) else "w"

    with open(stats_path, mode, encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("HEALTHCAREMAGIC CORPUS STATISTICS\n")
        f.write(f"Source: {HF_REPO}\n")
        f.write(f"Date built: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total rows in dataset:       {total_rows:,}\n")
        f.write(f"Documents in corpus:         {len(docs):,}\n")
        f.write(f"Skipped (quality filter):    {skipped_quality:,}\n")
        f.write("Content: question + doctor answer\n")
        f.write("Chunking:                    None (whole QA rows)\n")
        f.write("Filtering:                   Minimum answer and total length filters applied\n")
        f.write("---\n")
        if len(wc):
            f.write(f"Word count - mean:           {wc.mean():.1f}\n")
            f.write(f"Word count - median:         {np.median(wc):.1f}\n")
            f.write(f"Word count - std:            {wc.std():.1f}\n")
            f.write(f"Word count - min:            {int(wc.min())}\n")
            f.write(f"Word count - max:            {int(wc.max())}\n")
            for p in [10, 25, 50, 75, 90, 95, 99]:
                f.write(f"Word count - {p}th pctl:     {np.percentile(wc, p):.0f}\n")
        f.write(f"Build time:                  {elapsed:.1f}s\n")
        f.write("\n")

    logger.info(f"Statistics appended to {stats_path}")


def sanity_check():
    proc_path = os.path.join(config.PROCESSED_DIR, OUTPUT_FILENAME)
    if not os.path.exists(proc_path):
        logger.error(f"SANITY CHECK FAILED: {OUTPUT_FILENAME} not found")
        return

    samples = []
    with open(proc_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= 5:
                break

    if not samples:
        logger.error("SANITY CHECK FAILED: empty corpus")
        return

    required_keys = {"chunk_id", "source_id", "question_title", "question_text", "answer_text", "text", "word_count"}
    for s in samples:
        if not required_keys.issubset(s.keys()):
            logger.error(f"SANITY CHECK FAILED: missing keys in {s.get('chunk_id', '?')}")
            return

    try:
        out = subprocess.run(["wc", "-l", proc_path], capture_output=True, text=True, check=True)
        doc_count = int(out.stdout.split()[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        doc_count = -1

    random.seed(config.SEED)
    logger.info("Sanity check - sample HealthCareMagic documents:")
    for s in samples:
        logger.info(f"  [{s['chunk_id']}] source={s['source_id']} | {s['text'][:120]}...")
    if doc_count >= 0:
        logger.info(f"Total documents (wc -l): {doc_count:,}")
    logger.info("SANITY CHECK PASSED")


def build_healthcaremagic_corpus(force: bool = False):
    proc_path = os.path.join(config.PROCESSED_DIR, OUTPUT_FILENAME)
    if os.path.exists(proc_path) and not force:
        with open(proc_path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        if n > 0:
            logger.info(
                f"HealthCareMagic corpus already exists with {n:,} docs. "
                "Skipping. Use --force to rebuild."
            )
            return

    if force and os.path.exists(proc_path):
        os.remove(proc_path)
        logger.info(f"Removed existing {proc_path} (--force).")

    t0 = time.time()
    docs, total_rows, skipped_quality = load_all_healthcaremagic_posts()
    save_results(docs)
    elapsed = time.time() - t0
    write_statistics(docs, total_rows, skipped_quality, elapsed)
    logger.info(f"HealthCareMagic corpus build completed in {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the HealthCareMagic retrieval corpus.")
    p.add_argument("--force", action="store_true", help="Rebuild even if JSONL already exists.")
    args = p.parse_args()

    build_healthcaremagic_corpus(force=args.force)
    sanity_check()
