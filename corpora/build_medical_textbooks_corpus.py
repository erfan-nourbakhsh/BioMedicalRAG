import json
import logging
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.environ["HF_HOME"] = config.HF_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HF_REPO = "MedRAG/textbooks"
OUTPUT_FILENAME = "medical_textbooks_corpus.jsonl"


def _clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def load_textbook_snippets():
    """Load the full MedRAG textbooks dataset from Hugging Face."""
    from datasets import load_dataset

    logger.info(f"Loading full Hugging Face dataset: {HF_REPO}")
    ds = load_dataset(HF_REPO, split="train", cache_dir=config.HF_HOME)

    docs = []
    skipped_empty = 0
    for i, row in enumerate(tqdm(ds, desc="Processing medical textbooks", unit="doc")):
        title = _clean_text(row.get("title", ""))
        content = _clean_text(row.get("content", ""))
        contents = _clean_text(row.get("contents", ""))

        text = contents or f"{title} {content}".strip()
        if not text:
            skipped_empty += 1
            continue

        source_id = _clean_text(row.get("id", "")) or f"medical_textbooks_{i}"
        docs.append(
            {
                "chunk_id": f"medical_textbooks_{len(docs)}",
                "source_id": source_id,
                "source_type": "medical_textbook_snippet",
                "title": title[:200],
                "content": content,
                "text": text,
                "word_count": _word_count(text),
            }
        )

    logger.info(f"Rows loaded: {len(ds):,}")
    logger.info(f"Skipped empty rows: {skipped_empty:,}")
    logger.info(f"Documents kept: {len(docs):,}")
    return docs, len(ds), skipped_empty


def save_results(docs):
    proc_path = os.path.join(config.PROCESSED_DIR, OUTPUT_FILENAME)
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    with open(proc_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(docs):,} documents to {proc_path}")


def write_statistics(docs, total_rows, skipped_empty, elapsed):
    wc = np.array([d["word_count"] for d in docs], dtype=np.float32)
    stats_path = os.path.join(config.BASE_DIR, "statistics.txt")
    mode = "a" if os.path.exists(stats_path) else "w"

    with open(stats_path, mode, encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MEDICAL TEXTBOOKS CORPUS STATISTICS\n")
        f.write(f"Source: {HF_REPO}\n")
        f.write(f"Date built: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total rows in dataset:       {total_rows:,}\n")
        f.write(f"Documents in corpus:         {len(docs):,}\n")
        f.write(f"Skipped empty rows:          {skipped_empty:,}\n")
        f.write("Content: title + content/contents (MedRAG textbook snippets)\n")
        f.write("Chunking:                    Source-provided snippets\n")
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

    required_keys = {"chunk_id", "source_id", "source_type", "title", "text", "word_count"}
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
    logger.info("Sanity check - sample medical textbook documents:")
    for s in samples:
        logger.info(f"  [{s['chunk_id']}] source={s['source_id']} | {s['text'][:120]}...")
    if doc_count >= 0:
        logger.info(f"Total documents (wc -l): {doc_count:,}")
    logger.info("SANITY CHECK PASSED")


def build_medical_textbooks_corpus(force: bool = False):
    proc_path = os.path.join(config.PROCESSED_DIR, OUTPUT_FILENAME)
    if os.path.exists(proc_path) and not force:
        with open(proc_path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        if n > 0:
            logger.info(
                f"Medical textbooks corpus already exists with {n:,} docs. "
                "Skipping. Use --force to rebuild."
            )
            return

    if force and os.path.exists(proc_path):
        os.remove(proc_path)
        logger.info(f"Removed existing {proc_path} (--force).")

    t0 = time.time()
    docs, total_rows, skipped_empty = load_textbook_snippets()
    save_results(docs)
    elapsed = time.time() - t0
    write_statistics(docs, total_rows, skipped_empty, elapsed)
    logger.info(f"Medical textbooks corpus build completed in {elapsed:.1f}s")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the MedRAG medical textbooks retrieval corpus.")
    p.add_argument("--force", action="store_true", help="Rebuild even if JSONL already exists.")
    args = p.parse_args()

    build_medical_textbooks_corpus(force=args.force)
    sanity_check()
