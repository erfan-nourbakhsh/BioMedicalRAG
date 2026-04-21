import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BIOASQ_HF_REPO = "jmhb/pubmed_bioasq_2022"
BIOASQ_PARQUET = "data/allMeSH_2022.parquet"

PARQUET_BATCH_SIZE = 16_384
STATS_SAMPLE_CAP = 100_000


def ensure_bioasq_parquet(redownload: bool = False) -> str:
    """
    Download the full BioASQ Task A parquet from Hugging Face (~67 GB on disk).
    Creates data/raw/bioasq/allMeSH_2022.parquet -> symlink to the HF cache file
    (no second full copy). Returns the resolved path to the real parquet file.
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(config.BIOASQ_RAW_DIR, exist_ok=True)
    link_path = config.BIOASQ_PARQUET_LINK

    if redownload and os.path.lexists(link_path):
        os.remove(link_path)

    logger.info(f"Fetching parquet from {BIOASQ_HF_REPO} (full ~16.2M articles; download is large).")
    cached_path = hf_hub_download(
        repo_id=BIOASQ_HF_REPO,
        filename=BIOASQ_PARQUET,
        repo_type="dataset",
        cache_dir=config.HF_HOME,
    )
    cached_path = os.path.abspath(os.path.realpath(cached_path))
    logger.info(f"Parquet available at: {cached_path}")

    if not os.path.lexists(link_path):
        os.symlink(cached_path, link_path)
        logger.info(f"Symlinked project copy: {link_path} -> {cached_path}")
    else:
        if os.path.realpath(link_path) != cached_path:
            os.remove(link_path)
            os.symlink(cached_path, link_path)
            logger.info(f"Updated symlink: {link_path} -> {cached_path}")

    return cached_path


def _cell_str(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


def _update_welford(n: int, mean: float, m2: float, x: float):
    n += 1
    delta = x - mean
    mean += delta / n
    delta2 = x - mean
    m2 += delta * delta2
    return n, mean, m2


def _reservoir_add(sample: list, cap: int, x: float, seen: int):
    if len(sample) < cap:
        sample.append(x)
    else:
        j = random.randint(1, seen)
        if j <= cap:
            sample[j - 1] = x


def build_corpus_streaming(
    parquet_path: str,
    output_path: str,
    max_docs: Optional[int] = None,
) -> tuple:
    """
    Stream parquet row groups; write one JSONL doc per article with abstract.
    Returns (docs_written, skipped_no_abstract, total_rows, mean_wc, std_wc, min_wc, max_wc, wc_sample).
    """
    pf = pq.ParquetFile(parquet_path)
    schema_names = set(pf.schema_arrow.names)
    need = {"title", "abstractText", "pmid"}
    missing = need - schema_names
    if missing:
        raise ValueError(f"Parquet missing columns {missing}; have {sorted(schema_names)}")

    total_rows = pf.metadata.num_rows
    logger.info(f"Parquet rows (metadata): {total_rows:,}")

    doc_idx = 0
    skipped_no_abstract = 0
    rows_seen = 0

    n_wc, mean_wc, m2_wc = 0, 0.0, 0.0
    min_wc, max_wc = 10**9, 0
    wc_sample: list = []
    random.seed(config.SEED)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pbar = tqdm(total=total_rows, desc="BioASQ corpus", unit="row")

    with open(output_path, "w", encoding="utf-8") as out:
        for batch in pf.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=list(need)):
            cols = batch.to_pydict()
            titles = cols["title"]
            abstracts = cols["abstractText"]
            pmids = cols["pmid"]
            batch_len = len(titles)
            rows_seen += batch_len
            pbar.update(batch_len)

            for title, abstract, pmid in zip(titles, abstracts, pmids):
                if max_docs is not None and doc_idx >= max_docs:
                    std = (m2_wc / n_wc) ** 0.5 if n_wc else 0.0
                    pbar.close()
                    return (
                        doc_idx,
                        skipped_no_abstract,
                        rows_seen,
                        mean_wc,
                        std,
                        min_wc if n_wc else 0,
                        max_wc,
                        wc_sample,
                    )

                title_s = _cell_str(title)
                abstract_s = _cell_str(abstract)
                if not abstract_s:
                    skipped_no_abstract += 1
                    continue

                text = f"{title_s} {abstract_s}".strip()
                pmid_s = _cell_str(pmid)
                wc = len(text.split())

                doc = {
                    "chunk_id": f"bioasq_{doc_idx}",
                    "source_id": pmid_s,
                    "source_type": "pubmed_article",
                    "title": title_s[:200],
                    "abstract": abstract_s,
                    "text": text,
                    "word_count": wc,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                doc_idx += 1

                n_wc, mean_wc, m2_wc = _update_welford(n_wc, mean_wc, m2_wc, float(wc))
                min_wc = min(min_wc, wc)
                max_wc = max(max_wc, wc)
                _reservoir_add(wc_sample, STATS_SAMPLE_CAP, float(wc), doc_idx)

    pbar.close()
    std = (m2_wc / n_wc) ** 0.5 if n_wc else 0.0
    return (
        doc_idx,
        skipped_no_abstract,
        rows_seen,
        mean_wc,
        std,
        min_wc if n_wc else 0,
        max_wc,
        wc_sample,
    )


def write_statistics(
    total_raw: int,
    docs_written: int,
    skipped: int,
    mean_wc: float,
    std_wc: float,
    min_wc: int,
    max_wc: int,
    wc_sample: list,
    elapsed: float,
):
    stats_path = os.path.join(config.BASE_DIR, "statistics.txt")
    mode = "a" if os.path.exists(stats_path) else "w"

    pct_lines = ""
    if wc_sample:
        arr = np.array(wc_sample)
        for p in [10, 25, 50, 75, 90, 95, 99]:
            pct_lines += f"Word count — {p}th pctl:     {np.percentile(arr, p):.0f}\n"

    with open(stats_path, mode) as f:
        f.write("=" * 60 + "\n")
        f.write("BIOASQ TASK A CORPUS STATISTICS\n")
        f.write(f"Source: {BIOASQ_HF_REPO} (BioASQ Task 10a, Training v.2022)\n")
        f.write(f"Official: https://participants-area.bioasq.org/datasets/\n")
        f.write(f"Date built: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total rows in parquet:       {total_raw:,}\n")
        f.write(f"Documents in corpus:         {docs_written:,}\n")
        f.write(f"Skipped (no abstract):       {skipped:,}\n")
        f.write(f"Content: title + abstractText (PubMed articles)\n")
        f.write(f"Chunking:                    None (whole articles)\n")
        f.write(f"Build:                       streaming (pyarrow)\n")
        f.write(f"---\n")
        f.write(f"Word count — mean:           {mean_wc:.1f}\n")
        f.write(f"Word count — std:            {std_wc:.1f}\n")
        f.write(f"Word count — min:            {min_wc}\n")
        f.write(f"Word count — max:            {max_wc}\n")
        f.write(pct_lines)
        if wc_sample:
            f.write(f"(Percentiles from reservoir sample n={len(wc_sample):,})\n")
        f.write(f"Build time:                  {elapsed:.1f}s\n")
        f.write("\n")

    logger.info(f"Statistics appended to {stats_path}")


def sanity_check():
    proc_path = os.path.join(config.PROCESSED_DIR, "bioasq_corpus.jsonl")
    if not os.path.exists(proc_path):
        logger.error("SANITY CHECK FAILED: bioasq_corpus.jsonl not found")
        return
    samples = []
    with open(proc_path, encoding="utf-8") as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            samples.append(json.loads(line))

    if len(samples) < 1:
        logger.error("SANITY CHECK FAILED: empty corpus")
        return

    try:
        out = subprocess.run(
            ["wc", "-l", proc_path], capture_output=True, text=True, check=True
        )
        doc_count = int(out.stdout.split()[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        doc_count = -1

    required_keys = {"chunk_id", "source_id", "source_type", "text", "word_count"}
    logger.info("Sanity check — sample BioASQ Task A documents:")
    for s in samples[:5]:
        if not required_keys.issubset(s.keys()):
            logger.error(f"SANITY CHECK FAILED: missing keys in {s.get('chunk_id', '?')}")
            return
        logger.info(f"  [{s['chunk_id']}] pmid={s['source_id']} | {s['text'][:120]}...")
    if doc_count >= 0:
        logger.info(f"Total documents (wc -l): {doc_count:,}")
    else:
        logger.info("Total documents: (could not run wc -l)")
    logger.info("SANITY CHECK PASSED")


def build_bioasq_corpus(
    force: bool = False,
    download_only: bool = False,
    max_docs: Optional[int] = None,
    redownload_parquet: bool = False,
):
    proc_path = os.path.join(config.PROCESSED_DIR, "bioasq_corpus.jsonl")

    parquet_path = ensure_bioasq_parquet(redownload=redownload_parquet)
    if download_only:
        logger.info("Download-only: skipping JSONL corpus build.")
        return

    if os.path.exists(proc_path) and not force:
        with open(proc_path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        if n > 0:
            logger.info(f"BioASQ corpus already exists with {n:,} docs. Skipping. Use --force to rebuild.")
            return

    if force and os.path.exists(proc_path):
        os.remove(proc_path)
        logger.info(f"Removed existing {proc_path} (--force).")

    t0 = time.time()
    pf = pq.ParquetFile(parquet_path)
    total_raw = pf.metadata.num_rows

    docs_written, skipped, rows_seen, mean_wc, std_wc, min_wc, max_wc, wc_sample = (
        build_corpus_streaming(parquet_path, proc_path, max_docs=max_docs)
    )
    elapsed = time.time() - t0

    logger.info(
        f"Saved {docs_written:,} documents to {proc_path} "
        f"(skipped {skipped:,} without abstract; scanned {rows_seen:,} rows)"
    )
    write_statistics(
        total_raw, docs_written, skipped, mean_wc, std_wc, min_wc, max_wc, wc_sample, elapsed
    )
    logger.info(f"BioASQ Task A corpus build completed in {elapsed:.1f}s")


def _parse_args():
    p = argparse.ArgumentParser(description="Download full BioASQ Task A parquet and build bioasq_corpus.jsonl")
    p.add_argument("--force", action="store_true", help="Rebuild JSONL even if it already exists")
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Only download parquet + symlink; do not build JSONL",
    )
    p.add_argument(
        "--redownload-parquet",
        action="store_true",
        help="Re-fetch parquet from Hugging Face (large)",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap documents written (for testing; default = all with abstract)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_bioasq_corpus(
        force=args.force,
        download_only=args.download_only,
        max_docs=args.max_docs,
        redownload_parquet=args.redownload_parquet,
    )
    if not args.download_only:
        sanity_check()
