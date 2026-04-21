import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import json
import logging
import random
import sys
import time

from Bio import Entrez, Medline
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

Entrez.email = "research@example.com"

SEARCH_QUERY = (
    '("disease"[MeSH Terms] OR "drug therapy"[MeSH Terms] '
    'OR "diagnosis"[MeSH Terms] OR "symptoms"[MeSH Terms] '
    'OR "patient care"[MeSH Terms])'
)


def fetch_pubmed_ids(max_results):
    logger.info(f"Searching PubMed for up to {max_results} articles...")
    for attempt in range(1, 4):
        try:
            handle = Entrez.esearch(
                db="pubmed", term=SEARCH_QUERY,
                retmax=max_results, sort="relevance",
                usehistory="y"
            )
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]
            logger.info(f"Found {len(id_list)} PubMed IDs")
            return id_list, record.get("WebEnv"), record.get("QueryKey")
        except Exception as e:
            logger.warning(f"esearch attempt {attempt}/3 failed: {e}")
            if attempt == 3:
                raise
            time.sleep(5)


def fetch_abstracts(id_list, webenv, query_key, batch_size=200):
    articles = []
    total = len(id_list)
    logger.info(f"Fetching {total} abstracts in batches of {batch_size}...")

    for start in tqdm(range(0, total, batch_size), desc="Fetching PubMed"):
        for attempt in range(1, 4):
            try:
                handle = Entrez.efetch(
                    db="pubmed", rettype="medline", retmode="text",
                    retstart=start, retmax=batch_size,
                    webenv=webenv, query_key=query_key
                )
                records = list(Medline.parse(handle))
                handle.close()
                for rec in records:
                    pmid = rec.get("PMID", "")
                    title = rec.get("TI", "")
                    abstract = rec.get("AB", "")
                    if pmid and title:
                        articles.append({
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract
                        })
                break
            except Exception as e:
                logger.warning(f"efetch attempt {attempt}/3 at offset {start} failed: {e}")
                if attempt == 3:
                    logger.error(f"Skipping batch at offset {start}")
                time.sleep(5)
        time.sleep(0.4)

    logger.info(f"Fetched {len(articles)} raw articles")
    return articles


def deduplicate(articles):
    """Remove duplicate PMIDs only. No filtering by length or content."""
    seen = set()
    deduped = []
    dups = 0
    for art in articles:
        if art["pmid"] in seen:
            dups += 1
            continue
        seen.add(art["pmid"])
        deduped.append(art)
    if dups:
        logger.info(f"Removed {dups} duplicate PMIDs")
    return deduped


def build_documents(articles):
    """Convert articles to corpus documents. No chunking — whole documents."""
    docs = []
    for i, art in enumerate(articles):
        doc_text = art["title"]
        if art["abstract"]:
            doc_text = art["title"] + ". " + art["abstract"]
        wc = len(doc_text.split())
        docs.append({
            "chunk_id": f"pubmed_{i}",
            "pmid": art["pmid"],
            "title": art["title"],
            "text": doc_text,
            "word_count": wc,
        })
    logger.info(f"Built {len(docs)} documents from {len(articles)} articles (no chunking)")
    return docs


def save_results(articles, docs):
    raw_path = os.path.join(config.RAW_DIR, "pubmed", "pubmed_raw.jsonl")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w") as f:
        for art in articles:
            f.write(json.dumps(art) + "\n")
    logger.info(f"Saved {len(articles)} raw articles to {raw_path}")

    proc_path = os.path.join(config.PROCESSED_DIR, "pubmed_corpus.jsonl")
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    with open(proc_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    logger.info(f"Saved {len(docs)} documents to {proc_path}")


def write_statistics(articles_raw, docs, elapsed):
    import numpy as np
    wc = np.array([d["word_count"] for d in docs])
    has_abstract = sum(1 for d in docs if d["word_count"] > len(d["title"].split()) + 2)

    stats_path = os.path.join(config.BASE_DIR, "statistics.txt")
    mode = "a" if os.path.exists(stats_path) else "w"
    with open(stats_path, mode) as f:
        f.write("=" * 60 + "\n")
        f.write("PUBMED CORPUS STATISTICS\n")
        f.write(f"Source: PubMed via Entrez API\n")
        f.write(f"Date built: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Articles fetched (raw):      {len(articles_raw)}\n")
        f.write(f"Documents in corpus (all):   {len(docs)}\n")
        f.write(f"With abstract:               {has_abstract}\n")
        f.write(f"Title-only (no abstract):    {len(docs) - has_abstract}\n")
        f.write(f"Chunking:                    None (whole documents)\n")
        f.write(f"Filtering:                   None (all articles kept)\n")
        f.write(f"---\n")
        f.write(f"Word count — mean:           {wc.mean():.1f}\n")
        f.write(f"Word count — median:         {np.median(wc):.1f}\n")
        f.write(f"Word count — std:            {wc.std():.1f}\n")
        f.write(f"Word count — min:            {wc.min()}\n")
        f.write(f"Word count — max:            {wc.max()}\n")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            f.write(f"Word count — {p}th pctl:     {np.percentile(wc, p):.0f}\n")
        f.write(f"Build time:                  {elapsed:.1f}s\n")
        f.write("\n")

    logger.info(f"Statistics appended to {stats_path}")


def sanity_check():
    proc_path = os.path.join(config.PROCESSED_DIR, "pubmed_corpus.jsonl")
    if not os.path.exists(proc_path):
        logger.error("SANITY CHECK FAILED: pubmed_corpus.jsonl not found")
        return
    docs = []
    with open(proc_path) as f:
        for line in f:
            docs.append(json.loads(line))
    random.seed(config.SEED)
    samples = random.sample(docs, min(5, len(docs)))
    logger.info("Sanity check — 5 random PubMed documents:")
    for s in samples:
        logger.info(f"  [{s['chunk_id']}] PMID={s['pmid']} | {s['text'][:120]}...")
    required_keys = {"chunk_id", "pmid", "title", "text", "word_count"}
    for s in samples:
        if not required_keys.issubset(s.keys()):
            logger.error(f"SANITY CHECK FAILED: missing keys in {s['chunk_id']}")
            return
    logger.info(f"Total documents: {len(docs)}")
    logger.info("SANITY CHECK PASSED")


def build_pubmed_corpus():
    proc_path = os.path.join(config.PROCESSED_DIR, "pubmed_corpus.jsonl")
    if os.path.exists(proc_path):
        with open(proc_path) as f:
            n = sum(1 for _ in f)
        if n > 0:
            logger.info(f"PubMed corpus already exists with {n} docs. Skipping.")
            return

    t0 = time.time()
    id_list, webenv, query_key = fetch_pubmed_ids(config.PUBMED_MAX)
    articles_raw = fetch_abstracts(id_list, webenv, query_key)
    articles = deduplicate(articles_raw)
    docs = build_documents(articles)
    save_results(articles, docs)
    elapsed = time.time() - t0
    write_statistics(articles_raw, docs, elapsed)
    logger.info(f"PubMed corpus build completed in {elapsed:.1f}s")


if __name__ == "__main__":
    build_pubmed_corpus()
    sanity_check()
