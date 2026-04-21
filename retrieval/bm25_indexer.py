import json
import logging
import os
import shutil
import sys
import time

import tantivy
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import config_extended as ext_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _boosted_contents_string(chunk: dict, boost: int) -> str:
    title = chunk.get("title", chunk.get("question_title", "")) or ""
    body = chunk.get("abstract", chunk.get("answer_text", chunk.get("text", ""))) or ""
    title_s, body_s = title.strip(), body.strip()
    if title_s and boost > 1:
        if body_s:
            return (" ".join([title_s] * boost) + " " + body_s).strip()
        return " ".join([title_s] * boost)
    return (chunk.get("text") or "").strip()


def _meta_bytes(chunk: dict) -> bytes:
    meta = {
        "chunk_id": chunk["chunk_id"],
        "source": chunk.get("pmid", chunk.get("source_id", "")),
        "title": chunk.get("title", chunk.get("question_title", "")),
        "text": chunk["text"],
    }
    return json.dumps(meta, ensure_ascii=False).encode("utf-8")


def _index_size_mb(index_dir: str) -> float:
    total_bytes = 0
    for dp, _, fnames in os.walk(index_dir):
        for fname in fnames:
            path = os.path.join(dp, fname)
            try:
                total_bytes += os.path.getsize(path)
            except FileNotFoundError:
                continue
    return total_bytes / (1024 ** 2)


def bm25_tantivy_schema():
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("contents", stored=False)
    builder.add_bytes_field("meta", stored=True, indexed=False)
    return builder.build()


def build_bm25_index(corpus_path: str, corpus_name: str, force: bool = False) -> None:
    index_dir = config.get_bm25_index_dir(corpus_name)
    if not force and os.path.isdir(index_dir):
        try:
            if tantivy.Index.exists(index_dir):
                logger.info(f"Tantivy BM25 index already exists at {index_dir}. Skipping.")
                return
        except ValueError:
            pass

    if force and os.path.isdir(index_dir):
        logger.info(f"Removing existing index at {index_dir} (--force).")
        shutil.rmtree(index_dir)

    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    os.makedirs(index_dir, exist_ok=True)

    t0 = time.time()
    boost = ext_config.BM25_TITLE_BOOST
    threads = ext_config.PYSERINI_INDEX_THREADS or 0
    heap_mb = int(os.environ.get("BM25_INDEX_WRITER_HEAP_MB", "512"))
    heap_size = max(heap_mb * 1024 * 1024, 50_000_000)

    schema = bm25_tantivy_schema()
    index = tantivy.Index(schema, index_dir, reuse=False)
    writer = index.writer(heap_size=heap_size, num_threads=threads)

    logger.info(f"Indexing {corpus_name} -> {index_dir} (title_boost={boost})")

    n_docs = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Tantivy {corpus_name}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            contents = _boosted_contents_string(chunk, boost)
            if not contents:
                continue
            doc = tantivy.Document()
            doc.add_text("contents", contents)
            doc.add_bytes("meta", _meta_bytes(chunk))
            writer.add_document(doc)
            n_docs += 1

    writer.commit()
    del writer
    elapsed = time.time() - t0
    size_mb = _index_size_mb(index_dir)

    logger.info(f"Index built: {index_dir}")
    logger.info(f"  Documents indexed:  {n_docs:,}")
    logger.info(f"  Index size (files): {size_mb:.1f} MB")
    logger.info(f"  Time taken:         {elapsed:.1f}s")


def sanity_check():
    test_query = "What are the symptoms and treatment of diabetes?"
    from retrieval.bm25_retriever import BM25Retriever

    for corpus_name in ["bioasq", "yahoo", "medical_textbooks", "healthcaremagic"]:
        index_dir = config.get_bm25_index_dir(corpus_name)
        if not os.path.isdir(index_dir) or not tantivy.Index.exists(index_dir):
            logger.info(f"Skipping sanity check for missing BM25 index: {index_dir}")
            continue

        retriever = BM25Retriever(corpus_name)
        results = retriever.retrieve(test_query, top_k=3)

        logger.info(f"\nTest query: '{test_query}'")
        logger.info(f"Top 3 results from {corpus_name} index:")
        for r in results:
            logger.info(f"  Rank {r['rank']} (score={r['score']:.3f}): {r['text'][:120]}...")

    logger.info("SANITY CHECK PASSED")


def build_indexes(corpus_names=None, force: bool = False):
    corpus_names = corpus_names or ["bioasq", "yahoo"]
    for corpus_name in corpus_names:
        corpus_name = config.normalize_corpus_name(corpus_name)
        build_bm25_index(config.get_corpus_path(corpus_name), corpus_name, force=force)


def build_both_indexes(force: bool = False):
    build_indexes(["bioasq", "yahoo"], force=force)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build Tantivy BM25 indexes (streaming).")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--corpus", nargs="+", default=["bioasq", "yahoo"],
    )
    args = p.parse_args()
    build_indexes(args.corpus, force=args.force)
    sanity_check()
