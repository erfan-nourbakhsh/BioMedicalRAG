import json
import logging
import os
import sys

import tantivy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from retrieval.bm25_indexer import bm25_tantivy_schema
from retrieval.retrieval_utils import build_context_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _meta_from_doc(doc) -> dict:
    raw = doc.get_first("meta")
    if raw is None:
        return {}
    return json.loads(bytes(bytearray(raw)))


class BM25Retriever:

    def __init__(self, corpus_name: str):
        corpus_name = config.normalize_corpus_name(corpus_name)
        index_dir = config.get_bm25_index_dir(corpus_name)
        if not tantivy.Index.exists(index_dir):
            raise FileNotFoundError(
                f"Tantivy BM25 index not found: {index_dir}\n"
                f"Build with: python retrieval/bm25_indexer.py"
            )

        logger.info(f"Opening Tantivy BM25 index at {index_dir}...")
        self.index = tantivy.Index(bm25_tantivy_schema(), index_dir, reuse=True)
        self.index.config_reader(reload_policy="OnCommit", num_warmers=0)
        self.index.reload()
        self.corpus_name = corpus_name
        self._searcher = self.index.searcher()
        logger.info(f"Loaded Tantivy index for {corpus_name} ({self._searcher.num_docs:,} documents)")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        q = query.strip()
        if not q:
            return []

        parsed, errs = self.index.parse_query_lenient(q, ["contents"])
        if errs:
            logger.debug("Query lenient parse notes for %r: %s", q, errs)

        result = self._searcher.search(parsed, top_k)
        out = []
        for rank, (score, addr) in enumerate(result.hits, start=1):
            doc = self._searcher.doc(addr)
            meta = _meta_from_doc(doc)
            out.append({
                "rank": rank,
                "score": float(score),
                "text": meta.get("text", ""),
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "corpus": self.corpus_name,
            })
        return out

    def get_context_string(self, query: str, top_k: int = 5, max_chars: int = 1500) -> str:
        return build_context_string(self.retrieve(query, top_k=top_k), max_chars=max_chars)


def sanity_check():
    queries = [
        "What are the symptoms and treatment of diabetes?",
        "How is hypertension diagnosed and managed?",
        "What causes chronic lower back pain?",
    ]

    for corpus_name in ["bioasq", "yahoo"]:
        retriever = BM25Retriever(corpus_name)
        logger.info(f"\nTesting {corpus_name} retriever:")
        for q in queries:
            results = retriever.retrieve(q, top_k=2)
            logger.info(f"\n  Query: {q}")
            for r in results:
                logger.info(f"    Rank {r['rank']} (score={r['score']:.3f}): {r['text'][:100]}...")

    logger.info("\nSANITY CHECK PASSED")


if __name__ == "__main__":
    sanity_check()
