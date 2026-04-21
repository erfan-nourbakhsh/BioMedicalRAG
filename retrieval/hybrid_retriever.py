import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import logging
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import config_extended as ext_config
from retrieval.retrieval_utils import build_context_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class HybridRetriever:

    def __init__(self, corpus_name: str, bm25_retriever, medcpt_retriever):
        corpus_name = config.normalize_corpus_name(corpus_name)
        self.corpus_name = corpus_name
        self.bm25 = bm25_retriever
        self.medcpt = medcpt_retriever
        self.k = ext_config.RRF_K
        logger.info(f"Hybrid retriever ready: {corpus_name} (RRF k={self.k})")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        n_candidates = max(top_k * 4, 20)

        bm25_results = self.bm25.retrieve(query, top_k=n_candidates)
        medcpt_results = self.medcpt.retrieve(query, top_k=n_candidates)

        rrf_scores = {}

        for r in bm25_results:
            key = r["text"][:200]
            rrf_scores.setdefault(key, {"score": 0.0, "data": r})
            rrf_scores[key]["score"] += 1.0 / (self.k + r["rank"])

        for r in medcpt_results:
            key = r["text"][:200]
            rrf_scores.setdefault(key, {"score": 0.0, "data": r})
            rrf_scores[key]["score"] += 1.0 / (self.k + r["rank"])

        ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

        results = []
        for rank, item in enumerate(ranked[:top_k]):
            d = item["data"]
            results.append({
                "rank": rank + 1,
                "score": float(item["score"]),
                "text": d["text"],
                "source": d["source"],
                "title": d["title"],
                "corpus": self.corpus_name,
            })
        return results

    def get_context_string(self, query: str, top_k: int = 5, max_chars: int = 1500) -> str:
        return build_context_string(self.retrieve(query, top_k=top_k), max_chars=max_chars)


def sanity_check():
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.medcpt_retriever import MedCPTRetriever

    query = "What are the symptoms and treatment of diabetes?"

    for corpus in ["bioasq", "yahoo"]:
        bm25 = BM25Retriever(corpus)
        medcpt = MedCPTRetriever(corpus)
        hybrid = HybridRetriever(corpus, bm25, medcpt)
        results = hybrid.retrieve(query, top_k=3)
        logger.info(f"\n[Hybrid {corpus}] {query}")
        for r in results:
            logger.info(f"  Rank {r['rank']} (RRF={r['score']:.4f}): {r['text'][:80]}...")
    logger.info("SANITY CHECK PASSED")


if __name__ == "__main__":
    sanity_check()
