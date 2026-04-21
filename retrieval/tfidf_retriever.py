import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import logging
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from retrieval.retrieval_utils import build_context_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TfidfRetriever:

    def __init__(self, corpus_name: str):
        corpus_name = config.normalize_corpus_name(corpus_name)
        index_path = os.path.join(config.INDEX_DIR, f"{corpus_name}_tfidf.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"TF-IDF index not found: {index_path}")

        logger.info(f"Loading TF-IDF index from {index_path}...")
        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]
        self.corpus_name = corpus_name
        logger.info(f"Loaded {len(self.chunks)} documents from {corpus_name} TF-IDF index")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        q_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "rank": rank + 1,
                "score": float(scores[idx]),
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "title": self.metadata[idx]["title"],
                "corpus": self.corpus_name,
            })
        return results

    def get_context_string(self, query: str, top_k: int = 5, max_chars: int = 1500) -> str:
        return build_context_string(self.retrieve(query, top_k=top_k), max_chars=max_chars)


def sanity_check():
    queries = [
        "What are the symptoms and treatment of diabetes?",
        "How is hypertension diagnosed and managed?",
    ]
    for corpus in ["bioasq", "yahoo"]:
        r = TfidfRetriever(corpus)
        for q in queries:
            results = r.retrieve(q, top_k=2)
            logger.info(f"\n[TF-IDF {corpus}] {q}")
            for res in results:
                logger.info(f"  Rank {res['rank']} ({res['score']:.4f}): {res['text'][:80]}...")
    logger.info("SANITY CHECK PASSED")


if __name__ == "__main__":
    sanity_check()
