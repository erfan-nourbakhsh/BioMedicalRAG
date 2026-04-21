import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import logging
import pickle
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import config_extended as ext_config
from retrieval.retrieval_utils import build_context_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MedCPTRetriever:

    def __init__(self, corpus_name: str):
        corpus_name = config.normalize_corpus_name(corpus_name)
        index_path = os.path.join(config.INDEX_DIR, f"{corpus_name}_medcpt.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"MedCPT index not found: {index_path}")

        logger.info(f"Loading MedCPT index from {index_path}...")
        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.embeddings = data["embeddings"]
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]
        self.corpus_name = corpus_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading MedCPT query encoder on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(ext_config.MEDCPT_QUERY_ENCODER)
        self.model = AutoModel.from_pretrained(ext_config.MEDCPT_QUERY_ENCODER).to(self.device)
        self.model.eval()
        logger.info(f"MedCPT retriever ready: {len(self.chunks)} docs, dim={self.embeddings.shape[1]}")

    @torch.no_grad()
    def _encode_query(self, query: str) -> np.ndarray:
        encoded = self.tokenizer(
            query, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        q_embed = self.model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
        q_embed = q_embed / np.linalg.norm(q_embed)
        return q_embed

    def retrieve(self, query: str, top_k: int = 5) -> list:
        q_embed = self._encode_query(query)
        scores = (self.embeddings @ q_embed.T).flatten()
        top_indices = scores.argsort()[::-1][:top_k]

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

    def retrieve_with_all_scores(self, query: str) -> np.ndarray:
        q_embed = self._encode_query(query)
        return (self.embeddings @ q_embed.T).flatten()

    def get_context_string(self, query: str, top_k: int = 5, max_chars: int = 1500) -> str:
        return build_context_string(self.retrieve(query, top_k=top_k), max_chars=max_chars)


def sanity_check():
    queries = [
        "What are the symptoms and treatment of diabetes?",
        "How is hypertension diagnosed and managed?",
    ]
    for corpus in ["bioasq", "yahoo"]:
        r = MedCPTRetriever(corpus)
        for q in queries:
            results = r.retrieve(q, top_k=2)
            logger.info(f"\n[MedCPT {corpus}] {q}")
            for res in results:
                logger.info(f"  Rank {res['rank']} ({res['score']:.4f}): {res['text'][:80]}...")
    logger.info("SANITY CHECK PASSED")


if __name__ == "__main__":
    sanity_check()
