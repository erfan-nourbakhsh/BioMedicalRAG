import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import json
import logging
import pickle
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import config_extended as ext_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_corpus(corpus_path):
    chunks = []
    with open(corpus_path) as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(corpus_path)}", unit="doc"):
            chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} chunks from {corpus_path}")
    return chunks


def _title_boosted_text(chunk, boost):
    title = chunk.get("title", chunk.get("question_title", ""))
    if boost > 1 and title:
        return (title + " ") * boost + chunk["text"]
    return chunk["text"]


def build_tfidf_index(corpus_path, index_save_path):
    if os.path.exists(index_save_path):
        size_mb = os.path.getsize(index_save_path) / (1024 ** 2)
        logger.info(f"TF-IDF index already exists at {index_save_path} ({size_mb:.1f} MB). Skipping.")
        return

    t0 = time.time()
    chunks = load_corpus(corpus_path)
    boost = ext_config.BM25_TITLE_BOOST
    texts = [
        _title_boosted_text(c, boost)
        for c in tqdm(chunks, desc="Preparing TF-IDF texts", unit="doc")
    ]
    raw_texts = [c["text"] for c in tqdm(chunks, desc="Collecting raw texts", unit="doc")]

    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        sublinear_tf=True,
        dtype="float32",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    metadata = []
    for c in tqdm(chunks, desc="Building metadata", unit="doc"):
        metadata.append({
            "id": c["chunk_id"],
            "source": c.get("pmid", c.get("source_id", "")),
            "title": c.get("title", c.get("question_title", "")),
        })

    data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "chunks": raw_texts,
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(index_save_path), exist_ok=True)
    with open(index_save_path, "wb") as f:
        pickle.dump(data, f)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(index_save_path) / (1024 ** 2)
    logger.info(f"TF-IDF index built: {index_save_path}")
    logger.info(f"  Documents: {tfidf_matrix.shape[0]}, Features: {tfidf_matrix.shape[1]}")
    logger.info(f"  File size: {size_mb:.1f} MB, Time: {elapsed:.1f}s")


def build_all_tfidf_indexes():
    build_tfidf_indexes(["bioasq", "yahoo"])


def build_tfidf_indexes(corpus_names):
    for corpus_name in corpus_names:
        corpus_name = config.normalize_corpus_name(corpus_name)
        build_tfidf_index(
            config.get_corpus_path(corpus_name),
            os.path.join(config.INDEX_DIR, f"{corpus_name}_tfidf.pkl"),
        )


def sanity_check():
    from sklearn.metrics.pairwise import cosine_similarity

    query = "What are the symptoms and treatment of diabetes?"
    for corpus_name in ["bioasq", "yahoo", "medical_textbooks", "healthcaremagic"]:
        path = os.path.join(config.INDEX_DIR, f"{corpus_name}_tfidf.pkl")
        if not os.path.exists(path):
            logger.info(f"Skipping sanity check for missing TF-IDF index: {path}")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        q_vec = data["vectorizer"].transform([query])
        scores = cosine_similarity(q_vec, data["tfidf_matrix"]).flatten()
        top_idx = scores.argsort()[::-1][:3]
        logger.info(f"\nTF-IDF {corpus_name} — query: '{query}'")
        for rank, idx in enumerate(top_idx):
            logger.info(f"  Rank {rank+1} (score={scores[idx]:.4f}): {data['chunks'][idx][:100]}...")
    logger.info("SANITY CHECK PASSED")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build TF-IDF indexes.")
    p.add_argument("--corpus", nargs="+", default=["bioasq", "yahoo"])
    args = p.parse_args()
    build_tfidf_indexes(args.corpus)
    sanity_check()
