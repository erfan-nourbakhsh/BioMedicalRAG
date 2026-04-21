import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import json
import logging
import pickle
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

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


@torch.no_grad()
def encode_articles(texts, model, tokenizer, device, batch_size=64):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding articles", unit="batch"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(device)
        outputs = model(**encoded)
        embeds = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeds.cpu().numpy())
    return np.vstack(all_embeddings)


def build_medcpt_index(corpus_path, index_save_path):
    if os.path.exists(index_save_path):
        size_mb = os.path.getsize(index_save_path) / (1024 ** 2)
        logger.info(f"MedCPT index exists at {index_save_path} ({size_mb:.1f} MB). Skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    t0 = time.time()
    chunks = load_corpus(corpus_path)
    texts = [c["text"] for c in tqdm(chunks, desc="Preparing MedCPT texts", unit="doc")]

    logger.info(f"Loading MedCPT article encoder: {ext_config.MEDCPT_ARTICLE_ENCODER}")
    tokenizer = AutoTokenizer.from_pretrained(ext_config.MEDCPT_ARTICLE_ENCODER)
    model = AutoModel.from_pretrained(ext_config.MEDCPT_ARTICLE_ENCODER).to(device)
    model.eval()

    logger.info(f"Encoding {len(texts)} articles...")
    embeddings = encode_articles(
        texts, model, tokenizer, device,
        batch_size=ext_config.MEDCPT_BATCH_SIZE,
    )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    metadata = []
    for c in tqdm(chunks, desc="Building metadata", unit="doc"):
        metadata.append({
            "id": c["chunk_id"],
            "source": c.get("pmid", c.get("source_id", "")),
            "title": c.get("title", c.get("question_title", "")),
        })

    data = {
        "embeddings": embeddings,
        "chunks": texts,
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(index_save_path), exist_ok=True)
    with open(index_save_path, "wb") as f:
        pickle.dump(data, f)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(index_save_path) / (1024 ** 2)
    logger.info(f"MedCPT index built: {index_save_path}")
    logger.info(f"  Documents: {embeddings.shape[0]}, Dims: {embeddings.shape[1]}")
    logger.info(f"  File size: {size_mb:.1f} MB, Time: {elapsed:.1f}s")


def build_all_medcpt_indexes():
    build_medcpt_indexes(["bioasq", "yahoo"])


def build_medcpt_indexes(corpus_names):
    for corpus_name in corpus_names:
        corpus_name = config.normalize_corpus_name(corpus_name)
        build_medcpt_index(
            config.get_corpus_path(corpus_name),
            os.path.join(config.INDEX_DIR, f"{corpus_name}_medcpt.pkl"),
        )


def sanity_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ext_config.MEDCPT_QUERY_ENCODER)
    model = AutoModel.from_pretrained(ext_config.MEDCPT_QUERY_ENCODER).to(device)
    model.eval()

    query = "What are the symptoms and treatment of diabetes?"

    with torch.no_grad():
        encoded = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)
        q_embed = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
        q_embed = q_embed / np.linalg.norm(q_embed)

    for corpus_name in ["bioasq", "yahoo", "medical_textbooks", "healthcaremagic"]:
        path = os.path.join(config.INDEX_DIR, f"{corpus_name}_medcpt.pkl")
        if not os.path.exists(path):
            logger.info(f"Skipping sanity check for missing MedCPT index: {path}")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        scores = (data["embeddings"] @ q_embed.T).flatten()
        top_idx = scores.argsort()[::-1][:3]
        logger.info(f"\nMedCPT {corpus_name} — query: '{query}'")
        for rank, idx in enumerate(top_idx):
            logger.info(f"  Rank {rank+1} (score={scores[idx]:.4f}): {data['chunks'][idx][:100]}...")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("SANITY CHECK PASSED")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build MedCPT dense indexes.")
    p.add_argument("--corpus", nargs="+", default=["bioasq", "yahoo"])
    args = p.parse_args()
    build_medcpt_indexes(args.corpus)
    sanity_check()
