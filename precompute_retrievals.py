import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import argparse
import json
import logging
import sys
import time
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import config_extended as ext_config

DEFAULT_SAVE_TOP_K = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "precompute_retrievals.log")),
    ],
)
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval_cache")

DATASET_FILES = {
    "meqsum":             "meqsum_test_queries.csv",
    "bioasq_taskb":       "bioasq_taskb_test_queries.csv",
    "medredqa":           "medredqa_test_queries.csv",
    "medquad":            "medquad_test_queries.csv",
    "medicationqa":       "medicationqa_test_queries.csv",
    "medqa":              "medqa_test_queries.csv",
    "mashqa":             "mashqa_test_queries.csv",
    "medmcqa":            "medmcqa_test_queries.csv",
    "chatdoctor_icliniq": "chatdoctor_icliniq_test_queries.csv",
    "mmlu_medical":       "mmlu_medical_test_queries.csv",
}


def load_datasets():
    datasets = {}
    for name, fname in DATASET_FILES.items():
        path = os.path.join(config.PROCESSED_DIR, fname)
        if os.path.exists(path):
            datasets[name] = pd.read_csv(path)
            logger.info(f"  Loaded {name}: {len(datasets[name])} rows")
        else:
            logger.warning(f"  Dataset not found: {path}")
    return datasets


_retriever_cache = {}

def load_retriever(retriever_type, corpus_name):
    key = (retriever_type, corpus_name)
    if key in _retriever_cache:
        return _retriever_cache[key]

    logger.info(f"Loading retriever: {retriever_type} / {corpus_name} ...")
    t0 = time.time()

    if retriever_type == "bm25":
        from retrieval.bm25_retriever import BM25Retriever
        ret = BM25Retriever(corpus_name)
    elif retriever_type == "tfidf":
        from retrieval.tfidf_retriever import TfidfRetriever
        ret = TfidfRetriever(corpus_name)
    elif retriever_type == "medcpt":
        from retrieval.medcpt_retriever import MedCPTRetriever
        ret = MedCPTRetriever(corpus_name)
    elif retriever_type == "hybrid":
        from retrieval.bm25_retriever import BM25Retriever
        from retrieval.medcpt_retriever import MedCPTRetriever
        from retrieval.hybrid_retriever import HybridRetriever
        bm25   = load_retriever("bm25",   corpus_name)
        medcpt = load_retriever("medcpt", corpus_name)
        ret = HybridRetriever(corpus_name, bm25, medcpt)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    logger.info(f"  Loaded in {time.time() - t0:.1f}s")
    _retriever_cache[key] = ret
    return ret


def retrieve_condition(condition_id, cond_info, queries_df, save_top_k=DEFAULT_SAVE_TOP_K):
    retriever_type = cond_info["retriever"]
    corpus_name    = cond_info["corpus"]
    retriever = load_retriever(retriever_type, corpus_name)

    rows = []
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df),
                       desc=f"  Retrieving {condition_id}"):
        query = str(row["query"])
        retrieved = retriever.retrieve(query, top_k=save_top_k)
        rows.append({
            "query_id":            row["id"],
            "retrieved_docs_json": json.dumps(retrieved),
        })

    return pd.DataFrame(rows)


def run(condition_ids=None, retriever_filter=None, corpus_filter=None,
        save_top_k=DEFAULT_SAVE_TOP_K):
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger.info("Loading all query datasets...")
    datasets = load_datasets()

    all_conditions = ext_config.EXTENDED_CONDITIONS
    if condition_ids:
        target = {cid: all_conditions[cid] for cid in condition_ids
                  if cid in all_conditions}
    else:
        target = {cid: info for cid, info in all_conditions.items()
                  if info["retriever"] != "baseline"}

    if retriever_filter:
        target = {cid: info for cid, info in target.items()
                  if info["retriever"] == retriever_filter}
    if corpus_filter:
        target = {cid: info for cid, info in target.items()
                  if info["corpus"] == corpus_filter}

    groups = defaultdict(list)
    for cid, info in target.items():
        groups[(info["retriever"], info["corpus"])].append(cid)

    total = len(target)
    done = 0
    skipped = 0
    failed = 0

    logger.info(f"Conditions to process: {total}  |  groups: {len(groups)}  |  save_top_k={save_top_k}")

    for (ret_type, corpus), cids in sorted(groups.items()):
        logger.info(f"\nRetriever: {ret_type}  |  Corpus: {corpus}  |  Conditions: {cids}")

        for cid in sorted(cids):
            out_path = os.path.join(CACHE_DIR, f"condition_{cid}.csv")
            cond_info  = all_conditions[cid]
            dataset_name = cond_info["queries"]
            if dataset_name not in datasets:
                logger.warning(f"  [{cid}] Dataset '{dataset_name}' not available — skipping")
                failed += 1
                continue

            queries_df = datasets[dataset_name]
            if os.path.exists(out_path):
                try:
                    existing = pd.read_csv(out_path)
                    if len(existing) >= len(queries_df) and "retrieved_docs_json" in existing.columns:
                        sample = json.loads(existing.iloc[0]["retrieved_docs_json"])
                        if len(sample) >= save_top_k:
                            logger.info(f"  [{cid}] Already complete ({len(existing)} rows, top-{len(sample)}) — skipping")
                            skipped += 1
                            done += 1
                            continue
                except Exception:
                    pass

            logger.info(f"  [{cid}] {cond_info['label']}  ({len(queries_df)} queries, save top-{save_top_k})")
            t0 = time.time()
            try:
                result_df = retrieve_condition(cid, cond_info, queries_df, save_top_k=save_top_k)
                result_df.to_csv(out_path, index=False)
                logger.info(f"  [{cid}] Done: {len(result_df)} rows saved  ({time.time() - t0:.1f}s)")
                done += 1
            except Exception as e:
                logger.error(f"  [{cid}] FAILED: {e}", exc_info=True)
                failed += 1

    logger.info(f"\nFinished.  Done={done}  Skipped={skipped}  Failed={failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute retrieval results for all conditions and save to retrieval_cache/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--retriever", default=None,
                        choices=["bm25", "tfidf", "medcpt", "hybrid"])
    parser.add_argument("--corpus", default=None,
                        choices=["bioasq", "yahoo", "medical_textbooks", "healthcaremagic"])
    parser.add_argument("--save-top-k", type=int, default=DEFAULT_SAVE_TOP_K)
    args = parser.parse_args()

    run(
        condition_ids=args.conditions,
        retriever_filter=args.retriever,
        corpus_filter=args.corpus,
        save_top_k=args.save_top_k,
    )


if __name__ == "__main__":
    main()
