import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import logging
import sys
import time

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import config_extended as ext_config
from generation.generator import MedicalGenerator
from retrieval.retrieval_utils import build_context_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


RETRIEVAL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "retrieval_cache",
)


def load_retrieval_cache(condition_id, top_k=None):
    import json as _json
    if top_k is None:
        top_k = config.TOP_K

    path = os.path.join(RETRIEVAL_CACHE_DIR, f"condition_{condition_id}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "query_id" not in df.columns:
            return None
        if "retrieved_docs_json" in df.columns:
            result = {}
            for _, row in df.iterrows():
                docs = _json.loads(row["retrieved_docs_json"])
                docs = docs[:top_k]
                context = build_context_string(docs, max_chars=config.MAX_CONTEXT_LEN)
                result[str(row["query_id"])] = context
            logger.info(f"  Cache loaded: {len(result)} queries, top_k={top_k}")
            return result
        elif "context" in df.columns:
            logger.warning(f"  Cache for {condition_id} is legacy format (top_k fixed).")
            return dict(zip(df["query_id"].astype(str), df["context"].fillna("")))
    except Exception as e:
        logger.warning(f"Failed to read retrieval cache for {condition_id}: {e}")
    return None


def get_retriever(retriever_type, corpus_name, _cache={}):
    key = (retriever_type, corpus_name)
    if key in _cache:
        return _cache[key]

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
        bm25 = get_retriever("bm25", corpus_name, _cache)
        medcpt = get_retriever("medcpt", corpus_name, _cache)
        ret = HybridRetriever(corpus_name, bm25, medcpt)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    _cache[key] = ret
    return ret


def run_condition(condition_id, cond_info, queries_df, generator, model_name=None):
    output_dir = config.get_raw_outputs_dir(model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"condition_{condition_id}.csv")

    already_done = set()
    existing_results = []
    if os.path.exists(output_path):
        try:
            done_df = pd.read_csv(output_path)
            if len(done_df) >= len(queries_df):
                logger.info(f"Condition {condition_id}: already complete ({len(done_df)} rows). Skipping.")
                return done_df
            already_done = set(done_df["query_id"].tolist())
            existing_results = done_df.to_dict("records")
            logger.info(f"Condition {condition_id}: resuming, {len(already_done)} done")
        except Exception:
            pass

    retriever_type = cond_info["retriever"]
    corpus_name = cond_info["corpus"]
    user_type = cond_info["user_type"]

    retrieval_cache = None
    if retriever_type != "baseline":
        retrieval_cache = load_retrieval_cache(condition_id)
        if retrieval_cache is not None:
            logger.info(f"  Using pre-computed retrieval cache ({len(retrieval_cache)} entries)")

    retriever = None
    if retriever_type != "baseline" and retrieval_cache is None:
        retriever = get_retriever(retriever_type, corpus_name)

    remaining = len(queries_df) - len(already_done)
    logger.info(f"Running {condition_id}: {cond_info['label']} ({remaining} remaining)")

    new_results = []

    use_two_phase = getattr(generator, "generation_batch_size", 16) >= 500

    if use_two_phase and remaining > 0:
        remaining_df = queries_df[~queries_df["id"].isin(already_done)]

        if retrieval_cache is not None:
            logger.info(f"  [vLLM] Phase 1: loading {remaining} contexts from cache...")
            all_prepared = []
            for _, row in remaining_df.iterrows():
                qid = str(row["id"])
                context = retrieval_cache.get(qid, "") or None
                all_prepared.append({
                    "query_id": row["id"],
                    "query": str(row["query"]),
                    "reference": str(row.get("reference", "")),
                    "user_type": user_type,
                    "context": context,
                    "retrieval_scores": "[]",
                })
        else:
            logger.info(f"  [vLLM] Phase 1: retrieving contexts for {remaining} rows...")
            all_prepared = []
            for _, row in tqdm(remaining_df.iterrows(), total=remaining,
                               desc=f"Retrieving {condition_id}"):
                query = str(row["query"])
                if retriever is not None:
                    retrieved_docs = retriever.retrieve(query, top_k=config.TOP_K)
                    context = build_context_string(retrieved_docs, max_chars=config.MAX_CONTEXT_LEN)
                    top_scores = [d["score"] for d in retrieved_docs]
                else:
                    context = None
                    top_scores = []
                all_prepared.append({
                    "query_id": row["id"],
                    "query": query,
                    "reference": str(row.get("reference", "")),
                    "user_type": user_type,
                    "context": context,
                    "retrieval_scores": str(top_scores),
                })

        gen_subbatch = getattr(generator, "generation_batch_size", 500)
        all_results = list(existing_results)
        logger.info(f"  [vLLM] Phase 2: generating {len(all_prepared)} responses (sub-batch={gen_subbatch})...")
        for start in range(0, len(all_prepared), gen_subbatch):
            sub = all_prepared[start:start + gen_subbatch]
            logger.info(f"  Generating rows {start + 1}–{start + len(sub)} / {len(all_prepared)}...")
            generated = generator.generate_batch(sub)
            for prepared_row, gen_out in zip(sub, generated):
                all_results.append({
                    "condition_id": condition_id,
                    "query_id": prepared_row["query_id"],
                    "query": prepared_row["query"],
                    "reference": prepared_row["reference"],
                    "user_type": user_type,
                    "retriever": retriever_type,
                    "corpus": corpus_name if retriever_type != "baseline" else "none",
                    "context": prepared_row["context"] or "",
                    "response": gen_out["response"],
                    "retrieval_scores": prepared_row["retrieval_scores"],
                    "input_tokens": gen_out["input_len"],
                    "output_tokens": gen_out["output_len"],
                })
            pd.DataFrame(all_results).to_csv(output_path, index=False)
            logger.info(f"  Checkpoint: {len(all_results)} rows saved")

    else:
        flush_batch_size = getattr(generator, "generation_batch_size",
                                   config.GENERATION_BATCH_SIZE)
        pending_rows = []

        def flush_pending():
            nonlocal pending_rows, new_results
            if not pending_rows:
                return
            generated = generator.generate_batch(pending_rows)
            for prepared_row, gen_out in zip(pending_rows, generated):
                new_results.append({
                    "condition_id": condition_id,
                    "query_id": prepared_row["query_id"],
                    "query": prepared_row["query"],
                    "reference": prepared_row["reference"],
                    "user_type": user_type,
                    "retriever": retriever_type,
                    "corpus": corpus_name if retriever_type != "baseline" else "none",
                    "context": prepared_row["context"] or "",
                    "response": gen_out["response"],
                    "retrieval_scores": prepared_row["retrieval_scores"],
                    "input_tokens": gen_out["input_len"],
                    "output_tokens": gen_out["output_len"],
                })
            checkpoint = pd.DataFrame(existing_results + new_results)
            checkpoint.to_csv(output_path, index=False)
            logger.info(f"  Checkpoint: {len(existing_results) + len(new_results)} rows")
            pending_rows = []

        for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"Cond {condition_id}"):
            qid = row["id"]
            if qid in already_done:
                continue
            query = str(row["query"])
            if retrieval_cache is not None:
                context = retrieval_cache.get(str(qid), "") or None
                top_scores = []
            elif retriever is not None:
                retrieved_docs = retriever.retrieve(query, top_k=config.TOP_K)
                context = build_context_string(retrieved_docs, max_chars=config.MAX_CONTEXT_LEN)
                top_scores = [d["score"] for d in retrieved_docs]
            else:
                context = None
                top_scores = []
            pending_rows.append({
                "query_id": qid,
                "query": query,
                "reference": str(row.get("reference", "")),
                "user_type": user_type,
                "context": context,
                "retrieval_scores": str(top_scores),
            })
            if len(pending_rows) >= flush_batch_size:
                flush_pending()
        flush_pending()
        all_results = existing_results + new_results

    if use_two_phase and remaining > 0:
        try:
            final_df = pd.read_csv(output_path)
        except Exception:
            final_df = pd.DataFrame(all_results)
            final_df.to_csv(output_path, index=False)
    else:
        all_results = existing_results + new_results
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(output_path, index=False)
    logger.info(f"Condition {condition_id} done: {len(final_df)} rows -> {output_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_df


def merge_all_conditions(model_name=None):
    raw_dir = config.get_raw_outputs_dir(model_name)
    all_dfs = []
    for cid in sorted(ext_config.EXTENDED_CONDITIONS.keys()):
        cpath = os.path.join(raw_dir, f"condition_{cid}.csv")
        if os.path.exists(cpath):
            df = pd.read_csv(cpath)
            if len(df) > 0:
                all_dfs.append(df)
                logger.info(f"  {cid}: {len(df)} rows")
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(raw_dir, "all_conditions_extended.csv")
        merged.to_csv(out_path, index=False)
        logger.info(f"Merged {len(merged)} rows -> {out_path}")
        return merged
    return pd.DataFrame()


def run_extended_conditions(condition_ids=None, model_name=None):
    if condition_ids is None:
        condition_ids = ext_config.NEW_CONDITION_IDS

    meqsum_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "meqsum_test_queries.csv"))
    bioasq_taskb_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "bioasq_taskb_test_queries.csv"))
    medredqa_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "medredqa_test_queries.csv"))
    medquad_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "medquad_test_queries.csv"))
    medicationqa_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "medicationqa_test_queries.csv"))
    medqa_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "medqa_test_queries.csv"))
    mashqa_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "mashqa_test_queries.csv"))
    medmcqa_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "medmcqa_test_queries.csv"))
    chatdoctor_icliniq_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "chatdoctor_icliniq_test_queries.csv"))
    mmlu_medical_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "mmlu_medical_test_queries.csv"))

    logger.info(f"Loading generator ({model_name or config.GENERATOR_MODEL})...")
    use_vllm = os.environ.get("USE_VLLM", "0") == "1"
    if use_vllm:
        from generation.vllm_generator import VLLMMedicalGenerator
        generator = VLLMMedicalGenerator(model_name=model_name)
    else:
        generator = MedicalGenerator(model_name=model_name)

    query_map = {
        "meqsum": meqsum_df,
        "bioasq_taskb": bioasq_taskb_df,
        "medredqa": medredqa_df,
        "medquad": medquad_df,
        "medicationqa": medicationqa_df,
        "medqa": medqa_df,
        "mashqa": mashqa_df,
        "medmcqa": medmcqa_df,
        "chatdoctor_icliniq": chatdoctor_icliniq_df,
        "mmlu_medical": mmlu_medical_df,
    }

    total_t0 = time.time()
    for cid in condition_ids:
        if cid not in ext_config.EXTENDED_CONDITIONS:
            logger.warning(f"Unknown condition: {cid}")
            continue
        cond_info = ext_config.EXTENDED_CONDITIONS[cid]
        queries_df = query_map[cond_info["queries"]]
        t0 = time.time()
        run_condition(cid, cond_info, queries_df, generator, model_name=model_name)
        logger.info(f"Condition {cid} completed in {time.time() - t0:.1f}s")
    logger.info(f"Total time: {time.time() - total_t0:.1f}s")


def merge_results(model_name=None):
    logger.info("Merging all extended condition outputs...")
    return merge_all_conditions(model_name=model_name)


def sanity_check(model_name=None):
    raw_dir = config.get_raw_outputs_dir(model_name)
    ext_path = os.path.join(raw_dir, "all_conditions_extended.csv")
    if not os.path.exists(ext_path):
        logger.error("SANITY CHECK FAILED: all_conditions_extended.csv not found")
        return
    df = pd.read_csv(ext_path)
    conds = sorted(df["condition_id"].unique())
    logger.info(f"Extended results: {len(df)} rows, conditions: {conds}")
    logger.info("SANITY CHECK PASSED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    run_extended_conditions(args.conditions, model_name=args.model)
    sanity_check(model_name=args.model)
