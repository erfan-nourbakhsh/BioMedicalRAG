import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import config
import config_extended as ext_config
from retrieval.retrieval_utils import build_context_string

SUBSET_CACHE_DIR = BASE_DIR / "retrieved_subset_cache"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LETTERS = list("ABCDEFGHIJKLMNOPQR")
CONDITION_IDS = (
    LETTERS
    + [f"{letter}1" for letter in LETTERS]
    + [f"{letter}2" for letter in LETTERS]
    + [f"{letter}3" for letter in LETTERS]
    + [f"{letter}4" for letter in LETTERS]
    + [f"{letter}11" for letter in LETTERS]
    + [f"{letter}1N" for letter in LETTERS]
    + [f"{letter}22" for letter in LETTERS]
    + [f"{letter}33" for letter in LETTERS]
    + [f"{letter}44" for letter in LETTERS]
)
DATASETS = [
    "meqsum", "bioasq_taskb", "medredqa", "medquad",
    "medicationqa", "medqa", "mashqa", "medmcqa",
    "chatdoctor_icliniq", "mmlu_medical",
]
SHOT_VALUES = [1, 3, 5, 10]
TRAINING_PATHS = {
    dataset: BASE_DIR / "training" / dataset / f"{dataset}_training_pool.csv"
    for dataset in DATASETS
}


def _output_dir(model_name, shots, seed):
    return os.path.join(
        config.get_results_dir(model_name),
        "ablations", "few-shot", f"shots_{shots}", f"seed_{seed}", "raw_outputs",
    )


def _load_ablation_sets():
    eval_sets = {}
    for dataset in DATASETS:
        path = BASE_DIR / "ablation_subset" / dataset / f"{dataset}_test_queries.csv"
        if not path.exists():
            raise FileNotFoundError(f"Ablation subset not found: {path}")
        eval_sets[dataset] = pd.read_csv(path)
        logger.info("Loaded %s: %s rows", dataset, len(eval_sets[dataset]))
    return eval_sets


def _load_training_pool(dataset):
    path = TRAINING_PATHS[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Training pool not found: {path}")
    return pd.read_csv(path)


def _valid_conditions(condition_ids):
    missing = [cid for cid in condition_ids if cid not in ext_config.EXTENDED_CONDITIONS]
    if missing:
        raise ValueError(f"Unknown conditions: {missing}")
    return list(condition_ids)


def _load_subset_cache(condition_id, top_k=None):
    if top_k is None:
        top_k = config.TOP_K
    path = SUBSET_CACHE_DIR / f"condition_{condition_id}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "query_id" not in df.columns or "retrieved_docs_json" not in df.columns:
            return None
        result = {}
        for _, row in df.iterrows():
            docs    = json.loads(row["retrieved_docs_json"])[:top_k]
            context = build_context_string(docs, max_chars=config.MAX_CONTEXT_LEN)
            result[str(row["query_id"])] = context
        logger.info("  Subset cache hit: %s (top_k=%s, %s entries)", condition_id, top_k, len(result))
        return result
    except Exception as e:
        logger.warning("Failed to read subset cache for %s: %s", condition_id, e)
    return None


_retriever_memory = {}

def _get_retriever(retriever_type, corpus_name):
    key = (retriever_type, corpus_name)
    if key in _retriever_memory:
        return _retriever_memory[key]
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
        bm25   = _get_retriever("bm25",   corpus_name)
        medcpt = _get_retriever("medcpt", corpus_name)
        from retrieval.hybrid_retriever import HybridRetriever
        ret = HybridRetriever(corpus_name, bm25, medcpt)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    _retriever_memory[key] = ret
    return ret


def _sample_shot_rows(dataset, shots, seed, cache):
    key = (dataset, shots, seed)
    if key in cache:
        return cache[key]
    pool_df = _load_training_pool(dataset)
    if len(pool_df) < shots:
        raise ValueError(f"Requested {shots} shots for {dataset}, but only {len(pool_df)} rows exist.")
    sample = pool_df.sample(n=shots, random_state=seed).reset_index(drop=True)
    cache[key] = sample
    return sample


def _build_few_shot_examples(cond_info, retriever, shots, seed, sample_cache):
    dataset = cond_info["queries"]
    sampled = _sample_shot_rows(dataset, shots, seed, sample_cache)
    examples = []
    for _, row in sampled.iterrows():
        query   = str(row["query"])
        context = None
        if retriever is not None:
            docs    = retriever.retrieve(query, top_k=config.TOP_K)
            context = build_context_string(docs, max_chars=config.MAX_CONTEXT_LEN)
        examples.append({
            "example_id": str(row["id"]),
            "query":      query,
            "reference":  str(row.get("reference", "")),
            "context":    context,
            "user_type":  cond_info["user_type"],
        })
    return examples


def _save_shot_manifest(output_dir, condition_id, examples):
    if not examples:
        return
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"condition_{condition_id}_shots.csv")
    pd.DataFrame(examples).to_csv(manifest_path, index=False)
    logger.info("Saved shot manifest -> %s", manifest_path)


def run_condition(condition_id, cond_info, queries_df, generator, shots, seed,
                  sample_cache, model_name=None):
    output_dir  = _output_dir(model_name, shots, seed)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"condition_{condition_id}.csv")

    already_done     = set()
    existing_results = []
    if os.path.exists(output_path):
        done_df = pd.read_csv(output_path)
        if len(done_df) >= len(queries_df):
            logger.info("shots=%s condition %s already complete (%s rows).", shots, condition_id, len(done_df))
            return done_df
        already_done     = set(done_df["query_id"].astype(str).tolist())
        existing_results = done_df.to_dict("records")

    retriever_type = cond_info["retriever"]
    corpus_name    = cond_info["corpus"]
    user_type      = cond_info["user_type"]

    subset_cache = None
    if retriever_type != "baseline":
        subset_cache = _load_subset_cache(condition_id, top_k=config.TOP_K)

    retriever = None
    if retriever_type != "baseline":
        retriever = _get_retriever(retriever_type, corpus_name)

    few_shot_examples = _build_few_shot_examples(cond_info, retriever, shots, seed, sample_cache)
    _save_shot_manifest(output_dir, condition_id, few_shot_examples)
    few_shot_ids = ",".join(ex["example_id"] for ex in few_shot_examples)

    remaining = len(queries_df) - len(already_done)
    logger.info("shots=%s condition %s: %s remaining", shots, condition_id, remaining)

    use_two_phase = getattr(generator, "generation_batch_size", 16) >= 500

    if use_two_phase and remaining > 0:
        remaining_df = queries_df[~queries_df["id"].isin(already_done)]
        all_prepared = []

        for _, row in tqdm(remaining_df.iterrows(), total=remaining,
                           desc=f"shots={shots} Retrieving {condition_id}"):
            query = str(row["query"])
            qid   = str(row["id"])
            if subset_cache is not None:
                context = subset_cache.get(qid); scores = []
            elif retriever is not None:
                docs    = retriever.retrieve(query, top_k=config.TOP_K)
                context = build_context_string(docs, max_chars=config.MAX_CONTEXT_LEN)
                scores  = [d["score"] for d in docs]
            else:
                context = None; scores = []
            all_prepared.append({
                "query_id": row["id"], "query": query, "reference": str(row.get("reference", "")),
                "user_type": user_type, "context": context, "retrieval_scores": str(scores),
                "few_shot_examples": few_shot_examples,
            })

        gen_subbatch = getattr(generator, "generation_batch_size", 500)
        all_results  = list(existing_results)
        for start in range(0, len(all_prepared), gen_subbatch):
            sub = all_prepared[start:start + gen_subbatch]
            generated = generator.generate_batch(sub)
            for prep, gen_out in zip(sub, generated):
                all_results.append({
                    "condition_id": condition_id, "query_id": prep["query_id"],
                    "query": prep["query"], "reference": prep["reference"],
                    "user_type": user_type, "retriever": retriever_type,
                    "corpus": corpus_name if retriever_type != "baseline" else "none",
                    "few_shot_count": shots, "few_shot_seed": seed,
                    "few_shot_example_ids": few_shot_ids,
                    "context": prep["context"] or "", "response": gen_out["response"],
                    "retrieval_scores": prep["retrieval_scores"],
                    "input_tokens": gen_out["input_len"], "output_tokens": gen_out["output_len"],
                })
            pd.DataFrame(all_results).to_csv(output_path, index=False)
            logger.info("  Checkpoint: %s rows saved", len(all_results))

    else:
        import torch
        new_results  = []
        pending_rows = []
        flush_batch  = getattr(generator, "generation_batch_size", config.GENERATION_BATCH_SIZE)

        def flush_pending(force=False):
            nonlocal pending_rows, new_results
            if not pending_rows:
                return
            generated = generator.generate_batch(pending_rows)
            for prep, gen_out in zip(pending_rows, generated):
                new_results.append({
                    "condition_id": condition_id, "query_id": prep["query_id"],
                    "query": prep["query"], "reference": prep["reference"],
                    "user_type": user_type, "retriever": retriever_type,
                    "corpus": corpus_name if retriever_type != "baseline" else "none",
                    "few_shot_count": shots, "few_shot_seed": seed,
                    "few_shot_example_ids": few_shot_ids,
                    "context": prep["context"] or "", "response": gen_out["response"],
                    "retrieval_scores": prep["retrieval_scores"],
                    "input_tokens": gen_out["input_len"], "output_tokens": gen_out["output_len"],
                })
            pending_rows.clear()
            if force or len(new_results) % 10 == 0:
                pd.DataFrame(existing_results + new_results).to_csv(output_path, index=False)

        for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"shots={shots} {condition_id}"):
            qid = str(row["id"])
            if qid in already_done:
                continue
            query = str(row["query"])
            if subset_cache is not None:
                context = subset_cache.get(qid); scores = []
            elif retriever is not None:
                docs    = retriever.retrieve(query, top_k=config.TOP_K)
                context = build_context_string(docs, max_chars=config.MAX_CONTEXT_LEN)
                scores  = [d["score"] for d in docs]
            else:
                context = None; scores = []
            pending_rows.append({
                "query_id": qid, "query": query, "reference": str(row.get("reference", "")),
                "user_type": user_type, "context": context, "retrieval_scores": str(scores),
                "few_shot_examples": few_shot_examples,
            })
            if len(pending_rows) >= flush_batch:
                flush_pending()

        flush_pending(force=True)
        all_results = existing_results + new_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(output_path, index=False)
    logger.info("shots=%s condition %s complete -> %s", shots, condition_id, output_path)
    return final_df


def merge_results(model_name=None, shots=None, seed=config.SEED):
    shot_values = [shots] if shots is not None else SHOT_VALUES
    for shot_count in shot_values:
        raw_dir = _output_dir(model_name, shot_count, seed)
        all_dfs = []
        for cid in CONDITION_IDS:
            path = os.path.join(raw_dir, f"condition_{cid}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                if len(df) > 0:
                    all_dfs.append(df)
        if not all_dfs:
            logger.warning("No outputs found for model=%s shots=%s seed=%s", model_name, shot_count, seed)
            continue
        merged   = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(raw_dir, "all_conditions_extended.csv")
        merged.to_csv(out_path, index=False)
        logger.info("Merged %s rows -> %s", len(merged), out_path)


def run_few_shot_ablation(model_name=None, condition_ids=None, shot_values=None,
                          seed=config.SEED, use_vllm=False):
    condition_ids = _valid_conditions(condition_ids or CONDITION_IDS)
    shot_values   = shot_values or SHOT_VALUES
    eval_sets     = _load_ablation_sets()
    sample_cache  = {}

    logger.info("Loading generator (model=%s, vllm=%s)...",
                model_name or config.DEFAULT_GENERATOR_MODEL_KEY, use_vllm)
    if use_vllm:
        from generation.vllm_generator import VLLMMedicalGenerator
        generator = VLLMMedicalGenerator(model_name=model_name)
    else:
        from generation.generator import MedicalGenerator
        generator = MedicalGenerator(model_name=model_name)

    total_t0 = time.time()
    for shots in shot_values:
        logger.info("=" * 60)
        logger.info("Running shots=%s", shots)
        logger.info("=" * 60)
        for cid in condition_ids:
            cond_info  = ext_config.EXTENDED_CONDITIONS[cid]
            queries_df = eval_sets[cond_info["queries"]]
            t0 = time.time()
            run_condition(cid, cond_info, queries_df, generator, shots, seed,
                          sample_cache, model_name=model_name)
            logger.info("shots=%s condition %s completed in %.1fs", shots, cid, time.time() - t0)
        merge_results(model_name=model_name, shots=shots, seed=seed)
    logger.info("few-shot ablation completed in %.1fs", time.time() - total_t0)


def main():
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--cuda-devices", default=None)
    _pre_args, _rest = _pre.parse_known_args()
    if _pre_args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.cuda_devices
    sys.argv = [sys.argv[0]] + _rest

    parser = argparse.ArgumentParser(description="Run few-shot ablations on ablation_subset.")
    parser.add_argument("--step", choices=["run_experiments", "merge_results"], default="run_experiments")
    parser.add_argument("--models", nargs="+", default=[config.DEFAULT_GENERATOR_MODEL_KEY])
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=SHOT_VALUES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--use-vllm", action="store_true", default=False)
    parser.add_argument("--cuda-devices", default=None)
    args = parser.parse_args()

    condition_ids = _valid_conditions(args.conditions or CONDITION_IDS)
    for model_name in args.models:
        if args.step == "run_experiments":
            run_few_shot_ablation(
                model_name=model_name,
                condition_ids=condition_ids,
                shot_values=args.shots,
                seed=args.seed,
                use_vllm=args.use_vllm,
            )
        else:
            for shots in args.shots:
                merge_results(model_name=model_name, shots=shots, seed=args.seed)


if __name__ == "__main__":
    main()
