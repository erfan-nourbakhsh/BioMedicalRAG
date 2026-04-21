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
from retrieval.bm25_retriever import BM25Retriever
from generation.generator import MedicalGenerator
from retrieval.retrieval_utils import build_context_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_condition(condition_id, queries_df, retriever, generator,
                  user_type, corpus_name, model_name=None):
    output_path = os.path.join(config.get_raw_outputs_dir(model_name),
                               f"condition_{condition_id}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    already_done = set()
    existing_results = []
    if os.path.exists(output_path):
        try:
            done_df = pd.read_csv(output_path)
            if len(done_df) > 0:
                already_done = set(done_df["query_id"].tolist())
                existing_results = done_df.to_dict("records")
                logger.info(f"Condition {condition_id}: resuming, {len(already_done)} already done")
        except Exception:
            pass

    if len(already_done) >= len(queries_df):
        logger.info(f"Condition {condition_id}: all {len(already_done)} rows done. Skipping.")
        return pd.read_csv(output_path)

    new_results = []
    pending_rows = []
    label = config.CONDITIONS[condition_id]["label"]
    logger.info(f"Running Condition {condition_id}: {label} ({len(queries_df) - len(already_done)} queries remaining)")

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
                "corpus": corpus_name,
                "context": prepared_row["context"] or "",
                "response": gen_out["response"],
                "retrieval_scores": prepared_row["retrieval_scores"],
                "input_tokens": gen_out["input_len"],
                "output_tokens": gen_out["output_len"],
            })

            checkpoint = pd.DataFrame(existing_results + new_results)
            checkpoint.to_csv(output_path, index=False)
            logger.info(f"  Checkpoint: {len(existing_results) + len(new_results)} rows saved")

        pending_rows = []

    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc=f"Cond {condition_id}"):
        qid = row["id"]
        if qid in already_done:
            continue

        query = str(row["query"])

        if retriever is not None:
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

        if len(pending_rows) >= config.GENERATION_BATCH_SIZE:
            flush_pending()

    flush_pending()
    all_results = existing_results + new_results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Condition {condition_id} done: {len(final_df)} rows -> {output_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_df


def run_all_conditions(conditions=None, model_name=None):
    if conditions is None:
        conditions = ["A", "B", "C", "D", "E", "F"]

    logger.info("Loading test datasets...")
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

    bioasq_retriever = None
    yahoo_retriever = None
    if any(c in conditions for c in ["C", "D", "C1", "D1", "C2", "D2", "C3", "D3", "C4", "D4"]):
        logger.info("Loading BioASQ BM25 retriever...")
        bioasq_retriever = BM25Retriever("bioasq")
    if any(c in conditions for c in ["E", "F", "E1", "F1", "E2", "F2", "E3", "F3", "E4", "F4"]):
        logger.info("Loading Yahoo BM25 retriever...")
        yahoo_retriever = BM25Retriever("yahoo")

    logger.info(f"Loading generator ({model_name or config.GENERATOR_MODEL})...")
    generator = MedicalGenerator(model_name=model_name)

    condition_map = {
        "A": (meqsum_df, None, "layperson", "baseline"),
        "B": (bioasq_taskb_df, None, "expert", "baseline"),
        "C": (meqsum_df, bioasq_retriever, "layperson", "bioasq"),
        "D": (bioasq_taskb_df, bioasq_retriever, "expert", "bioasq"),
        "E": (meqsum_df, yahoo_retriever, "layperson", "yahoo"),
        "F": (bioasq_taskb_df, yahoo_retriever, "expert", "yahoo"),
        "A1": (medredqa_df, None, "layperson", "baseline"),
        "B1": (medquad_df, None, "expert", "baseline"),
        "C1": (medredqa_df, bioasq_retriever, "layperson", "bioasq"),
        "D1": (medquad_df, bioasq_retriever, "expert", "bioasq"),
        "E1": (medredqa_df, yahoo_retriever, "layperson", "yahoo"),
        "F1": (medquad_df, yahoo_retriever, "expert", "yahoo"),
        "A2": (medicationqa_df, None, "layperson", "baseline"),
        "B2": (medqa_df, None, "expert", "baseline"),
        "C2": (medicationqa_df, bioasq_retriever, "layperson", "bioasq"),
        "D2": (medqa_df, bioasq_retriever, "expert", "bioasq"),
        "E2": (medicationqa_df, yahoo_retriever, "layperson", "yahoo"),
        "F2": (medqa_df, yahoo_retriever, "expert", "yahoo"),
        "A3": (mashqa_df, None, "layperson", "baseline"),
        "B3": (medmcqa_df, None, "expert", "baseline"),
        "C3": (mashqa_df, bioasq_retriever, "layperson", "bioasq"),
        "D3": (medmcqa_df, bioasq_retriever, "expert", "bioasq"),
        "E3": (mashqa_df, yahoo_retriever, "layperson", "yahoo"),
        "F3": (medmcqa_df, yahoo_retriever, "expert", "yahoo"),
        "A4": (chatdoctor_icliniq_df, None, "layperson", "baseline"),
        "B4": (mmlu_medical_df, None, "expert", "baseline"),
        "C4": (chatdoctor_icliniq_df, bioasq_retriever, "layperson", "bioasq"),
        "D4": (mmlu_medical_df, bioasq_retriever, "expert", "bioasq"),
        "E4": (chatdoctor_icliniq_df, yahoo_retriever, "layperson", "yahoo"),
        "F4": (mmlu_medical_df, yahoo_retriever, "expert", "yahoo"),
    }

    results = {}
    total_t0 = time.time()

    for cid in conditions:
        if cid not in condition_map:
            logger.warning(f"Unknown condition: {cid}")
            continue
        queries_df, retriever, user_type, corpus_name = condition_map[cid]
        t0 = time.time()
        results[cid] = run_condition(
            cid, queries_df, retriever, generator, user_type, corpus_name,
            model_name=model_name
        )
        logger.info(f"Condition {cid} completed in {time.time() - t0:.1f}s")

    logger.info(f"Total experiment time: {time.time() - total_t0:.1f}s")
    return results


def merge_results(model_name=None):
    raw_dir = config.get_raw_outputs_dir(model_name)
    all_dfs = []
    for cid in ["A", "B", "C", "D", "E", "F",
                "A1", "B1", "C1", "D1", "E1", "F1",
                "A2", "B2", "C2", "D2", "E2", "F2",
                "A3", "B3", "C3", "D3", "E3", "F3",
                "A4", "B4", "C4", "D4", "E4", "F4"]:
        cpath = os.path.join(raw_dir, f"condition_{cid}.csv")
        if os.path.exists(cpath):
            cdf = pd.read_csv(cpath)
            if len(cdf) > 0:
                all_dfs.append(cdf)
                logger.info(f"  {cid}: {len(cdf)} rows")
    if all_dfs:
        all_results = pd.concat(all_dfs, ignore_index=True)
        all_path = os.path.join(raw_dir, "all_conditions.csv")
        all_results.to_csv(all_path, index=False)
        logger.info(f"Merged {len(all_results)} rows -> {all_path}")
        return all_results
    logger.warning("No condition outputs found to merge.")
    return pd.DataFrame()


def sanity_check(model_name=None):
    all_path = os.path.join(config.get_raw_outputs_dir(model_name), "all_conditions.csv")
    if not os.path.exists(all_path):
        logger.error("SANITY CHECK FAILED: all_conditions.csv not found")
        return
    df = pd.read_csv(all_path)
    assert len(df) > 0, "Empty results"
    assert "response" in df.columns, "Missing response column"
    conditions_present = set(df["condition_id"].unique())
    logger.info(f"Conditions present: {conditions_present}")
    logger.info(f"Total rows: {len(df)}")
    logger.info("SANITY CHECK PASSED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    run_all_conditions(args.conditions, model_name=args.model)
    sanity_check(model_name=args.model)
