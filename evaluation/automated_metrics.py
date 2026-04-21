import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import logging
import sys
import time
import json
import re

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _reference_candidates(ref):
    if isinstance(ref, list):
        return [str(r).strip() for r in ref if str(r).strip()]
    if isinstance(ref, str):
        text = ref.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(r).strip() for r in parsed if str(r).strip()]
            except Exception:
                pass
        return [text]
    return []


def _normalize_eval_text(text):
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    return normalized.strip()


def _parse_mcq_options(query):
    options = {}
    if not isinstance(query, str):
        return options

    pattern = re.compile(
        r"^\(([A-E])\)\s*(.+?)(?=\n\([A-E]\)\s*|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(query):
        label = match.group(1).upper()
        text = " ".join(match.group(2).split()).strip()
        if text:
            options[label] = text
    return options


def _extract_mcq_prediction(response, options):
    if not isinstance(response, str):
        return None, None

    compact = " ".join(response.split())
    for pattern in [
        r"<answer>\s*([A-E])\s*</answer>",
        r"the answer is\s*\[([A-E])\]",
        r"the answer is\s*\(([A-E])\)",
        r"the answer is\s*([A-E])\b",
        r"(?:answer|short answer):\s*\(([A-E])\)",
        r"(?:answer|short answer):\s*\[?([A-E])\]?",
        r"\*\*answer[:\*]*\s*\[?([A-E])\]?",
        r"(?:^|\n)\s*\(?([A-E])\)?\s*$",
        r"option\s+([A-E])\b",
        r"(?:choose|select|pick)\s+([A-E])\b",
        r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-E])\)?\b",
        r"\b\(?([A-E])\)?\s+is\s+correct\b",
        r"^\s*\(?([A-E])\)?[\).\:\-]?\s",
        r"\(([A-E])\)",
    ]:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            label = match.group(1).upper()
            return label, options.get(label)

    normalized_response = _normalize_eval_text(compact)
    if not normalized_response:
        return None, None

    for label, option_text in options.items():
        normalized_option = _normalize_eval_text(option_text)
        if normalized_option and (
            normalized_option in normalized_response or normalized_response in normalized_option
        ):
            return label, option_text

    return None, None


def compute_medqa_accuracy(queries, responses, references):
    accuracy = []
    predicted_option = []
    predicted_text = []

    for query, response, reference in zip(queries, responses, references):
        options = _parse_mcq_options(query)
        if not options:
            accuracy.append(np.nan)
            predicted_option.append("")
            predicted_text.append("")
            continue

        pred_label, pred_text = _extract_mcq_prediction(response, options)
        predicted_option.append(pred_label or "")
        predicted_text.append(pred_text or "")

        ref_norm = _normalize_eval_text(reference)
        resp_norm = _normalize_eval_text(response)
        pred_norm = _normalize_eval_text(pred_text)

        is_correct = False
        if ref_norm:
            is_correct = (
                pred_norm == ref_norm
                or (pred_norm and pred_norm in ref_norm)
                or (ref_norm and ref_norm in resp_norm)
            )
        accuracy.append(float(is_correct))

    return accuracy, predicted_option, predicted_text


def compute_rouge(responses, references):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    r1, r2, rl = [], [], []
    for resp, ref in tqdm(zip(responses, references), total=len(responses), desc="ROUGE"):
        if not isinstance(resp, str):
            r1.append(0.0); r2.append(0.0); rl.append(0.0)
            continue
        candidates = _reference_candidates(ref)
        if not candidates:
            r1.append(0.0); r2.append(0.0); rl.append(0.0)
            continue
        best_r1 = best_r2 = best_rl = 0.0
        for candidate in candidates:
            scores = scorer.score(candidate, resp)
            best_r1 = max(best_r1, scores["rouge1"].fmeasure)
            best_r2 = max(best_r2, scores["rouge2"].fmeasure)
            best_rl = max(best_rl, scores["rougeL"].fmeasure)
        r1.append(best_r1)
        r2.append(best_r2)
        rl.append(best_rl)
    return r1, r2, rl


def compute_bleu(responses, references):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoothing = SmoothingFunction().method1
    bleu_scores = []
    for resp, ref in tqdm(zip(responses, references), total=len(responses), desc="BLEU"):
        if not isinstance(resp, str) or not resp.strip():
            bleu_scores.append(0.0)
            continue
        candidates = _reference_candidates(ref)
        if not candidates:
            bleu_scores.append(0.0)
            continue

        response_tokens = resp.split()
        reference_tokens = [candidate.split() for candidate in candidates if candidate.split()]
        if not response_tokens or not reference_tokens:
            bleu_scores.append(0.0)
            continue

        bleu_scores.append(
            sentence_bleu(reference_tokens, response_tokens, smoothing_function=smoothing)
        )
    return bleu_scores


def compute_meteor(responses, references):
    from nltk.translate.meteor_score import meteor_score
    import nltk

    meteor_scores = []
    resources_ready = False
    for resp, ref in tqdm(zip(responses, references), total=len(responses), desc="METEOR"):
        if not isinstance(resp, str) or not resp.strip():
            meteor_scores.append(0.0)
            continue
        candidates = _reference_candidates(ref)
        if not candidates:
            meteor_scores.append(0.0)
            continue

        response_tokens = resp.split()
        if not response_tokens:
            meteor_scores.append(0.0)
            continue

        best_score = 0.0
        for candidate in candidates:
            reference_tokens = candidate.split()
            if not reference_tokens:
                continue
            try:
                best_score = max(best_score, meteor_score([reference_tokens], response_tokens))
            except LookupError as exc:
                if not resources_ready:
                    nltk.download("wordnet", quiet=True)
                    nltk.download("omw-1.4", quiet=True)
                    resources_ready = True
                    try:
                        best_score = max(best_score, meteor_score([reference_tokens], response_tokens))
                        continue
                    except LookupError:
                        best_score = 0.0
                        break
        meteor_scores.append(best_score)
    return meteor_scores


def compute_bertscore(responses, references, batch_size=32):
    from bert_score import score as bert_score_fn
    import torch

    valid_responses = []
    valid_references = []
    valid_indices = []
    for i, (resp, ref) in enumerate(zip(responses, references)):
        if not isinstance(resp, str) or not resp.strip():
            continue
        candidates = _reference_candidates(ref)
        for candidate in candidates:
            valid_responses.append(resp)
            valid_references.append(candidate)
            valid_indices.append(i)

    precision = [0.0] * len(responses)
    recall = [0.0] * len(responses)
    f1 = [0.0] * len(responses)

    if not valid_responses:
        return precision, recall, f1

    logger.info(f"Computing BERTScore for {len(valid_responses)} valid pairs...")
    P, R, F = bert_score_fn(
        valid_responses, valid_references,
        model_type="bert-base-uncased",
        batch_size=batch_size,
        verbose=True,
    )

    for j, idx in enumerate(valid_indices):
        precision[idx] = max(precision[idx], float(P[j]))
        recall[idx] = max(recall[idx], float(R[j]))
        f1[idx] = max(f1[idx], float(F[j]))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return precision, recall, f1



def evaluate_results(results_df):
    logger.info(f"Evaluating {len(results_df)} responses...")
    t0 = time.time()

    responses = results_df["response"].tolist()
    references = results_df["reference"].tolist()

    logger.info("Computing ROUGE scores...")
    r1, r2, rl = compute_rouge(responses, references)
    results_df["rouge1_f"] = r1
    results_df["rouge2_f"] = r2
    results_df["rougeL_f"] = rl

    logger.info("Computing BLEU score...")
    results_df["bleu"] = compute_bleu(responses, references)

    logger.info("Computing METEOR score...")
    results_df["meteor"] = compute_meteor(responses, references)

    logger.info("Computing BERTScore...")
    bp, br, bf = compute_bertscore(responses, references)
    results_df["bertscore_precision"] = bp
    results_df["bertscore_recall"] = br
    results_df["bertscore_f1"] = bf

    logger.info("Computing MCQ accuracy...")
    mcq_acc, mcq_pred_option, mcq_pred_text = compute_medqa_accuracy(
        results_df["query"].tolist(), responses, references,
    )
    results_df["mcq_accuracy"] = mcq_acc
    results_df["mcq_predicted_option"] = mcq_pred_option
    results_df["mcq_predicted_text"] = mcq_pred_text

    out_path = os.path.join(config.RESULTS_DIR, "evaluated_results.csv")
    results_df.to_csv(out_path, index=False)
    logger.info(f"Evaluation complete in {time.time() - t0:.1f}s. Saved to {out_path}")

    return results_df


def summarize_by_condition(evaluated_df):
    metric_cols = [
        "rouge1_f", "rouge2_f", "rougeL_f", "bleu", "meteor",
        "bertscore_precision", "bertscore_recall", "bertscore_f1",
        "mcq_accuracy",
    ]

    rows = []
    for cid in sorted(evaluated_df["condition_id"].unique()):
        subset = evaluated_df[evaluated_df["condition_id"] == cid]
        row = {"condition_id": cid, "n": len(subset)}
        label = config.CONDITIONS.get(cid, {}).get("label", cid)
        row["label"] = label
        for col in metric_cols:
            if col in subset.columns:
                row[f"{col}_mean"] = subset[col].mean()
                row[f"{col}_std"] = subset[col].std()
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(config.RESULTS_DIR, "condition_summary.csv")
    summary_df.to_csv(out_path, index=False)
    logger.info(f"Condition summary saved to {out_path}")
    return summary_df


def sanity_check():
    path = os.path.join(config.RESULTS_DIR, "evaluated_results.csv")
    if not os.path.exists(path):
        logger.error("SANITY CHECK FAILED: evaluated_results.csv not found")
        return
    df = pd.read_csv(path)
    required = ["rouge1_f", "rougeL_f", "bleu", "meteor", "bertscore_f1", "mcq_accuracy"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    logger.info(f"Evaluated results: {len(df)} rows, all metric columns present")
    logger.info("SANITY CHECK PASSED")


def run_evaluation():
    all_path = os.path.join(config.RESULTS_DIR, "raw_outputs", "all_conditions.csv")
    eval_path = os.path.join(config.RESULTS_DIR, "evaluated_results.csv")

    if os.path.exists(eval_path):
        df = pd.read_csv(eval_path)
        if "bertscore_f1" in df.columns and len(df) > 0:
            logger.info(f"Evaluated results already exist ({len(df)} rows). Skipping.")
            return df

    df = pd.read_csv(all_path)
    evaluated = evaluate_results(df)
    summarize_by_condition(evaluated)
    return evaluated


if __name__ == "__main__":
    run_evaluation()
    sanity_check()
