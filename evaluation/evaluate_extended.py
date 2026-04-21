import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import logging
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from evaluation.automated_metrics import evaluate_results, summarize_by_condition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_extended_evaluation(model_name=None):
    results_dir = config.get_results_dir(model_name)
    in_path = os.path.join(config.get_raw_outputs_dir(model_name), "all_conditions_extended.csv")
    out_path = os.path.join(results_dir, "evaluated_results_extended.csv")

    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        if "bertscore_f1" in df.columns and len(df) > 0:
            logger.info(f"Extended evaluated results already exist ({len(df)} rows). Skipping.")
            return df

    if not os.path.exists(in_path):
        logger.error(f"Input not found: {in_path}")
        return None

    df = pd.read_csv(in_path)
    logger.info(f"Evaluating {len(df)} rows from extended conditions...")

    evaluated = evaluate_results(df)
    evaluated.to_csv(out_path, index=False)
    logger.info(f"Extended evaluation saved to {out_path}")

    import config_extended as ext_config
    metric_cols = [c for c in evaluated.columns if any(
        c.startswith(p) for p in ["rouge", "bleu", "meteor", "bertscore", "mcq_accuracy"]
    )]
    rows = []
    for cid in sorted(evaluated["condition_id"].unique()):
        subset = evaluated[evaluated["condition_id"] == cid]
        cinfo = ext_config.EXTENDED_CONDITIONS.get(cid, {})
        row = {"condition_id": cid, "n": len(subset),
               "label": cinfo.get("label", cid),
               "retriever": cinfo.get("retriever", ""),
               "corpus": cinfo.get("corpus", "")}
        for col in metric_cols:
            if col in subset.columns:
                row[f"{col}_mean"] = subset[col].mean()
                row[f"{col}_std"] = subset[col].std()
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    sum_path = os.path.join(results_dir, "condition_summary_extended.csv")
    summary_df.to_csv(sum_path, index=False)
    logger.info(f"Extended summary saved to {sum_path}")

    for _, r in summary_df.iterrows():
        logger.info(f"{r['condition_id']}: {r['label']} (n={r['n']})  "
                     f"ACC={r.get('mcq_accuracy_mean',float('nan')):.4f}  "
                     f"ROUGE-L={r.get('rougeL_f_mean',0):.4f}  "
                     f"BERT={r.get('bertscore_f1_mean',0):.4f}")

    return evaluated


def sanity_check(model_name=None):
    path = os.path.join(config.get_results_dir(model_name), "evaluated_results_extended.csv")
    if not os.path.exists(path):
        logger.error("SANITY CHECK FAILED: evaluated_results_extended.csv not found")
        return
    df = pd.read_csv(path)
    for col in ["rouge1_f", "rougeL_f", "bleu", "meteor", "bertscore_f1", "mcq_accuracy"]:
        assert col in df.columns, f"Missing {col}"
    logger.info(f"Extended evaluation: {len(df)} rows, all metrics present. SANITY CHECK PASSED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    run_extended_evaluation(model_name=args.model)
    sanity_check(model_name=args.model)
