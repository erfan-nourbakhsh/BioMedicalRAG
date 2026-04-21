import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import argparse
import logging
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.RESULTS_DIR, "run_log.txt")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def build_both_corpora(args):
    from corpora.build_bioasq_corpus import build_bioasq_corpus, sanity_check as bioasq_check
    from corpora.build_yahoo_corpus import build_yahoo_corpus, sanity_check as yahoo_check
    from corpora.build_medical_textbooks_corpus import build_medical_textbooks_corpus, sanity_check as textbooks_check
    from corpora.build_healthcaremagic_corpus import build_healthcaremagic_corpus, sanity_check as healthcaremagic_check
    build_bioasq_corpus(
        force=args.bioasq_force,
        download_only=args.bioasq_download_only,
        max_docs=args.bioasq_max_docs,
        redownload_parquet=args.bioasq_redownload_parquet,
    )
    if not args.bioasq_download_only:
        bioasq_check()
    build_yahoo_corpus()
    yahoo_check()
    build_medical_textbooks_corpus(force=args.new_corpora_force)
    textbooks_check()
    build_healthcaremagic_corpus(force=args.new_corpora_force)
    healthcaremagic_check()


def build_both_indexes(args):
    from retrieval.bm25_indexer import build_indexes, sanity_check
    build_indexes(args.corpus, force=args.bm25_force)
    sanity_check()


def load_both_datasets(args):
    from datasets_prep.load_meqsum import load_meqsum, sanity_check as meqsum_check
    from datasets_prep.load_bioasq_taskb import load_bioasq_taskb, sanity_check as bioasq_taskb_check
    from datasets_prep.load_medredqa import load_medredqa, sanity_check as medredqa_check
    from datasets_prep.load_medquad import load_medquad, sanity_check as medquad_check
    from datasets_prep.load_medicationqa import load_medicationqa, sanity_check as medicationqa_check
    from datasets_prep.load_medqa import load_medqa, sanity_check as medqa_check
    from datasets_prep.load_mashqa import load_mashqa, sanity_check as mashqa_check
    from datasets_prep.load_medmcqa import load_medmcqa, sanity_check as medmcqa_check
    from datasets_prep.load_chatdoctor_icliniq import load_chatdoctor_icliniq, sanity_check as chatdoctor_icliniq_check
    from datasets_prep.load_mmlu_medical import load_mmlu_medical, sanity_check as mmlu_medical_check
    load_meqsum(); meqsum_check()
    load_bioasq_taskb(); bioasq_taskb_check()
    load_medredqa(); medredqa_check()
    load_medquad(); medquad_check()
    load_medicationqa(); medicationqa_check()
    load_medqa(); medqa_check()
    load_mashqa(); mashqa_check()
    load_medmcqa(); medmcqa_check()
    load_chatdoctor_icliniq(); chatdoctor_icliniq_check()
    load_mmlu_medical(); mmlu_medical_check()


def run_all_conditions(args):
    from experiments.run_all_conditions import run_all_conditions as run_exp
    run_exp(args.conditions, model_name=args.generator_model)


def merge_results(args):
    from experiments.run_all_conditions import merge_results as merge_fn
    merge_fn(model_name=args.generator_model)


def run_evaluation(args):
    from evaluation.automated_metrics import run_evaluation as eval_fn, sanity_check
    eval_fn()
    sanity_check()


def main():
    parser = argparse.ArgumentParser(description="Medical RAG Study Pipeline")
    parser.add_argument(
        "--step", type=str, default="all",
        choices=[
            "build_corpora", "build_indexes", "load_datasets",
            "run_experiments", "merge_results", "evaluate", "all",
        ],
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["A", "B", "C", "D", "E", "F"],
    )
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--bioasq-force", action="store_true")
    parser.add_argument("--bioasq-download-only", action="store_true")
    parser.add_argument("--bioasq-redownload-parquet", action="store_true")
    parser.add_argument("--bioasq-max-docs", type=int, default=None)
    parser.add_argument("--bm25-force", action="store_true")
    parser.add_argument("--new-corpora-force", action="store_true")
    parser.add_argument(
        "--corpus", nargs="+",
        default=["bioasq", "yahoo", "medical_textbooks", "healthcaremagic"],
    )
    args = parser.parse_args()

    results_dir = config.get_results_dir(args.generator_model)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(config.get_raw_outputs_dir(args.generator_model), exist_ok=True)
    os.makedirs(config.get_figures_dir(args.generator_model), exist_ok=True)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    logger.addHandler(logging.FileHandler(os.path.join(results_dir, "run_log.txt")))

    steps = {
        "build_corpora": build_both_corpora,
        "build_indexes": build_both_indexes,
        "load_datasets": load_both_datasets,
        "run_experiments": run_all_conditions,
        "merge_results": merge_results,
        "evaluate": run_evaluation,
    }

    logger.info("=" * 60)
    logger.info("Medical RAG Study Pipeline")
    logger.info("=" * 60)

    if args.step == "all":
        total_t0 = time.time()
        for name, fn in steps.items():
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Starting step: {name}")
            logger.info("=" * 50)
            t0 = time.time()
            try:
                fn(args)
            except Exception as e:
                logger.error(f"Step {name} failed: {e}", exc_info=True)
                raise
            logger.info(f"Completed {name} in {time.time() - t0:.1f}s")
        logger.info(f"ALL STEPS COMPLETED in {time.time() - total_t0:.1f}s")
    else:
        t0 = time.time()
        steps[args.step](args)
        logger.info(f"Step {args.step} completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
