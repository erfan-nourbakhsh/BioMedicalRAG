import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import argparse
import logging
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import config_extended as ext_config

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "extended_pipeline.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def step_build_indexes(args):
    from retrieval.bm25_indexer import build_both_indexes as build_bm25
    build_bm25()

    from retrieval.tfidf_indexer import build_all_tfidf_indexes
    build_all_tfidf_indexes()

    from retrieval.medcpt_indexer import build_all_medcpt_indexes
    build_all_medcpt_indexes()


def step_run_experiments(args):
    from experiments.run_extended_conditions import run_extended_conditions
    conds = args.conditions if args.conditions else ext_config.NEW_CONDITION_IDS
    logger.info(f"Running conditions: {conds}")
    run_extended_conditions(conds, model_name=args.generator_model)


def step_evaluate(args):
    from evaluation.evaluate_extended import run_extended_evaluation
    run_extended_evaluation(model_name=args.generator_model)


def step_merge_results(args):
    from experiments.run_extended_conditions import merge_results
    merge_results(model_name=args.generator_model)


STEPS = {
    "build_indexes": step_build_indexes,
    "run_experiments": step_run_experiments,
    "merge_results": step_merge_results,
    "evaluate": step_evaluate,
}


def main():
    parser = argparse.ArgumentParser(
        description="Extended RAG Pipeline (TF-IDF, MedCPT, Hybrid)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--step", type=str, default="all",
                        choices=list(STEPS.keys()) + ["all"])
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--generator-model", default=None)
    parser.add_argument(
        "--use-vllm", action="store_true", default=False,
    )
    args = parser.parse_args()

    if args.use_vllm:
        os.environ["USE_VLLM"] = "1"

    os.makedirs(config.get_raw_outputs_dir(args.generator_model), exist_ok=True)
    os.makedirs(config.get_figures_dir(args.generator_model), exist_ok=True)

    if args.step == "all":
        for name, fn in STEPS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"STEP: {name}")
            logger.info(f"{'='*60}")
            t0 = time.time()
            fn(args)
            logger.info(f"Completed {name} in {time.time() - t0:.1f}s")
    else:
        t0 = time.time()
        STEPS[args.step](args)
        logger.info(f"Completed {args.step} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--cuda-devices", default=None, metavar="IDS")
    pre_args, argv_rest = pre.parse_known_args()
    if pre_args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = pre_args.cuda_devices
    sys.argv = [sys.argv[0]] + argv_rest
    main()
