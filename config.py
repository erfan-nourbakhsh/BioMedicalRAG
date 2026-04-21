import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"
import re

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(BASE_DIR, "data", "raw")
BIOASQ_RAW_DIR  = os.path.join(RAW_DIR, "bioasq")
BIOASQ_PARQUET_LINK = os.path.join(BIOASQ_RAW_DIR, "allMeSH_2022.parquet")
PROCESSED_DIR   = os.path.join(BASE_DIR, "data", "processed")
INDEX_DIR       = os.path.join(BASE_DIR, "indexes")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PYSERINI_INPUT_DIR = os.path.join(INDEX_DIR, "pyserini_inputs")

CORPUS_FILES = {
    "bioasq": "bioasq_corpus.jsonl",
    "yahoo": "yahoo_corpus.jsonl",
    "medical_textbooks": "medical_textbooks_corpus.jsonl",
    "healthcaremagic": "healthcaremagic_corpus.jsonl",
}

CORPUS_ALIASES = {
    "medical-textbooks": "medical_textbooks",
    "medical_textbook": "medical_textbooks",
    "textbooks": "medical_textbooks",
    "healthcare-magic": "healthcaremagic",
    "healthcare_magic": "healthcaremagic",
}


def _load_env_file(env_path):
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file(os.path.join(BASE_DIR, ".env"))


def _ensure_java_home():
    if os.environ.get("JAVA_HOME"):
        return
    candidates = [
        "/usr/lib/jvm/java-21-openjdk-21.0.7.0.6-2.el8.x86_64",
        "/usr/lib/jvm/java-21-openjdk",
        "/usr/lib/jvm/default-java",
    ]
    for candidate in candidates:
        java_bin = os.path.join(candidate, "bin", "java")
        if os.path.exists(java_bin):
            os.environ["JAVA_HOME"] = candidate
            path = os.environ.get("PATH", "")
            java_bin_dir = os.path.join(candidate, "bin")
            if java_bin_dir not in path.split(":"):
                os.environ["PATH"] = f"{java_bin_dir}:{path}" if path else java_bin_dir
            return


_ensure_java_home()

if "HF_TOKEN" in os.environ and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

SUPPORTED_GENERATOR_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}
DEFAULT_GENERATOR_MODEL_KEY = "qwen2.5-7b"
GENERATOR_MODEL = SUPPORTED_GENERATOR_MODELS[DEFAULT_GENERATOR_MODEL_KEY]
HF_HOME         = os.environ.get("HF_HOME", "/raid/rsq813/hugging_face")


def normalize_model_key(model_name=None):
    if model_name is None:
        return DEFAULT_GENERATOR_MODEL_KEY
    if model_name in SUPPORTED_GENERATOR_MODELS:
        return model_name
    for key, repo_id in SUPPORTED_GENERATOR_MODELS.items():
        if model_name == repo_id:
            return key
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_name)).strip("-").lower()
    return slug or DEFAULT_GENERATOR_MODEL_KEY


def resolve_generator_model(model_name=None):
    if model_name is None:
        return GENERATOR_MODEL
    return SUPPORTED_GENERATOR_MODELS.get(model_name, model_name)


def get_results_dir(model_name=None):
    model_key = normalize_model_key(model_name)
    return os.path.join(RESULTS_DIR, model_key)


def get_raw_outputs_dir(model_name=None):
    return os.path.join(get_results_dir(model_name), "raw_outputs")


def get_figures_dir(model_name=None):
    return os.path.join(get_results_dir(model_name), "figures")


def get_bm25_index_dir(corpus_name):
    return os.path.join(INDEX_DIR, f"{corpus_name}_bm25_tantivy")


def get_bm25_input_dir(corpus_name):
    return os.path.join(PYSERINI_INPUT_DIR, corpus_name)


def normalize_corpus_name(corpus_name):
    key = str(corpus_name).strip()
    return CORPUS_ALIASES.get(key, key)


def get_corpus_path(corpus_name):
    key = normalize_corpus_name(corpus_name)
    if key not in CORPUS_FILES:
        known = ", ".join(sorted(CORPUS_FILES))
        raise ValueError(f"Unknown corpus '{corpus_name}'. Known corpora: {known}")
    return os.path.join(PROCESSED_DIR, CORPUS_FILES[key])

TOP_K           = 5
MAX_CONTEXT_LEN = 1500

PUBMED_MAX      = 10000
YAHOO_MAX       = 10000

MEQSUM_TEST_SIZE = 500
BIOASQ_TASKB_TEST_SIZE = 200
MEDREDQA_TEST_SIZE = 1000
MEDQUAD_TEST_SIZE = 1000
MEDICATIONQA_TEST_SIZE = 500
MEDQA_TEST_SIZE = 1273
MASHQA_TEST_SIZE = 1000
MEDMCQA_TEST_SIZE = 1000
CHATDOCTOR_ICLINIQ_TEST_SIZE = 1000
MMLU_MEDICAL_TEST_SIZE = 600

MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.1
DO_SAMPLE       = False
GENERATION_BATCH_SIZE = 16

SEED = 42

CONDITIONS = {
    "A": {"system": "baseline", "queries": "meqsum",
          "label": "Baseline + Lay Queries"},
    "B": {"system": "baseline", "queries": "bioasq_taskb",
          "label": "Baseline + Expert Queries"},
    "C": {"system": "bioasq",   "queries": "meqsum",
          "label": "RAG-BioASQ + Lay Queries"},
    "D": {"system": "bioasq",   "queries": "bioasq_taskb",
          "label": "RAG-BioASQ + Expert Queries"},
    "E": {"system": "yahoo",    "queries": "meqsum",
          "label": "RAG-Yahoo + Lay Queries"},
    "F": {"system": "yahoo",    "queries": "bioasq_taskb",
          "label": "RAG-Yahoo + Expert Queries"},
    "A1": {"system": "baseline", "queries": "medredqa",
           "label": "Baseline + MedRedQA Queries"},
    "B1": {"system": "baseline", "queries": "medquad",
           "label": "Baseline + MedQuAD Queries"},
    "C1": {"system": "bioasq",   "queries": "medredqa",
           "label": "RAG-BioASQ + MedRedQA Queries"},
    "D1": {"system": "bioasq",   "queries": "medquad",
           "label": "RAG-BioASQ + MedQuAD Queries"},
    "E1": {"system": "yahoo",    "queries": "medredqa",
           "label": "RAG-Yahoo + MedRedQA Queries"},
    "F1": {"system": "yahoo",    "queries": "medquad",
           "label": "RAG-Yahoo + MedQuAD Queries"},
    "A2": {"system": "baseline", "queries": "medicationqa",
           "label": "Baseline + MedicationQA Queries"},
    "B2": {"system": "baseline", "queries": "medqa",
           "label": "Baseline + MedQA Queries"},
    "C2": {"system": "bioasq",   "queries": "medicationqa",
           "label": "RAG-BioASQ + MedicationQA Queries"},
    "D2": {"system": "bioasq",   "queries": "medqa",
           "label": "RAG-BioASQ + MedQA Queries"},
    "E2": {"system": "yahoo",    "queries": "medicationqa",
           "label": "RAG-Yahoo + MedicationQA Queries"},
    "F2": {"system": "yahoo",    "queries": "medqa",
           "label": "RAG-Yahoo + MedQA Queries"},
    "A3": {"system": "baseline", "queries": "mashqa",
           "label": "Baseline + MASH-QA Queries"},
    "B3": {"system": "baseline", "queries": "medmcqa",
           "label": "Baseline + MedMCQA Queries"},
    "C3": {"system": "bioasq",   "queries": "mashqa",
           "label": "RAG-BioASQ + MASH-QA Queries"},
    "D3": {"system": "bioasq",   "queries": "medmcqa",
           "label": "RAG-BioASQ + MedMCQA Queries"},
    "E3": {"system": "yahoo",    "queries": "mashqa",
           "label": "RAG-Yahoo + MASH-QA Queries"},
    "F3": {"system": "yahoo",    "queries": "medmcqa",
           "label": "RAG-Yahoo + MedMCQA Queries"},
    "A4": {"system": "baseline", "queries": "chatdoctor_icliniq",
           "label": "Baseline + ChatDoctor-iCliniq Queries"},
    "B4": {"system": "baseline", "queries": "mmlu_medical",
           "label": "Baseline + MMLU Medical Queries"},
    "C4": {"system": "bioasq",   "queries": "chatdoctor_icliniq",
           "label": "RAG-BioASQ + ChatDoctor-iCliniq Queries"},
    "D4": {"system": "bioasq",   "queries": "mmlu_medical",
           "label": "RAG-BioASQ + MMLU Medical Queries"},
    "E4": {"system": "yahoo",    "queries": "chatdoctor_icliniq",
           "label": "RAG-Yahoo + ChatDoctor-iCliniq Queries"},
    "F4": {"system": "yahoo",    "queries": "mmlu_medical",
           "label": "RAG-Yahoo + MMLU Medical Queries"},
}
