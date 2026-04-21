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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MedicalGenerator:

    def __init__(self, model_name: str = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        model_name = config.resolve_generator_model(model_name)
        self.model_name = model_name
        logger.info(f"Loading {model_name}...")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kw = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            props = torch.cuda.get_device_properties(0)
            per_gib = max(1, int(0.90 * props.total_memory / (1024**3)))
            load_kw["max_memory"] = {i: f"{per_gib}GiB" for i in range(torch.cuda.device_count())}
            load_kw["max_memory"]["cpu"] = "0GiB"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
        self.model.eval()

        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def build_prompt(self, query: str, context: str = None,
                     user_type: str = "layperson") -> str:
        if user_type == "layperson":
            system_msg = (
                "You are a helpful health assistant answering "
                "questions from members of the general public. "
                "Use simple, everyday language that a non-medical "
                "person can easily understand. Avoid medical jargon. "
                "Be clear, friendly, and concise."
            )
        else:
            system_msg = (
                "You are a clinical decision support assistant. "
                "Answer questions from healthcare professionals "
                "using precise medical terminology. Provide "
                "evidence-based, clinically detailed responses "
                "with relevant diagnostic and therapeutic "
                "considerations."
            )

        if context:
            user_msg = (
                f"Use the following retrieved passages to help "
                f"answer the question.\n\n"
                f"RETRIEVED CONTEXT:\n{context}\n\n"
                f"QUESTION: {query}\n\n"
                f"ANSWER:"
            )
        else:
            user_msg = f"QUESTION: {query}\n\nANSWER:"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text

    def generate(self, query: str, context: str = None,
                 user_type: str = "layperson") -> dict:
        prompt = self.build_prompt(query, context, user_type)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3000
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=config.DO_SAMPLE,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "query": query,
            "context": context,
            "user_type": user_type,
            "condition": "baseline" if context is None else "rag",
            "response": response,
            "input_len": inputs["input_ids"].shape[1],
            "output_len": len(new_tokens),
        }

    def generate_batch(self, rows: list) -> list:
        if not rows:
            return []

        prompts = [
            self.build_prompt(
                row["query"],
                row.get("context"),
                row.get("user_type", "layperson"),
            )
            for row in rows
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3000,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=config.DO_SAMPLE,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_width = inputs["input_ids"].shape[1]
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        results = []

        for idx, row in enumerate(rows):
            new_tokens = output_ids[idx][prompt_width:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append({
                "query": row["query"],
                "context": row.get("context"),
                "user_type": row.get("user_type", "layperson"),
                "condition": "baseline" if not row.get("context") else "rag",
                "response": response,
                "input_len": int(input_lens[idx]),
                "output_len": int(len(new_tokens)),
            })

        return results

    def batch_generate(self, rows: list, output_path: str) -> pd.DataFrame:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        already_done = set()
        existing_results = []
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            already_done = set(existing_df["query_id"].tolist())
            existing_results = existing_df.to_dict("records")
            logger.info(f"Resuming: {len(already_done)} rows already done")

        new_results = []
        for i, row in enumerate(tqdm(rows, desc="Generating")):
            if row.get("query_id") in already_done:
                continue

            gen_out = self.generate(
                query=row["query"],
                context=row.get("context"),
                user_type=row.get("user_type", "layperson"),
            )

            new_results.append({
                **row,
                "response": gen_out["response"],
                "input_tokens": gen_out["input_len"],
                "output_tokens": gen_out["output_len"],
            })

            if len(new_results) % 10 == 0:
                checkpoint_df = pd.DataFrame(existing_results + new_results)
                checkpoint_df.to_csv(output_path, index=False)
                logger.info(f"Checkpoint saved: {len(existing_results) + len(new_results)} rows")

        all_results = existing_results + new_results
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(output_path, index=False)
        logger.info(f"Final results saved: {len(final_df)} rows to {output_path}")
        return final_df


def sanity_check():
    logger.info("Loading generator for sanity check...")
    gen = MedicalGenerator()

    test_query = "What are the common symptoms of type 2 diabetes?"
    result = gen.generate(test_query, context=None, user_type="layperson")
    logger.info(f"Baseline response ({result['output_len']} tokens): {result['response'][:300]}...")

    test_context = (
        "[Passage 1]: Type 2 diabetes symptoms include increased thirst, "
        "frequent urination, blurred vision, and fatigue."
    )
    result_rag = gen.generate(test_query, context=test_context, user_type="expert")
    logger.info(f"RAG response ({result_rag['output_len']} tokens): {result_rag['response'][:300]}...")

    assert len(result["response"]) > 10, "Response too short"
    assert len(result_rag["response"]) > 10, "RAG response too short"
    logger.info("SANITY CHECK PASSED")


if __name__ == "__main__":
    sanity_check()
