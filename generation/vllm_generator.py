import os
os.environ["HF_HOME"] = "/raid/rsq813/hugging_face"

import logging
import sys
import time

import pandas as pd
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VLLMMedicalGenerator:

    def __init__(self, model_name: str = None):
        from vllm import LLM, SamplingParams

        model_name = config.resolve_generator_model(model_name)
        self.model_name = model_name

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("vLLM requires at least one CUDA GPU.")
        logger.info(f"Loading {model_name} with vLLM (tensor_parallel_size={n_gpus})...")
        t0 = time.time()

        max_model_len = 3300

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=n_gpus,
            dtype="float16",
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            max_model_len=max_model_len,
            download_dir=os.environ.get("HF_HOME", "/raid/rsq813/hugging_face"),
            enable_prefix_caching=True,
        )

        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"vLLM model loaded in {time.time() - t0:.1f}s")

        self.generation_batch_size = 500
        self.checkpoint_every = 500

        if config.DO_SAMPLE:
            self.sampling_params = SamplingParams(
                max_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=1.0,
            )
        else:
            self.sampling_params = SamplingParams(
                max_tokens=config.MAX_NEW_TOKENS,
                temperature=0.0,
                top_p=1.0,
            )

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
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text.strip()
        input_len = len(outputs[0].prompt_token_ids)
        output_len = len(outputs[0].outputs[0].token_ids)

        return {
            "query": query,
            "context": context,
            "user_type": user_type,
            "condition": "baseline" if context is None else "rag",
            "response": response,
            "input_len": input_len,
            "output_len": output_len,
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

        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)

        results = []
        for row, out in zip(rows, outputs):
            response = out.outputs[0].text.strip()
            results.append({
                "query": row["query"],
                "context": row.get("context"),
                "user_type": row.get("user_type", "layperson"),
                "condition": "baseline" if not row.get("context") else "rag",
                "response": response,
                "input_len": len(out.prompt_token_ids),
                "output_len": len(out.outputs[0].token_ids),
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

        remaining = [r for r in rows if r.get("query_id") not in already_done]
        if not remaining:
            return pd.DataFrame(existing_results)

        new_results = []
        for i, row in enumerate(tqdm(remaining, desc="Generating (vLLM)")):
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
