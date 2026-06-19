import re
import time
import json
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from peft import PeftModel
from litellm import completion
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import (
    LLM_PREDICTIONS_DIR,
    EVAL_METRIC,
    SCORING,
    TARGET
)


load_dotenv(override=True)
LLM_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


class LLMPricePredictor:
    def __init__(
        self,
        model_name: str,
        predictions_dir: Path = LLM_PREDICTIONS_DIR,
        max_words: int = 100,
        checkpoint_every: int = 10,
        sleep_seconds: int = 5,
        max_retries: int = 20,
        base_sleep: int = 2,
        backend: str = 'litellm', #hf
        quant_4_bit: bool = False, # HF 
        hub_model_name: str = None, # HF 
        revision: str = None # HF 
    ):
        self.model_name = model_name
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        self.max_words = int(0.8 * max_words)
        self.tail_words = int(0.2 * max_words)
        self.checkpoint_every = checkpoint_every
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.base_sleep = base_sleep

        self.backend = backend

        # HF
        self.quant_4_bit = quant_4_bit
        self.hub_model_name = hub_model_name
        self.revision = revision

        self.tokenizer = None
        self.fine_tuned_model = None
        self.device = None

    
    def run(self, X_test: pd.DataFrame) -> pd.DataFrame:
        X_test_llm = X_test.copy()
        X_test_llm["catalog_content"] = X_test_llm["catalog_content"].apply(
            self._truncate_text_head_tail
        )

        model_predictions_path = self.predictions_dir / f"{self._safe_filename()}.parquet"

        if model_predictions_path.exists():
            predictions_df = pd.read_parquet(model_predictions_path)
            done_ids = set(
                predictions_df.loc[predictions_df["status"] == "ok", "row_id"].tolist()
            )
        else:
            predictions_df = pd.DataFrame(columns=[
                "row_id",
                "catalog_content",
                "model_name",
                "raw_response",
                "pred_price",
                "pred_log_price",
                "status",
            ])
            done_ids = set()

        
        if self.backend == "hf" and self.fine_tuned_model is None:
            self._load_hf_model()

        rows = []

        for _, row in X_test_llm[~X_test_llm["row_id"].isin(done_ids)].iterrows():
            row_id = row["row_id"]

            try:
                raw_response = self._call_with_retry(row["catalog_content"])
                pred_price = self._parse_price(raw_response)

                if pd.isna(pred_price) or pred_price <= 0:
                    pred_log_price = np.nan
                    status = "bad_parse"
                else:
                    pred_log_price = np.log(pred_price)
                    status = "ok"

                rows.append({
                    "row_id": row_id,
                    "catalog_content": row["catalog_content"],
                    "model_name": self.model_name,
                    "raw_response": raw_response,
                    "pred_price": pred_price,
                    "pred_log_price": pred_log_price,
                    "status": status,
                })

            except Exception as e:
                rows.append({
                    "row_id": row_id,
                    "catalog_content": row["catalog_content"],
                    "model_name": self.model_name,
                    "raw_response": None,
                    "pred_price": np.nan,
                    "pred_log_price": np.nan,
                    "status": f"error: {type(e).__name__}",
                })

            finally:
                time.sleep(self.sleep_seconds)

            if len(rows) % self.checkpoint_every == 0:
                chunk_df = pd.DataFrame(rows)
                predictions_df = pd.concat([predictions_df, chunk_df], ignore_index=True)
                predictions_df.to_parquet(model_predictions_path, index=False)
                rows = []
                print(f"Checkpoint saved at {len(predictions_df)} predictions")

        if rows:
            chunk_df = pd.DataFrame(rows)
            predictions_df = pd.concat([predictions_df, chunk_df], ignore_index=True)
            predictions_df.to_parquet(model_predictions_path, index=False)
            print(f"Saved predictions to {model_predictions_path}")

        return predictions_df

    @staticmethod
    def _messages_for(item: str) -> list[dict]:
        message = f"""
            Estimate the price in USD of the product described below.

            Use the product description and your knowledge of similar products to infer the most likely market price.
            If the description is incomplete or ambiguous, make your best reasonable guess.

            Rules:
            - Return exactly one number.
            - No currency symbol.
            - No commas.
            - No range.
            - No explanation.
            - No additional text.

            Examples:
            Product: Apple AirPods Pro (2nd Generation)
            Answer: 249

            Product: Stainless Steel Water Bottle 1L
            Answer: 19.99

            Product: {item}
            Answer:
        """
        return [{"role": "user", "content": message}]

    @staticmethod
    def _parse_price(response_text):
        if response_text is None:
            return np.nan

        text = str(response_text).strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            return float(match.group(0))

        return np.nan

    @staticmethod
    def _is_rate_limit_error(err: Exception) -> bool:
        msg = str(err).lower()
        return (
            "rate limit" in msg
            or "429" in msg
            or "resource exhausted" in msg
            or "too many requests" in msg
        )

    def _truncate_text_head_tail(self, text):
        words = str(text).split()
        if len(words) <= self.max_words:
            return text

        if self.max_words <= self.tail_words:
            return " ".join(words[: self.max_words])

        head_words = self.max_words - self.tail_words
        return " ".join(words[:head_words] + ["..."] + words[-self.tail_words:])


    def _generate_litellm(self, prompt):
        response = completion(model=self.model_name, messages=self._messages_for(prompt))
        return response.choices[0].message.content
    

    def _generate_hf(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_words)
        inputs = {k: v.to(self.fine_tuned_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.fine_tuned_model.generate(**inputs, max_new_tokens=8)
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


    def _call_with_retry(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                if self.backend == "litellm":
                    raw_response = self._generate_litellm(prompt)
                else:
                    raw_response = self._generate_hf(prompt)
                return raw_response

            except Exception as e:
                if self._is_rate_limit_error(e):
                    wait_s = min(
                        60,
                        self.base_sleep * (2 ** attempt) + random.uniform(0, 2),
                    )
                    print(f"Rate limit hit. Waiting {wait_s:.1f}s and retrying...")
                    time.sleep(wait_s)
                else:
                    raise

        raise RuntimeError("Max retries reached due to repeated rate limits.")

    def _safe_filename(self) -> str:
        file_name = (
            self.model_name
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
        if self.backend == "hf" and self.hub_model_name:
            adapter_name = (
                self.hub_model_name
                .replace("/", "_")
                .replace("-", "_")
                .replace(".", "_")
            )
            file_name = f"{file_name}__{adapter_name}"
        return file_name

    def _load_hf_model(self):

        self.pick_quant_config()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quant_config,
            device_map="cpu",
        )
        base_model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        if self.revision:
            self.fine_tuned_model = PeftModel.from_pretrained(base_model, self.hub_model_name, revision=self.revision)
        else:
            self.fine_tuned_model = PeftModel.from_pretrained(base_model, self.hub_model_name)

        print(f"Memory footprint: {self.fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")


    def pick_quant_config(self):
        if self.quant_4_bit:
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        else:
            self.quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
    


def evaluate_llm_predictions(
    y_test: pd.DataFrame,
    model_name: str,
    model_family: str,
    feature_label: str
):
    model_name_formatted = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
    predictions_df = pd.read_parquet(LLM_PREDICTIONS_DIR / f"{model_name_formatted}.parquet")
    ok_preds = predictions_df.loc[predictions_df["status"] == "ok", ["row_id", "pred_log_price"]].copy()

    merged = y_test.merge(ok_preds, on="row_id", how="inner")

    if merged.empty:
        raise ValueError("No valid predictions were found after merging with y_test.")

    test_score = np.sqrt(EVAL_METRIC(merged[TARGET], merged["pred_log_price"]))

    n_preds = ok_preds.shape[0]
    pct_of_pred = 100 * n_preds / y_test.shape[0]
    params = {
        "n_preds": int(n_preds),
        "pct_of_pred": f"{round(float(pct_of_pred), 2)}%",
        "n_test_rows": int(y_test.shape[0]),
    }

    results = pd.Series({
        "model_name": model_name,
        "model_family": model_family,
        "feature_label": feature_label,
        "target": TARGET,
        "scoring_name": SCORING,
        "params": json.dumps(params, default=str),
        "cv_score_mean": None,
        "cv_score_std": None,
        "train_score": None,
        "test_score": test_score,
    })

    return results
