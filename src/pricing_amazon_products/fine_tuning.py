import os
import re
import math
import weave
import numpy as np
from tqdm import tqdm
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed, BitsAndBytesConfig
from datasets import load_dataset, Dataset, DatasetDict
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
from peft import PeftModel
from .config import TARGET, RANDOM_STATE


class QloraTuning:

    def __init__(
        self,
        epochs=2,
        batch_size=32,
        max_sequence_length=128,
        gradient_accumulation_steps=1,
        quant_4_bit=True,
        lora_r=32,
        target_modules=None,
        lora_dropout=0.1,
        learning_rate=1e-3,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        weight_decay=0.001,
        optimizer="paged_adamw_32bit",
        val_size=500,
        log_steps=5,
        save_steps=100,
        log_to_wandb=True,
        project_name=None,
        project_run_name=None,
        run_name=None,
        hub_model_name=None,
        base_model_name=None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.quant_4_bit = quant_4_bit
        self.lora_r = lora_r
        self.lora_alpha = self.lora_r * 2
        self.attention_layers = ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.mlp_layers = ["gate_proj", "up_proj", "down_proj"]
        self.target_modules = target_modules or self.attention_layers
        self.lora_dropout = lora_dropout

        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self.weight_decay = weight_decay
        self.optimizer = optimizer

        self.val_size = val_size
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.log_to_wandb = log_to_wandb
        self.project_name = project_name
        self.project_run_name = project_run_name
        self.run_name = run_name
        self.hub_model_name = hub_model_name
        self.base_model_name = base_model_name

    
    def run_model(
        self,     
        X_train,
        y_train,
        pilot_mode=False,
        private=True
    ):
        self.compose_hf_datasets(    
            X_train,
            y_train,
            pilot_mode
        )
        self.build_tokenizer_and_model()
        self.build_lora_config()
        self.build_training_config()
        self.build_trainer()

        # Fine-tune!
        self.fine_tuning.train()
        
        self.fine_tuning.model.push_to_hub(self.project_run_name, private=private)
        self.tokenizer.push_to_hub(self.project_run_name, private=private)
        print(f"Saved to the hub: {self.project_run_name}")

        if self.log_to_wandb:
            wandb.finish()


    def save_model(self, output_dir=None):
        if output_dir is None:
            output_dir = self.project_run_name

        self.fine_tuning.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Saved locally to: {output_dir}")

        

    def build_quant_config(self):
        if self.quant_4_bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )


    def build_tokenizer_and_model(self):
        quant_config = self.build_quant_config()

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id

        print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")

        self.tokenizer = tokenizer
        self.base_model = base_model

        if self.log_to_wandb:
            wandb.init(project=self.project_name, name=self.project_run_name)


    def build_lora_config(self):
        self.lora_parameters = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.target_modules,
        )

    def build_training_config(self):
        self.train_parameters = SFTConfig(
                output_dir=self.project_run_name,
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                optim=self.optimizer,
                save_steps=self.save_steps,
                save_total_limit=10,
                logging_steps=self.log_steps,
                learning_rate=self.learning_rate,
                weight_decay=0.001,
                fp16=True,
                bf16=False,
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=self.warmup_ratio,
                lr_scheduler_type=self.lr_scheduler_type,
                report_to="wandb" if self.log_to_wandb else None,
                run_name=self.run_name,
                max_length=self.max_sequence_length,
                save_strategy="steps",
                hub_strategy="every_save",
                push_to_hub=True,
                hub_model_id=self.hub_model_name,
                hub_private_repo=True,
                eval_strategy="steps",
                eval_steps=self.save_steps
            )


    def build_trainer(self):
        self.fine_tuning = SFTTrainer(
            model=self.base_model,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            peft_config=self.lora_parameters,
            args=self.train_parameters
        )
    
        
    def compose_hf_datasets(
        self,
        X_train, 
        y_train,
        PILOT_MODE: bool = False,
        pilot_size: int = 1000,
        val_size: float = 0.2
    ):
        X_train_qlora = X_train[:pilot_size].copy() if PILOT_MODE else X_train.copy()
        y_train_qlora = y_train[:pilot_size].copy() if PILOT_MODE else y_train.copy()

        train_llm_df = X_train_qlora.merge(y_train_qlora, on="row_id", how="inner").copy()
        train_llm_df["price"] = np.exp(train_llm_df[TARGET])
        train_llm_df = train_llm_df[["row_id", "catalog_content", "price"]].copy()

        train_df, val_df = train_test_split(
            train_llm_df,
            test_size=val_size,
            random_state=RANDOM_STATE
        )

        train_df = train_df.copy()
        val_df = val_df.copy()

        train_df["prompt"] = train_df["catalog_content"].apply(build_prompt)
        train_df["completion"] = train_df["price"].apply(lambda x: f"{x:.2f}")

        val_df["prompt"] = val_df["catalog_content"].apply(build_prompt)
        val_df["completion"] = val_df["price"].apply(lambda x: f"{x:.2f}")

        self.train_dataset = Dataset.from_pandas(
            train_df[["prompt", "completion"]],
            preserve_index=False
        )

        self.val_dataset = Dataset.from_pandas(
            val_df[["prompt", "completion"]],
            preserve_index=False
        )


def build_prompt(item):
    return f"""Estimate the price in USD of the product described below.

Use the product description and your knowledge of similar products to infer the most likely market price.
If the description is incomplete or ambiguous, make your best reasonable guess.

Return exactly one number with no currency symbol, commas, range, or explanation.

Product:
{item}

Price:
"""