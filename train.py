"""
Training module for supervised fine-tuning (SFT) of Qwen3-4B.

Orchestrates model loading, data preparation, and the SFTTrainer loop.
"""

import os
from typing import Optional, Tuple

import torch
from datasets import Dataset
from peft import get_peft_model
from trl import SFTConfig, SFTTrainer

from config import FinetuneConfig
from data_utils import (
    convert_to_chat_format,
    format_chat_template,
    load_dataset_from_hub,
    load_dataset_from_json,
)
from model_utils import (
    adapt_settings_for_device,
    load_tokenizer,
    load_quantized_model,
    print_compute_device,
    print_trainable_params,
    setup_lora_config,
)


def _resolve_training_precision(torch_dtype: str) -> Tuple[bool, bool]:
    """Pick bf16/fp16 flags based on requested dtype and hardware support."""
    if torch_dtype == "float32" or not torch.cuda.is_available():
        return False, False
    if torch_dtype == "float16":
        return False, True
    if torch.cuda.is_bf16_supported():
        return True, False
    return False, True


def _load_and_prepare_data(cfg: FinetuneConfig, tokenizer) -> Dataset:
    """
    Load dataset from hub or local file, convert to ChatML, and apply template.

    Args:
        cfg:       Configuration with dataset_path, dataset_format, etc.
        tokenizer: Tokenizer used to apply the chat template.

    Returns:
        A Dataset with a "text" column containing formatted ChatML strings.
    """
    # 1. Load raw dataset
    if cfg.dataset_path.startswith("/") or cfg.dataset_path.startswith("./") or cfg.dataset_path.startswith("."):
        dataset = load_dataset_from_json(cfg.dataset_path)
    elif cfg.dataset_path == "":
        raise ValueError("dataset_path is required. Provide a Hugging Face path or local file.")
    else:
        dataset = load_dataset_from_hub(cfg.dataset_path)

    # 2. Convert each example to ChatML messages
    def _convert(example):
        messages = convert_to_chat_format(example, cfg.dataset_format)
        text = format_chat_template(messages, tokenizer, cfg.use_thinking)
        return {"text": text}

    dataset = dataset.map(_convert, remove_columns=dataset.column_names)
    return dataset


def _split_dataset(dataset: Dataset, test_split: float):
    """
    Split dataset into train and evaluation subsets.

    Args:
        dataset:    The full dataset.
        test_split: Fraction to reserve for evaluation (0.0 disables eval).

    Returns:
        Tuple of (train_dataset, eval_dataset). eval_dataset is None if test_split is 0.
    """
    if test_split <= 0.0:
        return dataset, None

    split = dataset.train_test_split(test_size=test_split, seed=42)
    return split["train"], split["test"]


def train(cfg: FinetuneConfig) -> None:
    """
    Run the full supervised fine-tuning pipeline.

    Steps:
        1. Load tokenizer and quantized base model.
        2. Wrap the model with LoRA adapters.
        3. Load, convert, and format the dataset.
        4. Split into train / eval sets.
        5. Configure and run SFTTrainer.
        6. Save the trained adapter weights and tokenizer.

    Args:
        cfg: Configuration object with all hyperparameters.
    """
    print("=" * 60)
    print_compute_device()
    cfg.load_in_4bit, cfg.torch_dtype = adapt_settings_for_device(
        cfg.load_in_4bit, cfg.torch_dtype
    )

    print("Step 1/6: Loading tokenizer...")
    tokenizer = load_tokenizer(cfg.model_id)

    print("Step 2/6: Loading quantized base model...")
    model = load_quantized_model(cfg.model_id, cfg.torch_dtype, cfg.load_in_4bit)

    print("Step 3/6: Applying LoRA configuration...")
    lora_config = setup_lora_config(cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.target_modules)
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False  # Required for gradient checkpointing
    model.print_trainable_parameters()
    print_trainable_params(model)

    print("Step 4/6: Loading and formatting dataset...")
    dataset = _load_and_prepare_data(cfg, tokenizer)
    train_dataset, eval_dataset = _split_dataset(dataset, cfg.test_split)
    print(f"  Train examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"  Eval examples:  {len(eval_dataset)}")

    print("Step 5/6: Configuring trainer and starting training...")
    use_bf16, use_fp16 = _resolve_training_precision(cfg.torch_dtype)
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accum_steps,
        warmup_ratio=cfg.warmup_ratio,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=cfg.save_steps if eval_dataset is not None else None,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        remove_unused_columns=False,
        dataloader_num_workers=0 if os.name == "nt" else 2,
        dataset_text_field="text",
        max_length=cfg.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    print("Step 6/6: Saving model and tokenizer...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"  Adapter saved to: {cfg.output_dir}")

    if cfg.push_to_hub:
        if not cfg.hub_model_id:
            raise ValueError("hub_model_id is required when push_to_hub is True.")
        print(f"  Pushing adapter to hub: {cfg.hub_model_id}")
        trainer.model.push_to_hub(cfg.hub_model_id)
        tokenizer.push_to_hub(cfg.hub_model_id)
        print("  Push complete.")

    print("=" * 60)
    print("Training finished successfully.")
