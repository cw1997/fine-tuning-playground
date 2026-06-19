#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) script for Qwen3 / Qwen3.5 models.

Usage:
    python src/train.py --dataset_path ./data/ntnu_dataset.jsonl --output_dir ./models/ntnu
    python src/train.py --dataset_path ./data --output_dir ./models/ntnu
    python src/train.py --dataset_path databricks/databricks-dolly-15k --dataset_format alpaca --epochs 5
"""

import argparse
import json
import os
from pathlib import Path

from model_utils import (
    DEFAULT_MODEL_ID,
    fix_ssl_certificates,
    load_base_model,
    load_tokenizer,
    resolve_device,
)

fix_ssl_certificates(verbose=True)

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer


def _str2bool(value: str) -> bool:
    """Parse a CLI boolean string."""
    return value.lower() == "true"


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the training pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Qwen3 supervised fine-tuning (LoRA / QLoRA)")

    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="Hugging Face model ID")
    parser.add_argument("--load_in_4bit", type=_str2bool, default=True,
                        help="Enable 4-bit QLoRA quantization")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Computation precision")

    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated LoRA target modules")

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Local JSON/JSONL file, directory of JSONL files "
                             "(recursive), or Hugging Face dataset name")
    parser.add_argument("--dataset_format", type=str, default="chat",
                        choices=["chat", "alpaca", "text"],
                        help="Dataset format schema")
    parser.add_argument("--test_split", type=float, default=0.05,
                        help="Validation ratio (0 disables evaluation)")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--use_thinking", type=_str2bool, default=False,
                        help="Enable Qwen3 thinking mode in chat template")

    parser.add_argument("--device", type=str, default=None, choices=["gpu", "cpu"],
                        help="Compute device (auto-detected by default)")

    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", "--epochs", dest="num_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", "--batch_size",
                        dest="per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accum_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./models/qwen3-4b-finetuned")

    parser.add_argument("--push_to_hub", type=_str2bool, default=False)
    parser.add_argument("--hub_model_id", type=str, default="")

    return parser.parse_args()


def _read_jsonl_records(file_path: Path) -> list[dict]:
    """Read all records from a JSONL file.

    Args:
        file_path: Path to a local JSONL file.

    Returns:
        List of parsed JSON objects, one per non-empty line.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_local_dataset(dataset_path: Path) -> Dataset:
    """Load a dataset from a local JSON or JSONL file.

    Args:
        dataset_path: Path to a local dataset file.

    Returns:
        Hugging Face Dataset loaded from the file.

    Raises:
        ValueError: If the JSON structure is unsupported.
    """
    if dataset_path.suffix == ".jsonl":
        return Dataset.from_list(_read_jsonl_records(dataset_path))

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return Dataset.from_list(data)
    if isinstance(data, dict):
        return Dataset.from_dict(data)
    raise ValueError(f"Unsupported JSON structure: {dataset_path}")


def _load_jsonl_directory(directory: Path) -> Dataset:
    """Load and merge all JSONL files under a directory recursively.

    Args:
        directory: Root directory to scan for ``*.jsonl`` files.

    Returns:
        Hugging Face Dataset containing all records from matched files.

    Raises:
        ValueError: If no JSONL files are found under the directory.
    """
    jsonl_files = sorted(directory.rglob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found under: {directory}")

    records: list[dict] = []
    for file_path in jsonl_files:
        records.extend(_read_jsonl_records(file_path))

    print(
        f"Loaded {len(records)} examples from {len(jsonl_files)} JSONL file(s) "
        f"under {directory}"
    )
    return Dataset.from_list(records)


def _example_to_messages(example: dict, dataset_format: str) -> list:
    """Convert a raw dataset row into chat messages.

    Args:
        example: One dataset record.
        dataset_format: One of ``chat``, ``alpaca``, or ``text``.

    Returns:
        List of role/content message dicts.

    Raises:
        ValueError: If the format is unsupported or required fields are missing.
    """
    if dataset_format == "chat":
        messages = example.get("messages", [])
        if not messages:
            raise ValueError("chat format requires a non-empty messages field.")
        return messages

    if dataset_format == "alpaca":
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        user_content = f"{instruction}\n\n{inp}".strip() if inp else instruction.strip()
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output.strip()},
        ]

    if dataset_format == "text":
        parts = example.get("text", "").split("<|im_end|>\n<|im_start|>")
        messages = []
        for i, part in enumerate(parts):
            part = part.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": part})
        return messages

    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def load_training_datasets(args: argparse.Namespace, tokenizer) -> tuple:
    """Load, format, and optionally split the training dataset.

    Args:
        args: Parsed CLI arguments.
        tokenizer: Tokenizer used for chat template rendering.

    Returns:
        Tuple of ``(train_dataset, eval_dataset)``; eval is None when no split.
    """
    path = Path(args.dataset_path)
    if path.is_file():
        dataset = _load_local_dataset(path)
    elif path.is_dir():
        dataset = _load_jsonl_directory(path)
    elif path.exists():
        raise ValueError(
            f"dataset_path '{args.dataset_path}' must be a JSON/JSONL file or directory."
        )
    else:
        dataset = load_dataset(args.dataset_path, split="train")

    def format_example(example):
        messages = _example_to_messages(example, args.dataset_format)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=args.use_thinking,
        )
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    if args.test_split > 0.0:
        split = dataset.train_test_split(test_size=args.test_split, seed=42)
        return split["train"], split["test"]
    return dataset, None


def _mixed_precision_flags(dtype_str: str, use_cuda: bool) -> tuple[bool, bool]:
    """Resolve bf16/fp16 flags for SFTConfig.

    Args:
        dtype_str: Precision string from CLI.
        use_cuda: Whether CUDA is available.

    Returns:
        Tuple of ``(use_bf16, use_fp16)``.
    """
    if dtype_str == "float32" or not use_cuda:
        return False, False
    if dtype_str == "float16":
        return False, True
    return (True, False) if torch.cuda.is_bf16_supported() else (False, True)


def main() -> None:
    """Run the full SFT pipeline."""
    args = parse_args()
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    device_map, use_cuda, load_in_4bit, compute_dtype = resolve_device(args)

    print(f"Loading model: {args.model_id}")
    tokenizer = load_tokenizer(args.model_id)
    model = load_base_model(
        args.model_id, load_in_4bit, compute_dtype, device_map, prepare_kbit=True,
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_dataset, eval_dataset = load_training_datasets(args, tokenizer)
    print(f"Train examples: {len(train_dataset)}"
          + (f", eval: {len(eval_dataset)}" if eval_dataset else ""))

    use_bf16, use_fp16 = _mixed_precision_flags(args.torch_dtype, use_cuda)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accum_steps,
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_strategy="steps",
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=args.save_steps if eval_dataset else None,
            use_cpu=not use_cuda,
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            remove_unused_columns=False,
            dataloader_num_workers=0 if os.name == "nt" else 2,
            dataset_text_field="text",
            max_length=args.max_seq_length,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to: {args.output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id is required when push_to_hub=True.")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print(f"Pushed to hub: {args.hub_model_id}")


if __name__ == "__main__":
    main()
