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
import warnings
from hashlib import file_digest
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from platform import python_version
from time import perf_counter

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedTokenizerBase, TrainerCallback, set_seed
from transformers.trainer_callback import TrainerControl, TrainerState
from trl import SFTConfig, SFTTrainer

from device_utils import add_device_argument, parse_4bit, resolve_device
from model_utils import (
    DEFAULT_MODEL_ID,
    fix_ssl_certificates,
    load_base_model,
    load_tokenizer,
)


def _str2bool(value: str) -> bool:
    """Parse a CLI boolean string.

    Args:
        value (str): Boolean value entered on the command line.

    Returns:
        bool: Whether the value is the case-insensitive string "true".
    """
    return value.lower() == "true"


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the training pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Qwen3 supervised fine-tuning (LoRA / QLoRA)")

    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="Hugging Face model ID")
    parser.add_argument("--load_in_4bit", type=parse_4bit,
                        help="Use NF4 QLoRA (enabled by default on NVIDIA CUDA)")
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

    add_device_argument(parser)

    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", "--epochs", dest="num_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", "--batch_size",
                        dest="per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accum_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./models/qwen3-4b-finetuned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient_checkpointing", type=_str2bool, default=True,
        help="Recompute forward activations to save memory",
    )
    parser.add_argument("--group_by_length", action="store_true")
    parser.add_argument("--assistant_only_loss", action="store_true")
    parser.add_argument("--loss_type", choices=["nll", "chunked_nll"])
    parser.add_argument("--eval_each_epoch", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--benchmark_steps", type=int, default=0,
        help="Measure this many optimizer steps after warmup without saving an adapter",
    )
    parser.add_argument("--benchmark_warmup_steps", type=int, default=3)
    parser.add_argument(
        "--run_metrics", type=Path,
        help="Write timing, model revision, software versions and results to JSON",
    )

    parser.add_argument("--push_to_hub", type=_str2bool, default=False)
    parser.add_argument("--hub_model_id", type=str, default="")

    args = parser.parse_args()
    if args.benchmark_steps < 0 or args.benchmark_warmup_steps < 0:
        parser.error("Benchmark step counts must be nonnegative.")
    if not 0 <= args.test_split < 1:
        parser.error("--test_split must be in [0, 1).")
    if not 0 <= args.warmup_ratio <= 1:
        parser.error("--warmup_ratio must be in [0, 1].")
    if min(
        args.per_device_batch_size,
        args.eval_batch_size,
        args.gradient_accum_steps,
        args.max_seq_length,
    ) < 1:
        parser.error("Batch sizes, accumulation and max length must be positive.")
    if args.benchmark_steps and args.push_to_hub:
        parser.error("A short benchmark does not produce an adapter to upload.")
    if args.assistant_only_loss and args.dataset_format != "chat":
        parser.error("--assistant_only_loss requires --dataset_format chat.")
    if args.eval_each_epoch and args.test_split <= 0:
        parser.error("--eval_each_epoch requires a positive --test_split.")
    return args


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


def load_training_datasets(
    args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase,
) -> tuple[Dataset, Dataset | None]:
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

    def format_example(example: dict) -> dict:
        """Render a record or retain the chat for TRL's assistant mask.

        Args:
            example (dict): Raw source record.

        Returns:
            dict: Text or chat fields understood by SFTTrainer.

        Raises:
            ValueError: If the record does not match its declared format.
        """
        messages = _example_to_messages(example, args.dataset_format)
        if args.assistant_only_loss:
            return {
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": args.use_thinking},
            }
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=args.use_thinking,
        )
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    if args.test_split > 0.0:
        split = dataset.train_test_split(test_size=args.test_split, seed=args.seed)
        return split["train"], split["test"]
    return dataset, None


def require_training_device(expected: torch.device, actual: torch.device) -> None:
    """Check that a training component uses the resolved PyTorch device.

    Args:
        expected (torch.device): Device selected by the command line.
        actual (torch.device): Device reported by the training component.

    Returns:
        None.

    Raises:
        RuntimeError: If the devices differ.
    """
    if (expected.type, expected.index or 0) != (actual.type, actual.index or 0):
        raise RuntimeError(
            f"--device selected {expected}, but training resolved to {actual}. "
            "Run in a PyTorch environment configured for the selected backend."
        )


class BenchmarkWindow(TrainerCallback):
    """Measure a fixed window of optimizer steps after warmup."""

    def __init__(self, warmup: int, steps: int, device: torch.device) -> None:
        """Set the measurement boundaries.

        Args:
            warmup (int): Optimizer steps excluded from timing.
            steps (int): Optimizer steps included in timing.
            device (torch.device): Device running the optimizer steps.

        Returns:
            None.
        """
        self.warmup = warmup
        self.steps = steps
        self.device = device
        self.accelerator = getattr(torch, device.type) if device.type != "cpu" else None
        peak_functions = (
            "max_memory_allocated", "max_memory_reserved", "reset_peak_memory_stats",
        )
        self.track_peak_memory = all(
            hasattr(self.accelerator, name) for name in peak_functions
        )
        self.started = 0.0
        self.start_tokens = 0
        self.warmup_peak_allocated = 0
        self.warmup_peak_reserved = 0
        self.result: dict[str, int | float] = {}

    def synchronize(self) -> None:
        """Wait for queued GPU work before measuring elapsed time.

        Returns:
            None.
        """
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.accelerator is not None:
            self.accelerator.synchronize(self.device)

    def on_step_begin(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Start timing after the last warmup step.

        Args:
            args (SFTConfig): Trainer configuration.
            state (TrainerState): Current counters.
            control (TrainerControl): Trainer flow control.
            **kwargs (object): Other callback event data.

        Returns:
            None.
        """
        if state.global_step == self.warmup:
            self.synchronize()
            if self.track_peak_memory:
                self.warmup_peak_allocated = self.accelerator.max_memory_allocated(
                    self.device,
                )
                self.warmup_peak_reserved = self.accelerator.max_memory_reserved(
                    self.device,
                )
                self.accelerator.reset_peak_memory_stats(self.device)
            self.start_tokens = state.num_input_tokens_seen
            self.started = perf_counter()

    def on_step_end(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Record elapsed time, throughput and memory after the final measured step.

        Args:
            args (SFTConfig): Trainer configuration.
            state (TrainerState): Current counters.
            control (TrainerControl): Trainer flow control.
            **kwargs (object): Other callback event data.

        Returns:
            None.
        """
        if self.started and state.global_step == self.warmup + self.steps:
            self.synchronize()
            elapsed = perf_counter() - self.started
            tokens = state.num_input_tokens_seen - self.start_tokens
            self.result = {
                "steps": self.steps,
                "seconds": elapsed,
                "seconds_per_step": elapsed / self.steps,
                "non_padding_tokens": tokens,
                "non_padding_tokens_per_second": tokens / elapsed,
            }
            if self.track_peak_memory:
                self.result.update(
                    peak_allocated_bytes=self.accelerator.max_memory_allocated(
                        self.device,
                    ),
                    peak_reserved_bytes=self.accelerator.max_memory_reserved(
                        self.device,
                    ),
                    warmup_peak_allocated_bytes=self.warmup_peak_allocated,
                    warmup_peak_reserved_bytes=self.warmup_peak_reserved,
                )


def inspect_sequences(
    dataset: Dataset, max_length: int, eos_token_id: int | None, assistant_only: bool,
) -> dict[str, int]:
    """Check sequence lengths and the integrity of assistant labels.

    Args:
        dataset (Dataset): Preprocessed TRL dataset with token IDs.
        max_length (int): Training sequence cap.
        eos_token_id (int | None): End-of-turn token.
        assistant_only (bool): Whether to validate assistant labels.

    Returns:
        dict[str, int]: Count, maximum length and number over the cap.

    Raises:
        ValueError: If any assistant targets are missing, incomplete or truncated.
    """
    lengths = [len(ids) for ids in dataset["input_ids"]]
    truncated = sum(length > max_length for length in lengths)
    if assistant_only:
        for index, row in enumerate(dataset):
            ids = row["input_ids"]
            mask = row.get("assistant_masks", [])
            if (
                len(ids) > max_length
                or len(ids) != len(mask)
                or sum(mask) < 2
                or eos_token_id not in (
                    token for token, keep in zip(ids, mask) if keep
                )
            ):
                raise ValueError(
                    f"Row {index} has no complete assistant target. "
                    "Check the template and increase --max_seq_length if needed."
                )
    return {
        "rows": len(lengths),
        "maximum": max(lengths, default=0),
        "over_cap": truncated,
    }


def runtime_versions() -> dict[str, str | None]:
    """Read the versions of installed training dependencies.

    Returns:
        dict[str, str | None]: Python, CUDA, HIP and package versions, with missing
            optional dependencies shown as None.
    """
    versions = {
        "python": python_version(),
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
    }
    for name in ("torch", "transformers", "peft", "trl", "accelerate",
                 "bitsandbytes", "datasets"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> None:
    """Run SFT or an optimizer-step benchmark and record optional metrics.

    Returns:
        None.

    Raises:
        RuntimeError: If requested timing or training did not complete correctly.
        ValueError: If the dataset or training configuration is invalid.
    """
    started = perf_counter()
    warnings.filterwarnings("ignore", message=".*_check_is_size.*")
    warnings.filterwarnings("ignore", message=".*triton not found.*")
    fix_ssl_certificates(verbose=True)
    args = parse_args()
    set_seed(args.seed)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    hardware = resolve_device(args, training=True)

    print(f"Loading model: {args.model_id}")
    tokenizer = load_tokenizer(args.model_id)
    model = load_base_model(
        args.model_id, hardware.load_in_4bit, hardware.dtype, hardware.device_map,
        prepare_kbit=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model_loaded = perf_counter()
    revision = model.config._commit_hash

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            revision=revision,
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_dataset, eval_dataset = load_training_datasets(args, tokenizer)
    print(f"Train examples: {len(train_dataset)}"
          + (f", eval: {len(eval_dataset)}" if eval_dataset else ""))

    benchmark = args.benchmark_steps > 0
    has_eval = eval_dataset is not None and not benchmark
    save_strategy = "epoch" if args.eval_each_epoch else "steps"
    callbacks = []
    if benchmark:
        window = BenchmarkWindow(
            args.benchmark_warmup_steps, args.benchmark_steps, hardware.device,
        )
        callbacks.append(window)
    trainer_options = {"loss_type": args.loss_type} if args.loss_type else {}
    if hardware.backend in ("mps", "rocm", "xpu"):
        trainer_options["optim"] = "adamw_torch"
    eval_steps = args.save_steps if has_eval and save_strategy == "steps" else None
    sampling = "group_by_length" if args.group_by_length else "random"
    # Accelerate must place batches on the same device as the loaded model.
    os.environ["ACCELERATE_TORCH_DEVICE"] = str(hardware.device)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.benchmark_warmup_steps + args.benchmark_steps
        if benchmark else -1,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accum_steps,
        warmup_steps=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="no" if benchmark else save_strategy,
        eval_strategy=save_strategy if has_eval else "no",
        eval_steps=eval_steps,
        use_cpu=hardware.backend == "cpu",
        bf16=hardware.dtype == torch.bfloat16,
        fp16=hardware.dtype == torch.float16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        remove_unused_columns=False,
        dataloader_num_workers=(
            0 if os.name == "nt" or hardware.backend in ("mps", "xpu") else 2
        ),
        dataloader_pin_memory=hardware.backend != "mps",
        dataset_text_field="text",
        max_length=args.max_seq_length,
        seed=args.seed,
        train_sampling_strategy=sampling,
        assistant_only_loss=args.assistant_only_loss,
        include_num_input_tokens_seen="non_padding" if benchmark else "no",
        **trainer_options,
    )
    require_training_device(hardware.device, training_args.device)
    require_training_device(hardware.device, model.get_input_embeddings().weight.device)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval or args.run_metrics else None,
        callbacks=callbacks,
    )
    require_training_device(hardware.device, trainer.accelerator.device)

    sequences = {}
    if benchmark or args.run_metrics or args.assistant_only_loss:
        for label, dataset in (
            ("train", trainer.train_dataset), ("validation", trainer.eval_dataset),
        ):
            if dataset is not None:
                sequences[label] = inspect_sequences(
                    dataset, args.max_seq_length, tokenizer.eos_token_id,
                    args.assistant_only_loss,
                )
        print(f"Token lengths: {json.dumps(sequences)}")
    ready = perf_counter()
    train_output = trainer.train()
    trained = perf_counter()
    if benchmark:
        if not window.result or not window.result["non_padding_tokens"]:
            raise RuntimeError("The benchmark did not record the requested steps.")
        print(f"Benchmark: {json.dumps(window.result)}")
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Adapter saved to: {args.output_dir}")
    finished = perf_counter()

    if args.run_metrics:
        data_path = Path(args.dataset_path)
        data_hash = None
        if data_path.is_file():
            with data_path.open("rb") as file:
                data_hash = file_digest(file, "sha256").hexdigest()
        result = {
            "arguments": vars(args).copy(),
            "model_revision": revision,
            "attention_implementation": model.config._attn_implementation,
            "dataset_sha256": data_hash,
            "device": hardware.device_name,
            "backend": hardware.backend,
            "dtype": hardware.dtype_name,
            "load_in_4bit": hardware.load_in_4bit,
            "versions": runtime_versions(),
            "sequence_lengths": sequences,
            "model_load_seconds": model_loaded - started,
            "setup_seconds": ready - model_loaded,
            "trainer_seconds": trained - ready,
            "export_seconds": finished - trained,
            "elapsed_seconds": finished - started,
            "train_metrics": {
                key: value if isfinite(value) else None
                for key, value in train_output.metrics.items()
            },
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_loss": trainer.state.best_metric,
            "validation": [
                row for row in trainer.state.log_history if "eval_loss" in row
            ],
        }
        if benchmark:
            result["benchmark"] = window.result
        args.run_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.run_metrics.write_text(
            json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
        )

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id is required when push_to_hub=True.")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print(f"Pushed to hub: {args.hub_model_id}")


if __name__ == "__main__":
    main()
