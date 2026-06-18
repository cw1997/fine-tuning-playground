#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) script for Qwen3 / Qwen3.5 models.

Executes sequentially from top to bottom for clear demonstration of the
fine-tuning pipeline. All configuration is passed via CLI arguments.

Usage:
    python src/train.py --dataset_path ./data/ntnu_dataset.jsonl --output_dir ./models/ntnu-finetuned
    python src/train.py --dataset_path databricks/databricks-dolly-15k --dataset_format alpaca --epochs 5
"""

import os
import shutil
import warnings

import certifi

# Fix SSL certificate path issues commonly found on Windows / conda environments
for _ssl_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    _ssl_val = os.environ.get(_ssl_var)
    if _ssl_val and not os.path.isfile(_ssl_val):
        os.environ[_ssl_var] = certifi.where()
        print(f"  Note: {_ssl_var} pointed to missing file; using certifi bundle.")

# Suppress FutureWarning from torch._check_is_size (to be removed in future PyTorch)
warnings.filterwarnings("ignore", message=".*_check_is_size.*")
# Suppress Triton FLOP counter warning on systems without Triton
warnings.filterwarnings("ignore", message=".*triton not found.*")

import argparse
import json

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the training pipeline.

    All hyperparameters are exposed as CLI flags — no separate config file needed.
    Covers model selection, LoRA configuration, dataset options, hardware settings,
    training parameters, and Hugging Face Hub publishing.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Qwen3 supervised fine-tuning (LoRA / QLoRA)")

    # Model
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-0.8B",
                        help="Hugging Face model ID (GPU recommended: Qwen/Qwen3.5-4B)")
    parser.add_argument("--load_in_4bit", type=lambda x: x.lower() == "true", default=True,
                        help="Enable 4-bit QLoRA quantization")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Computation precision (bfloat16 recommended on Ampere+ GPUs)")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated LoRA target module names")

    # Data
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Local JSON/JSONL path or Hugging Face dataset name")
    parser.add_argument("--dataset_format", type=str, default="chat",
                        choices=["chat", "alpaca", "text"],
                        help="Dataset format schema")
    parser.add_argument("--test_split", type=float, default=0.05,
                        help="Validation set ratio (0 disables evaluation)")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Maximum sequence length in tokens")
    parser.add_argument("--use_thinking", type=lambda x: x.lower() == "true", default=False,
                        help="Enable Qwen3 thinking mode in the chat template")

    # Hardware
    parser.add_argument("--device", type=str, default=None, choices=["gpu", "cpu"],
                        help="Compute device (auto-detected by default)")

    # Training
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", "--epochs", dest="num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_batch_size", "--batch_size", dest="per_device_batch_size",
                        type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient_accum_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Linear warmup ratio over total steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging interval in training steps")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Checkpoint save interval in steps")
    parser.add_argument("--output_dir", type=str, default="./models/qwen3-4b-finetuned",
                        help="LoRA adapter output directory")

    # Hub
    parser.add_argument("--push_to_hub", type=lambda x: x.lower() == "true", default=False,
                        help="Push adapter to Hugging Face Hub after training")
    parser.add_argument("--hub_model_id", type=str, default="",
                        help="Hub repository name (required when push_to_hub is True)")

    return parser.parse_args()


def resolve_device(args: argparse.Namespace):
    """Detect and configure the compute device based on CLI args and system capabilities.

    Determines device placement, device map, CUDA availability, and whether
    4-bit quantization and mixed precision are feasible. Falls back to CPU
    gracefully when GPU is unavailable.

    Args:
        args: Parsed CLI arguments containing --device and --load_in_4bit / --torch_dtype options.

    Returns:
        Tuple of (device_map, use_cuda, load_in_4bit, dtype_str, compute_dtype).
    """
    if args.device == "cpu":
        device_map = {"": "cpu"}
        use_cuda = False
        print("Using device: cpu (--device cpu specified)")
    elif args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device gpu was specified but PyTorch has no CUDA available. "
                "Install a CUDA-enabled PyTorch build or use --device cpu."
            )
        device_map = {"": 0}
        use_cuda = True
        print(f"Using device: cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        if torch.cuda.is_available():
            device_map = {"": 0}
            use_cuda = True
            print(f"Using device: cuda:0 ({torch.cuda.get_device_name(0)})")
        else:
            device_map = {"": "cpu"}
            use_cuda = False
            print("Using device: cpu (CUDA unavailable)")
            if torch.version.cuda is None or not torch.backends.cuda.is_built():
                print("  Hint: The current PyTorch is a CPU build. Install CUDA-enabled PyTorch for GPU training.")
            elif shutil.which("nvidia-smi"):
                print("  Hint: NVIDIA driver detected but PyTorch cannot use CUDA.")

    load_in_4bit = args.load_in_4bit
    dtype_str = args.torch_dtype
    if not use_cuda:
        if load_in_4bit:
            print("  Note: 4-bit quantization requires GPU; falling back to full precision on CPU.")
            load_in_4bit = False
        if dtype_str != "float32":
            print(f"  Note: Using float32 on CPU instead of {dtype_str}.")
            dtype_str = "float32"

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map[dtype_str]

    return device_map, use_cuda, load_in_4bit, compute_dtype


def load_tokenizer(model_id: str):
    """Load the tokenizer for a given Hugging Face model ID.

    Args:
        model_id: Hugging Face model identifier string.

    Returns:
        PreTrainedTokenizer: Loaded tokenizer with pad_token set to eos_token if missing.
    """
    print("\nStep 1/6: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer loaded: {model_id}")
    return tokenizer


def load_base_model(
    model_id: str,
    load_in_4bit: bool,
    compute_dtype: torch.dtype,
    device_map: dict,
):
    """Load the base causal language model, optionally with 4-bit quantization.

    Args:
        model_id: Hugging Face model identifier.
        load_in_4bit: Whether to apply 4-bit QLoRA quantization.
        compute_dtype: Target computation dtype (bfloat16, float16, or float32).
        device_map: Device placement mapping (e.g., {"": 0} for GPU, {"": "cpu"} for CPU).

    Returns:
        PreTrainedModel: The loaded (and optionally quantized) base model.
    """
    print("\nStep 2/6: Loading base model...")
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    print(f"  Model loaded: {model_id}")
    return model


def apply_lora(model, args: argparse.Namespace, target_modules: list):
    """Configure and apply LoRA adapters on top of the base model.

    Args:
        model: The base (or quantized) model to wrap with LoRA.
        args: Parsed CLI arguments containing LoRA hyperparameters.
        target_modules: List of target module names for LoRA adaptation.

    Returns:
        PeftModel: The model wrapped with LoRA adapters.
    """
    print("\nStep 3/6: Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False  # KV cache must be disabled for gradient checkpointing
    model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %:          {100 * trainable_params / total_params:.4f}%")
    return model


def load_dataset_from_path(args: argparse.Namespace):
    """Load and format the training dataset from a local file or Hugging Face hub.

    Supports JSONL (one JSON object per line), JSON array, JSON dict, and
    Hugging Face dataset identifiers. Auto-detects format based on the path prefix.

    Args:
        args: Parsed CLI arguments containing dataset_path, dataset_format, and use_thinking.

    Returns:
        Tuple of (train_dataset, eval_dataset) as Dataset objects.
    """
    print("\nStep 4/6: Loading and formatting dataset...")
    if args.dataset_path.startswith(("/", "./", ".")):
        if args.dataset_path.endswith(".jsonl"):
            records = []
            with open(args.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            dataset = Dataset.from_list(records)
        else:
            with open(args.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                dataset = Dataset.from_list(data)
            elif isinstance(data, dict):
                dataset = Dataset.from_dict(data)
            else:
                raise ValueError(f"Unsupported JSON structure: {args.dataset_path}")
    elif args.dataset_path == "":
        raise ValueError("dataset_path must not be empty.")
    else:
        dataset = load_dataset(args.dataset_path, split="train")

    print(f"  Raw examples: {len(dataset)}")

    # Format each example into ChatML text for SFTTrainer tokenization
    def format_example(example):
        if args.dataset_format == "chat":
            messages = example.get("messages", [])
            if not messages:
                raise ValueError("chat format requires a non-empty messages field.")
        elif args.dataset_format == "alpaca":
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            output = example.get("output", "")
            user_content = f"{instruction}\n\n{inp}".strip() if inp else instruction.strip()
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output.strip()},
            ]
        elif args.dataset_format == "text":
            text = example.get("text", "")
            parts = text.split("<|im_end|>\n<|im_start|>")
            messages = []
            for i, part in enumerate(parts):
                part = part.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": part})
        else:
            raise ValueError(f"Unsupported dataset format: {args.dataset_format}")

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
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"  Train examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"  Eval examples:  {len(eval_dataset)}")

    return train_dataset, eval_dataset


def configure_training(
    args: argparse.Namespace,
    use_cuda: bool,
    eval_dataset,
    dtype_str: str,
):
    """Configure and run the SFTTrainer training loop.

    Sets up mixed precision flags, SFTConfig with training hyperparameters,
    instantiates the SFTTrainer, and runs training.

    Args:
        model: The LoRA-wrapped PEFT model.
        tokenizer: The model tokenizer.
        train_dataset: Training dataset.
        args: Parsed CLI arguments with training hyperparameters.
        use_cuda: Whether CUDA is available.
        eval_dataset: Evaluation dataset (or None).
        dtype_str: Precision string from CLI ("bfloat16", "float16", "float32").

    Returns:
        The SFTTrainer instance after training completes.
    """
    print("\nStep 5/6: Configuring Trainer and starting training...")
    torch_dtype = dtype_str

    if torch_dtype == "float32" or not use_cuda:
        use_bf16, use_fp16 = False, False
    elif torch_dtype == "float16":
        use_bf16, use_fp16 = False, True
    elif torch.cuda.is_bf16_supported():
        use_bf16, use_fp16 = True, False
    else:
        use_bf16, use_fp16 = False, True

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accum_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        use_cpu=not use_cuda,
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
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    return trainer


def save_and_push(trainer, tokenizer, args: argparse.Namespace):
    """Save the trained LoRA adapter locally and optionally push to Hugging Face Hub.

    Args:
        trainer: The SFTTrainer instance with the trained model.
        tokenizer: The tokenizer to save alongside the adapter.
        args: Parsed CLI arguments with output_dir and hub settings.
    """
    print("\nStep 6/6: Saving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Adapter saved to: {args.output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id is required when push_to_hub=True.")
        print(f"  Pushing adapter to hub: {args.hub_model_id}")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print("  Push complete.")


def main():
    """Orchestrate the full SFT pipeline: parse args, load model, apply LoRA, train, and save."""
    args = parse_args()
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    print("=" * 60)
    print("Training configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"  target_modules: {target_modules}")
    print("=" * 60)

    global tokenizer, model, train_dataset

    device_map, use_cuda, load_in_4bit, compute_dtype = resolve_device(args)
    tokenizer = load_tokenizer(args.model_id)
    model = load_base_model(args.model_id, load_in_4bit, compute_dtype, device_map)
    model = apply_lora(model, args, target_modules)
    train_dataset, eval_dataset = load_dataset_from_path(args)
    trainer = configure_training(args, use_cuda, eval_dataset, args.torch_dtype)
    save_and_push(trainer, tokenizer, args)

    print("=" * 60)
    print("Training complete.")


if __name__ == "__main__":
    # Global references used across functions for simplicity in a single-file script
    tokenizer = None
    model = None
    train_dataset = None
    main()
