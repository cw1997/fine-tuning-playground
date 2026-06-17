#!/usr/bin/env python3
"""
CLI entry point for fine-tuning Qwen3-4B.

Usage:
    python run_sft.py --dataset_path /path/to/data.jsonl --output_dir ./my-adapter
    python run_sft.py --dataset_path databricks/databricks-dolly-15k --dataset_format alpaca --epochs 5
"""

from env_utils import fix_ssl_cert_env

fix_ssl_cert_env()

import argparse
import sys

from config import FinetuneConfig
from train import train


def parse_args(argv=None) -> FinetuneConfig:
    """
    Parse command-line arguments and return a FinetuneConfig.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        A FinetuneConfig populated with parsed CLI values.
    """
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning for Qwen3-4B with LoRA/QLoRA."
    )

    # --- Model ---
    parser.add_argument("--model_id", type=str, default=None,
                        help="Hugging Face model identifier (default: Qwen/Qwen3-4B)")
    parser.add_argument("--load_in_4bit", type=lambda x: x.lower() == "true", default=None,
                        help="Enable 4-bit QLoRA quantization (default: True)")
    parser.add_argument("--torch_dtype", type=str, default=None,
                        help="Computation dtype: bfloat16, float16, or float32")

    # --- LoRA ---
    parser.add_argument("--lora_r", type=int, default=None,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=None,
                        help="LoRA dropout (default: 0.05)")

    # --- Data ---
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset file or Hugging Face dataset name")
    parser.add_argument("--dataset_format", type=str, default=None,
                        choices=["chat", "alpaca", "text"],
                        help="Dataset format (default: chat)")
    parser.add_argument("--test_split", type=float, default=None,
                        help="Fraction for evaluation split (default: 0.05)")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Maximum sequence length (default: 4096)")

    # --- Training ---
    parser.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=None,
                        help="Peak learning rate (default: 2e-4)")
    parser.add_argument("--epochs", "--num_epochs", dest="num_epochs", type=int, default=None,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", "--per_device_batch_size",
                        dest="per_device_batch_size", type=int, default=None,
                        help="Per-device batch size (default: 2)")
    parser.add_argument("--gradient_accum_steps", type=int, default=None,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--warmup_ratio", type=float, default=None,
                        help="Warmup ratio (default: 0.03)")
    parser.add_argument("--logging_steps", type=int, default=None,
                        help="Log every N steps (default: 10)")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every N steps (default: 200)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for adapter (default: ./qwen3-4b-finetuned)")

    # --- Hub ---
    parser.add_argument("--push_to_hub", type=lambda x: x.lower() == "true", default=None,
                        help="Push adapter to Hugging Face Hub (default: False)")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Target hub repository name (required if push_to_hub is True)")

    # --- Inference ---
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max tokens for generation (default: 2048)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Nucleus sampling threshold (default: 0.9)")
    parser.add_argument("--use_thinking", type=lambda x: x.lower() == "true", default=None,
                        help="Enable Qwen3 thinking mode (default: False)")

    args = parser.parse_args(argv)
    return FinetuneConfig.from_args(args)


def main() -> None:
    """Parse CLI arguments and launch the training pipeline."""
    config = parse_args()
    print(config)
    train(config)


if __name__ == "__main__":
    main()
