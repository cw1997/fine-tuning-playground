#!/usr/bin/env python3
"""
Inference script for Qwen3 / Qwen3.5 models.

Supports base model inference, fine-tuned (LoRA adapter) inference,
and side-by-side comparison between base and fine-tuned outputs.

Usage:
    python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B --prompt "Hello"
    python src/inference.py --mode finetuned --adapter_path ./models/ntnu-finetuned
    python src/inference.py --mode compare --adapter_path ./models/ntnu-finetuned
"""

import os
import warnings

import certifi

# Fix SSL certificate path issues commonly found on Windows / conda environments
for _ssl_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    _ssl_val = os.environ.get(_ssl_var)
    if _ssl_val and not os.path.isfile(_ssl_val):
        os.environ[_ssl_var] = certifi.where()

# Suppress FutureWarning from torch internals and Triton FLOP counter warnings
warnings.filterwarnings("ignore", message=".*_check_is_size.*")
warnings.filterwarnings("ignore", message=".*triton not found.*")

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER_CONFIG_NAME = "adapter_config.json"


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the inference script.

    Supports three modes (base, finetuned, compare) with configurable model,
    quantization, generation parameters, and device selection.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Qwen3 inference")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B", help="Base model ID")
    parser.add_argument("--mode", type=str, default="base", choices=["base", "finetuned", "compare"],
                        help="Inference mode")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--prompt", type=str, default=None, help="Single test prompt")
    parser.add_argument("--load_in_4bit", type=lambda x: x.lower() == "true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Computation precision")
    parser.add_argument("--device", type=str, default=None, choices=["gpu", "cpu"],
                        help="Compute device (auto-detected by default)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--use_thinking", type=lambda x: x.lower() == "true", default=False,
                        help="Enable thinking mode")

    args = parser.parse_args()

    if args.mode in ("finetuned", "compare") and args.adapter_path is None:
        parser.error("--adapter_path is required in finetuned / compare mode.")

    return args


def resolve_device(args: argparse.Namespace):
    """Detect and return device configuration based on CLI args and system capabilities.

    Args:
        args: Parsed CLI arguments containing --device, --load_in_4bit, and --torch_dtype.

    Returns:
        Tuple of (device_map, use_cuda, load_in_4bit, compute_dtype).
    """
    if args.device == "cpu":
        device_map = {"": "cpu"}
        use_cuda = False
    elif args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--device gpu was specified but CUDA is unavailable.")
        device_map = {"": 0}
        use_cuda = True
    else:
        if torch.cuda.is_available():
            device_map = {"": 0}
            use_cuda = True
        else:
            device_map = {"": "cpu"}
            use_cuda = False

    load_in_4bit = args.load_in_4bit
    dtype_str = args.torch_dtype
    if not use_cuda:
        if load_in_4bit:
            print("  Note: 4-bit quantization requires GPU; falling back to full precision on CPU.")
            load_in_4bit = False
        if dtype_str != "float32":
            dtype_str = "float32"

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map[dtype_str]

    if use_cuda:
        print(f"Using device: cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")

    return device_map, use_cuda, load_in_4bit, compute_dtype


def load_base_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool,
    compute_dtype: torch.dtype,
    device_map: dict,
):
    """Load the base model and its tokenizer from Hugging Face.

    Args:
        model_id: Hugging Face model identifier.
        load_in_4bit: Whether to apply 4-bit quantization.
        compute_dtype: Target computation dtype.
        device_map: Device placement mapping.

    Returns:
        Tuple of (base_model, tokenizer).
    """
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading base model from {model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return base_model, tokenizer


def resolve_adapter_path(adapter_path: str) -> str:
    """Resolve the LoRA adapter directory path, auto-selecting the latest checkpoint if needed.

    If the exact path contains adapter_config.json, it is used directly.
    Otherwise, subdirectories matching checkpoint-* are scanned and the
    highest-numbered checkpoint is selected.

    Args:
        adapter_path: User-provided adapter path (relative or absolute).

    Returns:
        Resolved absolute path to the adapter directory.

    Raises:
        FileNotFoundError: If no adapter is found at the given or resolved path.
    """
    path = Path(adapter_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()

    if (path / ADAPTER_CONFIG_NAME).is_file():
        return str(path)

    checkpoints = sorted(
        (p for p in path.glob("checkpoint-*") if (p / ADAPTER_CONFIG_NAME).is_file()),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else 0,
    )
    if checkpoints:
        resolved = checkpoints[-1]
        print(f"  Using checkpoint adapter: {resolved}")
        return str(resolved)

    raise FileNotFoundError(
        f"LoRA adapter not found at '{adapter_path}' (resolved: {path})."
    )


def generate_response(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    use_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate a text response from the given model using a chat-formatted prompt.

    Applies the tokenizer's chat template, runs generation with the configured
    sampling parameters, and decodes the output tokens.

    Args:
        model: The model to use for generation (base or PEFT-wrapped).
        tokenizer: The model tokenizer.
        messages: List of chat messages (e.g., [{"role": "user", "content": "..."}]).
        use_thinking: Whether to enable Qwen3 thinking mode in the chat template.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling probability threshold.

    Returns:
        The decoded text response, with special tokens stripped.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=use_thinking,
    )
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_test_prompts(args: argparse.Namespace) -> List[List[Dict[str, str]]]:
    """Build a list of test prompts from CLI args or defaults.

    If --prompt is provided, a single-turn conversation is created.
    Otherwise, a set of default English prompts about NTNU are used.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of message lists, each suitable for tokenizer.apply_chat_template.
    """
    if args.prompt:
        return [[{"role": "user", "content": args.prompt}]]
    return [
        [{"role": "user", "content": "Tell me about National Taiwan Normal University."}],
        [{"role": "user", "content": "What is the history of NTNU?"}],
    ]


def run_base_mode(base_model, tokenizer, test_prompts: List[List[Dict[str, str]]], args: argparse.Namespace):
    """Run inference using the base model without any LoRA adapter.

    Args:
        base_model: The base pretrained model.
        tokenizer: The model tokenizer.
        test_prompts: List of test prompt message lists.
        args: Parsed CLI arguments with generation parameters.
    """
    print("Running inference with BASE model...")
    for messages in test_prompts:
        print(f"\nPrompt: {messages[-1]['content']}")
        response = generate_response(
            base_model, tokenizer, messages,
            args.use_thinking, args.max_new_tokens, args.temperature, args.top_p,
        )
        print(f"Response:\n{response}")


def run_finetuned_mode(base_model, tokenizer, test_prompts: List[List[Dict[str, str]]], args: argparse.Namespace):
    """Run inference using a fine-tuned LoRA adapter loaded on top of the base model.

    Args:
        base_model: The base pretrained model.
        tokenizer: The model tokenizer.
        test_prompts: List of test prompt message lists.
        args: Parsed CLI arguments with adapter_path and generation parameters.
    """
    resolved_path = resolve_adapter_path(args.adapter_path)
    print(f"Loading LoRA adapter from {resolved_path}...")
    model = PeftModel.from_pretrained(base_model, resolved_path)
    print("Running inference with FINE-TUNED model...")
    for messages in test_prompts:
        print(f"\nPrompt: {messages[-1]['content']}")
        response = generate_response(
            model, tokenizer, messages,
            args.use_thinking, args.max_new_tokens, args.temperature, args.top_p,
        )
        print(f"Response:\n{response}")


def run_compare_mode(base_model, tokenizer, test_prompts: List[List[Dict[str, str]]], args: argparse.Namespace):
    """Run side-by-side comparison of base vs. fine-tuned model outputs.

    Generates responses from both models for each prompt and displays them
    in a structured comparison format.

    Args:
        base_model: The base pretrained model.
        tokenizer: The model tokenizer.
        test_prompts: List of test prompt message lists.
        args: Parsed CLI arguments with adapter_path and generation parameters.
    """
    print("Comparing BASE vs FINE-TUNED model...")
    resolved_path = resolve_adapter_path(args.adapter_path)

    base_results = []
    for messages in test_prompts:
        base_results.append(
            generate_response(
                base_model, tokenizer, messages,
                args.use_thinking, args.max_new_tokens, args.temperature, args.top_p,
            )
        )

    ft_model = PeftModel.from_pretrained(base_model, resolved_path)
    for i, messages in enumerate(test_prompts):
        print(f"\n{'=' * 60}")
        print(f"Prompt {i + 1}: {messages[-1]['content'][:80]}...")

        print("\n--- Base model response ---")
        print(base_results[i])

        print("\n--- Fine-tuned model response ---")
        ft_response = generate_response(
            ft_model, tokenizer, messages,
            args.use_thinking, args.max_new_tokens, args.temperature, args.top_p,
        )
        print(ft_response)


def main():
    """Orchestrate the full inference pipeline: parse args, load model, and run the selected mode."""
    args = parse_args()
    device_map, use_cuda, load_in_4bit, compute_dtype = resolve_device(args)
    base_model, tokenizer = load_base_model_and_tokenizer(
        args.model_id, load_in_4bit, compute_dtype, device_map,
    )

    test_prompts = build_test_prompts(args)

    if args.mode == "base":
        run_base_mode(base_model, tokenizer, test_prompts, args)
    elif args.mode == "finetuned":
        run_finetuned_mode(base_model, tokenizer, test_prompts, args)
    elif args.mode == "compare":
        run_compare_mode(base_model, tokenizer, test_prompts, args)


if __name__ == "__main__":
    main()
