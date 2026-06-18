#!/usr/bin/env python3
"""
Inference script for Qwen3 / Qwen3.5 models.

Supports base model inference, fine-tuned (LoRA adapter) inference,
and side-by-side comparison between base and fine-tuned outputs.

Usage:
    python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B --prompt "Hello"
    python src/inference.py --mode finetuned --adapter_path ./models/ntnu
    python src/inference.py --mode compare --adapter_path ./models/ntnu

    # Interactive mode (default): keep entering prompts after loading; type quit to exit
    python src/inference.py --mode finetuned --adapter_path ./models/ntnu/qwen3.5-4b

    # One-shot batch mode for scripts
    python src/inference.py --mode base --prompt "Hello" --no_interactive
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel

from model_utils import (
    DEFAULT_MODEL_ID,
    fix_ssl_certificates,
    load_base_model,
    load_tokenizer,
    resolve_device,
)

fix_ssl_certificates()

ADAPTER_CONFIG_NAME = "adapter_config.json"
EXIT_COMMANDS = frozenset({"quit", "exit", "q"})


def configure_stdio_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows for Chinese output."""
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the inference script.

    Supports three modes (base, finetuned, compare) with configurable model,
    quantization, generation parameters, and device selection.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Qwen3 inference")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Base model ID")
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
    parser.add_argument(
        "--no_interactive",
        action="store_true",
        help="Run batch inference only and exit (no interactive prompt loop)",
    )

    args = parser.parse_args()

    if args.mode in ("finetuned", "compare") and args.adapter_path is None:
        parser.error("--adapter_path is required in finetuned / compare mode.")

    return args


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
    tokenizer = load_tokenizer(model_id)
    print(f"Loading base model from {model_id}...")
    base_model = load_base_model(model_id, load_in_4bit, compute_dtype, device_map)
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
    """Build a list of batch test prompts from CLI args or defaults.

    If --prompt is provided, a single-turn conversation is created.
    In interactive mode (default), an empty list is returned when --prompt
    is omitted so the script goes straight to the input loop.
    With --no_interactive and no --prompt, default NTNU test prompts are used.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of message lists, each suitable for tokenizer.apply_chat_template.
    """
    if args.prompt:
        return [[{"role": "user", "content": args.prompt}]]
    if args.no_interactive:
        return [
            [{"role": "user", "content": "Tell me about National Taiwan Normal University."}],
            [{"role": "user", "content": "What is the history of NTNU?"}],
        ]
    return []


def _generation_kwargs(args: argparse.Namespace) -> tuple:
    """Return generation keyword arguments as a tuple for ``generate_response``."""
    return (
        args.use_thinking,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )


def _print_compare_responses(base_response: str, ft_response: str) -> None:
    """Print base and fine-tuned responses in a consistent comparison format.

    Args:
        base_response: Response from the base model.
        ft_response: Response from the fine-tuned model.
    """
    print("\n--- Base model response ---")
    print(base_response)
    print("\n--- Fine-tuned model response ---")
    print(ft_response)


def run_batch_mode(
    model,
    tokenizer,
    label: str,
    test_prompts: List[List[Dict[str, str]]],
    args: argparse.Namespace,
) -> None:
    """Run batch inference with a single model.

    Args:
        model: The model to use for generation.
        tokenizer: The model tokenizer.
        label: Human-readable label for the model (e.g., "BASE").
        test_prompts: List of test prompt message lists.
        args: Parsed CLI arguments with generation parameters.
    """
    print(f"Running inference with {label} model...")
    gen_kwargs = _generation_kwargs(args)
    for messages in test_prompts:
        print(f"\nPrompt: {messages[-1]['content']}")
        response = generate_response(model, tokenizer, messages, *gen_kwargs)
        print(f"Response:\n{response}")


def run_compare_mode(
    base_model,
    ft_model,
    tokenizer,
    test_prompts: List[List[Dict[str, str]]],
    args: argparse.Namespace,
) -> None:
    """Run side-by-side comparison of base vs. fine-tuned model outputs.

    Args:
        base_model: The base pretrained model.
        ft_model: PEFT-wrapped model with the LoRA adapter already loaded.
        tokenizer: The model tokenizer.
        test_prompts: List of test prompt message lists.
        args: Parsed CLI arguments with generation parameters.
    """
    print("Comparing BASE vs FINE-TUNED model...")
    gen_kwargs = _generation_kwargs(args)
    for i, messages in enumerate(test_prompts):
        print(f"\n{'=' * 60}")
        prompt_preview = messages[-1]["content"]
        if len(prompt_preview) > 80:
            prompt_preview = f"{prompt_preview[:80]}..."
        print(f"Prompt {i + 1}: {prompt_preview}")

        base_response = generate_response(base_model, tokenizer, messages, *gen_kwargs)
        ft_response = generate_response(ft_model, tokenizer, messages, *gen_kwargs)
        _print_compare_responses(base_response, ft_response)


def run_interactive_loop(
    mode: str,
    base_model,
    tokenizer,
    args: argparse.Namespace,
    ft_model=None,
) -> None:
    """Read prompts from stdin and generate responses until the user exits.

    Args:
        mode: Inference mode (base, finetuned, or compare).
        base_model: The base pretrained model.
        tokenizer: The model tokenizer.
        args: Parsed CLI arguments with generation parameters.
        ft_model: PEFT-wrapped fine-tuned model; required for finetuned/compare.
    """
    print("\nInteractive mode. Enter prompts below (quit / exit / q to stop):")
    gen_kwargs = _generation_kwargs(args)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            break

        messages = [{"role": "user", "content": user_input}]

        if mode == "compare":
            base_response = generate_response(base_model, tokenizer, messages, *gen_kwargs)
            ft_response = generate_response(ft_model, tokenizer, messages, *gen_kwargs)
            _print_compare_responses(base_response, ft_response)
        else:
            model = base_model if mode == "base" else ft_model
            response = generate_response(model, tokenizer, messages, *gen_kwargs)
            print(f"Response:\n{response}")


def main():
    """Orchestrate the full inference pipeline: parse args, load model, and run the selected mode."""
    configure_stdio_utf8()
    args = parse_args()
    device_map, _, load_in_4bit, compute_dtype = resolve_device(args)
    base_model, tokenizer = load_base_model_and_tokenizer(
        args.model_id, load_in_4bit, compute_dtype, device_map,
    )

    ft_model = None
    if args.mode in ("finetuned", "compare"):
        resolved_path = resolve_adapter_path(args.adapter_path)
        print(f"Loading LoRA adapter from {resolved_path}...")
        ft_model = PeftModel.from_pretrained(base_model, resolved_path)

    test_prompts = build_test_prompts(args)

    if test_prompts:
        if args.mode == "base":
            run_batch_mode(base_model, tokenizer, "BASE", test_prompts, args)
        elif args.mode == "finetuned":
            run_batch_mode(ft_model, tokenizer, "FINE-TUNED", test_prompts, args)
        elif args.mode == "compare":
            run_compare_mode(base_model, ft_model, tokenizer, test_prompts, args)

    if not args.no_interactive:
        run_interactive_loop(args.mode, base_model, tokenizer, args, ft_model=ft_model)


if __name__ == "__main__":
    main()
