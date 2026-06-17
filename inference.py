"""
Inference module for Qwen3-4B model inference.

Supports both the base model (pre-fine-tuning) and the fine-tuned model
(with a trained LoRA adapter), using the Qwen3 chat template with
optional thinking mode.
"""

from env_utils import fix_ssl_cert_env

fix_ssl_cert_env()

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from model_utils import adapt_settings_for_device, print_compute_device
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3ForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)

ADAPTER_CONFIG_NAME = "adapter_config.json"


def resolve_adapter_path(adapter_path: str) -> str:
    """
    Resolve a local LoRA adapter path for PeftModel.from_pretrained.

    Converts relative paths to absolute (PEFT may treat './...' as a Hub repo ID),
    accepts checkpoint-* subdirectories, and raises a clear error when missing.
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

    search_dir = path.parent if path.exists() else Path.cwd()
    prefix = path.name[:6]
    suggestions = [
        str(p)
        for p in search_dir.iterdir()
        if p.is_dir()
        and prefix
        and prefix in p.name
        and p.name != path.name
        and (p / ADAPTER_CONFIG_NAME).is_file()
    ]

    message = (
        f"LoRA adapter not found at '{adapter_path}' (resolved: {path}).\n"
        f"Expected '{ADAPTER_CONFIG_NAME}' in the directory or a checkpoint-* subdirectory."
    )
    if suggestions:
        message += f"\nDid you mean: {', '.join(suggestions)}?"
    raise FileNotFoundError(message)


def _load_base_model_and_tokenizer(
    base_model_id: str,
    use_4bit: bool = True,
) -> Tuple[Qwen3ForCausalLM, PreTrainedTokenizer]:
    """
    Internal helper: load the base Qwen3 model and tokenizer.

    Args:
        base_model_id: Hugging Face identifier for the base Qwen3 model.
        use_4bit:      Whether to load with 4-bit quantization.

    Returns:
        Tuple of (base_model, tokenizer).
    """
    _, device_map, _ = print_compute_device()
    use_4bit, torch_dtype = adapt_settings_for_device(use_4bit, "bfloat16")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    print(f"Loading tokenizer from {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading base model from {base_model_id}...")
    model = Qwen3ForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    return model, tokenizer


def load_base_model(
    base_model_id: str,
    use_4bit: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the base Qwen3 model **without** any LoRA adapter.

    Use this for testing the model *before* fine-tuning.

    Args:
        base_model_id: Hugging Face identifier for the base Qwen3 model.
        use_4bit:      Whether to load with 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    model, tokenizer = _load_base_model_and_tokenizer(base_model_id, use_4bit)
    print("  (Base model loaded — no LoRA adapter attached)")
    return model, tokenizer


def load_finetuned_model(
    base_model_id: str,
    adapter_path: str,
    use_4bit: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the base model and attach the fine-tuned LoRA adapter.

    Use this for testing the model *after* fine-tuning.

    Args:
        base_model_id: Hugging Face identifier for the base Qwen3 model.
        adapter_path:  Path or hub ID where the LoRA adapter is saved.
        use_4bit:      Whether the base model was trained with 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    model, tokenizer = _load_base_model_and_tokenizer(base_model_id, use_4bit)

    resolved_path = resolve_adapter_path(adapter_path)
    print(f"Loading LoRA adapter from {resolved_path}...")
    model = PeftModel.from_pretrained(model, resolved_path)

    return model, tokenizer


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = False,
) -> str:
    """
    Generate a response from the fine-tuned model given a conversation.

    Applies the Qwen3 chat template, runs generation, and decodes the output,
    stripping the input prompt from the result.

    Args:
        model:           The fine-tuned Qwen3 model.
        tokenizer:       The associated tokenizer.
        messages:        Conversation history as [{"role": ..., "content": ...}, ...].
        max_new_tokens:  Maximum number of tokens to generate.
        temperature:     Sampling temperature (lower = more deterministic).
        top_p:           Nucleus sampling probability threshold.
        enable_thinking: Whether to enable the <think> reasoning mode.

    Returns:
        The generated text as a string.
    """
    # Apply chat template without tokenizing to get the formatted prompt text
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # Tokenize the prompt and move to the model device
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    # Generate
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

    # Decode only the newly generated tokens
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def compare_responses(
    base_model_id: str,
    adapter_path: str,
    test_prompts: List[Dict[str, str]],
    use_4bit: bool = True,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = False,
) -> List[Dict[str, Optional[str]]]:
    """
    Compare responses from the base model vs. the fine-tuned model.

    Loads both models (base without adapter, fine-tuned with adapter),
    runs each prompt through both, and returns the results side by side.

    Args:
        base_model_id: Hugging Face identifier for the base Qwen3 model.
        adapter_path:  Path or hub ID where the LoRA adapter is saved.
        test_prompts:  List of messages dicts (same format as generate_response).
        use_4bit:      Whether to use 4-bit quantization.
        max_new_tokens: Maximum tokens to generate.
        temperature:   Sampling temperature.
        top_p:         Nucleus sampling probability threshold.
        enable_thinking: Whether to enable the <think> reasoning mode.

    Returns:
        List of dicts with keys "prompt", "base_response", "finetuned_response".
    """
    # Load both models
    print("=" * 60)
    print("Loading base model (pre-fine-tuning)...")
    base_model, tokenizer = load_base_model(base_model_id, use_4bit)

    print("\nLoading fine-tuned model...")
    ft_model, _ = load_finetuned_model(base_model_id, adapter_path, use_4bit)

    results: List[Dict[str, Optional[str]]] = []
    for i, messages in enumerate(test_prompts):
        print(f"\n{'=' * 60}")
        print(f"Prompt {i + 1}: {messages[-1]['content'][:80]}...")

        print("\n--- Base model response ---")
        base_response = generate_response(
            base_model, tokenizer, messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        print(base_response)

        print("\n--- Fine-tuned model response ---")
        ft_response = generate_response(
            ft_model, tokenizer, messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        print(ft_response)

        results.append({
            "prompt": messages,
            "base_response": base_response,
            "finetuned_response": ft_response,
        })

    return results


def extract_thinking(text: str) -> Tuple[Optional[str], str]:
    """
    Extract the thinking / reasoning block from a Qwen3 response.

    Qwen3 wraps reasoning content in <think>...</think> tags when
    thinking mode is enabled.

    Args:
        text: The full model response, possibly containing <think> tags.

    Returns:
        Tuple of (thinking_content, answer_content).
        If no thinking block is found, thinking_content is None.
    """
    start_tag = "<think>"
    end_tag = "</think>"

    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx != -1 and end_idx != -1:
        thinking = text[start_idx + len(start_tag):end_idx].strip()
        answer = text[end_idx + len(end_tag):].strip()
        return thinking, answer

    return None, text.strip()


def main() -> None:
    """
    CLI entry point for inference.

    Usage examples:
        # Test the base model (pre-fine-tuning)
        python inference.py --mode base

        # Test the fine-tuned model
        python inference.py --mode finetuned --adapter_path ./ntnu-finetuned

        # Compare base vs. fine-tuned
        python inference.py --mode compare --adapter_path ./ntnu-finetuned
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on Qwen3-4B")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B",
                        help="Base model identifier")
    parser.add_argument("--mode", type=str, default="base",
                        choices=["base", "finetuned", "compare"],
                        help="Inference mode: base, finetuned, or compare")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter (required for finetuned/compare)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt string (uses default test prompts if not set)")
    parser.add_argument("--use_4bit", type=lambda x: x.lower() == "true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--use_thinking", type=lambda x: x.lower() == "true", default=False,
                        help="Enable Qwen3 thinking mode")

    args = parser.parse_args()

    if args.mode in ("finetuned", "compare") and args.adapter_path is None:
        parser.error("--adapter_path is required for finetuned and compare modes.")

    # Build test prompts
    if args.prompt:
        test_prompts = [[{"role": "user", "content": args.prompt}]]
    else:
        test_prompts = [
            [{"role": "user", "content": "Tell me about National Taiwan Normal University."}],
            [{"role": "user", "content": "What is the history of NTNU?"}],
        ]

    if args.mode == "base":
        print("Running inference with BASE model (pre-fine-tuning)...")
        model, tokenizer = load_base_model(args.model_id, args.use_4bit)
        for messages in test_prompts:
            print(f"\nPrompt: {messages[-1]['content']}")
            response = generate_response(
                model, tokenizer, messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=args.use_thinking,
            )
            print(f"Response:\n{response}")

    elif args.mode == "finetuned":
        print("Running inference with FINE-TUNED model...")
        model, tokenizer = load_finetuned_model(
            args.model_id, args.adapter_path, args.use_4bit
        )
        for messages in test_prompts:
            print(f"\nPrompt: {messages[-1]['content']}")
            response = generate_response(
                model, tokenizer, messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=args.use_thinking,
            )
            print(f"Response:\n{response}")

    elif args.mode == "compare":
        print("Comparing BASE vs FINE-TUNED model responses...")
        compare_responses(
            args.model_id, args.adapter_path, test_prompts,
            use_4bit=args.use_4bit,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.use_thinking,
        )


if __name__ == "__main__":
    main()
