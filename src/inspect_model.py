#!/usr/bin/env python3
"""
Model inspection script for Qwen3 / Qwen3.5 models.

Prints a detailed, standardized English report of model internals:
parameter counts, vocabulary size, token embedding dimension, layer layout,
attention configuration, and tokenizer metadata.

Usage:
    python src/inspect_model.py --model_id Qwen/Qwen3.5-0.8B
    python src/inspect_model.py --model_id Qwen/Qwen3.5-4B --load_in_4bit true
    python src/inspect_model.py --model_id Qwen/Qwen3.5-4B --no_load
"""

import argparse
from typing import Dict, Tuple

import torch
from transformers import AutoConfig

from model_utils import (
    DEFAULT_MODEL_ID,
    fix_ssl_certificates,
    load_base_model,
    load_tokenizer,
    resolve_device,
)

fix_ssl_certificates()

MIB = 1024 * 1024

_DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float64: 8,
}


def _str2bool(value: str) -> bool:
    """Parse a CLI boolean string."""
    return value.lower() == "true"


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the inspection script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Inspect Qwen3 model internals")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="Hugging Face model ID")
    parser.add_argument("--load_in_4bit", type=_str2bool, default=False,
                        help="Load the model in 4-bit quantization before inspecting")
    parser.add_argument("--torch_dtype", type=str, default="float32",
                        choices=["bfloat16", "float16", "float32"],
                        help="Computation precision")
    parser.add_argument("--device", type=str, default=None, choices=["gpu", "cpu"],
                        help="Compute device (auto-detected by default)")
    parser.add_argument("--no_load", action="store_true",
                        help="Inspect configuration and tokenizer only; do not load weights")
    return parser.parse_args()


def _format_number(value: int) -> str:
    """Format an integer with thousands separators for display.

    Args:
        value: Integer to format.

    Returns:
        Formatted string using comma grouping.
    """
    return f"{value:,}"


def _param_count(model) -> Tuple[int, int]:
    """Count total and trainable parameters of a model.

    Args:
        model: The inspected model (may be PEFT-wrapped or quantized).

    Returns:
        Tuple of ``(total_params, trainable_params)``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _bytes_per_param(dtype: torch.dtype) -> int:
    """Return the byte width of a given tensor dtype.

    Args:
        dtype: The torch dtype to inspect.

    Returns:
        Byte width of the dtype, or 1 if unknown.
    """
    return _DTYPE_BYTES.get(dtype, 1)


def _config_attr(config, *names, default="N/A"):
    """Fetch the first non-None attribute among the given config names.

    Different model families use different attribute names for the same
    concept; this helper normalizes the lookup.

    Args:
        config: The model configuration object.
        *names: Candidate attribute names, tried in order.
        default: Fallback value when none of the names resolve.

    Returns:
        The first matching attribute value or the default.
    """
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def inspect_config(config) -> None:
    """Print a standardized report of model architecture settings.

    Args:
        config: The model configuration object.
    """
    print(f"\n{'=' * 60}\nMODEL CONFIGURATION\n{'=' * 60}")

    hidden_size = _config_attr(config, "hidden_size", "n_embd")
    num_heads = _config_attr(config, "num_attention_heads", "n_head")
    head_dim = _config_attr(config, "head_dim", "N/A")
    if head_dim == "N/A" and isinstance(hidden_size, int) and isinstance(num_heads, int):
        head_dim = hidden_size // num_heads

    rows = [
        ("Vocabulary size", _config_attr(config, "vocab_size", "n_vocab")),
        ("Hidden size (embedding dimension)", hidden_size),
        ("Number of transformer layers",
         _config_attr(config, "num_hidden_layers", "n_layer")),
        ("Number of attention heads", num_heads),
        ("Number of KV heads", _config_attr(config, "num_key_value_heads", num_heads)),
        ("Head dimension", head_dim),
        ("Intermediate (FFN) size", _config_attr(config, "intermediate_size")),
        ("Max position embeddings", _config_attr(config, "max_position_embeddings", "n_positions")),
        ("Context length (seq_length)",
         _config_attr(config, "seq_length", "context_length", "N/A")),
        ("Activation function", _config_attr(config, "hidden_act", "activation")),
        ("RoPE theta", _config_attr(config, "rope_theta")),
        ("Tie word embeddings", _config_attr(config, "tie_word_embeddings", False)),
        ("Architecture", ", ".join(getattr(config, "architectures", []) or [])),
    ]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"  {key:<{width}} : {value}")


def inspect_tokenizer(tokenizer) -> None:
    """Print a standardized report of tokenizer metadata.

    Args:
        tokenizer: The loaded tokenizer.
    """
    print(f"\n{'=' * 60}\nTOKENIZER\n{'=' * 60}")

    special_names = ("bos_token", "eos_token", "pad_token", "unk_token", "mask_token")
    special_tokens = {
        name.upper().replace("_TOKEN", ""): getattr(tokenizer, name)
        for name in special_names
        if getattr(tokenizer, name, None) is not None
    }

    rows = [
        ("Tokenizer class", tokenizer.__class__.__name__),
        ("Vocabulary size (len(tokenizer))", _format_number(len(tokenizer))),
        ("Vocabulary size (tokenizer.vocab_size)",
         _format_number(getattr(tokenizer, "vocab_size", len(tokenizer)))),
        ("Padding side", getattr(tokenizer, "padding_side", "N/A")),
        ("Truncation side", getattr(tokenizer, "truncation_side", "N/A")),
        ("Chat template", "yes" if getattr(tokenizer, "chat_template", None) else "no"),
    ]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"  {key:<{width}} : {value}")

    print(f"\n  Special tokens ({len(special_tokens)}):")
    for name, token in special_tokens.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"    {name:<5} : {token!r} (id={token_id})")


def inspect_model(model, tokenizer) -> None:
    """Print a standardized report of the loaded model weights.

    Args:
        model: The loaded model (base or PEFT-wrapped).
        tokenizer: The tokenizer paired with the model.
    """
    print(f"\n{'=' * 60}\nMODEL WEIGHTS\n{'=' * 60}")

    total_params, trainable_params = _param_count(model)
    frozen_params = total_params - trainable_params

    print(f"  Model class                 : {model.__class__.__name__}")
    print(f"  Total parameters            : {_format_number(total_params)}")
    print(f"  Trainable parameters        : {_format_number(trainable_params)}")
    print(f"  Frozen parameters           : {_format_number(frozen_params)}")
    trainable_ratio = trainable_params / total_params if total_params else 0.0
    print(f"  Trainable ratio             : {trainable_ratio:.2%}")

    emb = model.get_input_embeddings()
    if emb is not None:
        emb_shape = tuple(emb.weight.shape)
        emb_mib = emb.weight.numel() * _bytes_per_param(emb.weight.dtype) / MIB
        emb_dtype = emb.weight.dtype
    else:
        emb_shape, emb_mib, emb_dtype = "N/A", 0.0, "N/A"

    print(f"  Token embedding shape       : {emb_shape}")
    print(f"  Embedding memory estimate   : {emb_mib:,.2f} MiB")
    print(f"  Parameter dtype             : {next(model.parameters()).dtype}")
    print(f"  Embedding dtype             : {emb_dtype}")
    print(f"  Device placement            : "
          f"{emb.weight.device if emb is not None else 'N/A'}")

    if tokenizer is not None and emb is not None:
        model_vocab = emb_shape[0]
        print(f"  Embedding rows vs vocab size: "
              f"{model_vocab} (model) vs {len(tokenizer)} (tokenizer)")

    peft_config = getattr(model, "peft_config", None)
    if peft_config:
        print(f"  PEFT adapters               : {list(peft_config.keys())}")


def main() -> None:
    """Orchestrate the inspection: parse args, load config/model, print the report."""
    args = parse_args()
    device_map, _, load_in_4bit, compute_dtype = resolve_device(args)

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    inspect_config(config)

    tokenizer = load_tokenizer(args.model_id)
    inspect_tokenizer(tokenizer)

    if not args.no_load:
        print(f"\nLoading model from {args.model_id}...")
        model = load_base_model(args.model_id, load_in_4bit, compute_dtype, device_map)
        inspect_model(model, tokenizer)
    else:
        print(f"\nSkipping weight loading (--no_load specified).")


if __name__ == "__main__":
    main()
