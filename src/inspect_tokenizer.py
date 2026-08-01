#!/usr/bin/env python3
"""
Tokenizer inspection script for Qwen3 / Qwen3.5 models.

Prints how a sentence is split into tokens, the vocabulary id of each piece,
and the token embedding vector (from the model's input embedding layer) for
every token. Also supports looking up a single token string or token id to
show its index position and embedding vector.

Usage:
    python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "請告訴我國立台灣師範大學的具體位置在哪裡？"
    python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "國立台灣師範大學的具體位置在哪裡？"
    python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Where is the National Taiwan Normal University?"
    python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --token "台"
    python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --token "臺"
    python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --token_id 120573
    python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Hello world" --no_model
"""

import argparse
import sys
from typing import List, Optional, Tuple

import torch

from model_utils import (
    DEFAULT_MODEL_ID,
    fix_ssl_certificates,
    load_base_model,
    load_tokenizer,
    resolve_device,
)

fix_ssl_certificates()

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


def _str2bool(value: str) -> bool:
    """Parse a CLI boolean string."""
    return value.lower() == "true"


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the tokenizer inspection script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inspect Qwen3 tokenizer tokenization and embeddings"
    )
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="Hugging Face model ID")
    parser.add_argument("--sentence", type=str, default=None,
                        help="Sentence to tokenize and inspect")
    parser.add_argument("--token", type=str, default=None,
                        help="Single token string to look up")
    parser.add_argument("--token_id", type=int, default=None,
                        help="Token id to look up")
    parser.add_argument("--load_in_4bit", type=_str2bool, default=False,
                        help="Load the model in 4-bit quantization")
    parser.add_argument("--torch_dtype", type=str, default="float32",
                        choices=["bfloat16", "float16", "float32"],
                        help="Computation precision")
    parser.add_argument("--device", type=str, default=None, choices=["gpu", "cpu"],
                        help="Compute device (auto-detected by default)")
    parser.add_argument("--vector_limit", type=int, default=8,
                        help="Number of vector elements to print per token (0 = all)")
    parser.add_argument("--no_model", action="store_true",
                        help="Show tokenization and ids only; do not load the model")
    args = parser.parse_args()

    if not args.sentence and args.token is None and args.token_id is None:
        parser.error("Provide at least one of --sentence, --token, or --token_id.")

    if args.vector_limit < 0:
        parser.error("--vector_limit must be >= 0 (0 prints the entire vector).")

    return args


def _tokenize_sentence(tokenizer, sentence: str) -> List[Tuple[str, int]]:
    """Split a sentence into ``(token_string, token_id)`` pairs.

    Args:
        tokenizer: The loaded tokenizer.
        sentence: Raw sentence text to tokenize.

    Returns:
        List of ``(token_string, token_id)`` pairs in token order.
    """
    tokens = tokenizer.tokenize(sentence)
    return [(token, tokenizer.convert_tokens_to_ids(token)) for token in tokens]


def _byte_token_reverse_map() -> dict:
    """Return the inverse of the GPT-style ``bytes_to_unicode`` mapping.

    Byte-level tokenizers map raw bytes outside the printable ASCII range to
    single Unicode characters (e.g. ``0x8f`` -> ``ı``). Reversing that map
    lets us recover the original UTF-8 bytes of a token for display.

    Returns:
        Dict mapping each byte-level token character back to its byte value.
    """
    kept = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    chars = kept[:]
    counter = 0
    for byte in range(2 ** 8):
        if byte not in kept:
            kept.append(byte)
            chars.append(2 ** 8 + counter)
            counter += 1
    return {chr(char): byte for byte, char in zip(kept, chars)}


def _display_token(token: str) -> str:
    """Repair a byte-level token string into readable text.

    Byte-level tokenizers store each UTF-8 byte as a single character, which
    prints as mojibake. This reverses the GPT-style ``bytes_to_unicode`` map
    and re-decodes the recovered bytes as UTF-8; plain tokens are returned
    unchanged.

    Args:
        token: Raw token string from the vocabulary.

    Returns:
        Readable token string for display.
    """
    reverse_map = _byte_token_reverse_map()
    bytes_out = bytearray()
    for char in token:
        if char in reverse_map:
            bytes_out.append(reverse_map[char])
        else:
            bytes_out.extend(char.encode("utf-8"))
    try:
        return bytes(bytes_out).decode("utf-8")
    except UnicodeDecodeError:
        return token


def _format_vector(vector: torch.Tensor, limit: int) -> str:
    """Format an embedding vector as a compact, human-readable string.

    Args:
        vector: Embedding row tensor for a single token.
        limit: Max number of elements to print (0 = all).

    Returns:
        Formatted string with values plus shape/dtype/norm statistics.
    """
    flat = vector.detach().float().flatten()
    total = flat.numel()
    if limit == 0 or limit >= total:
        limit = total
    values = ", ".join(f"{v:.4f}" for v in flat[:limit].tolist())
    ellipsis = ", ..." if limit < total else ""
    stats = (
        f"shape={tuple(vector.shape)} dtype={vector.dtype} "
        f"norm={flat.norm().item():.4f} mean={flat.mean().item():.4f} "
        f"std={flat.std().item():.4f}"
    )
    return f"[{values}{ellipsis}]  ({stats})"


def _embedding_row(embeddings, token_id: int) -> Optional[torch.Tensor]:
    """Return the embedding vector for a token id.

    Args:
        embeddings: The model's input embedding module, or None.
        token_id: Vocabulary id of the token.

    Returns:
        Embedding row tensor, or None when no embedding module is available.
    """
    if embeddings is None:
        return None
    return embeddings.weight[token_id]


def _print_embedding_info(embeddings) -> None:
    """Print a one-line summary of the model's token embedding layer.

    Args:
        embeddings: The model's input embedding module.
    """
    weight = embeddings.weight
    print(f"  Token embedding matrix: shape={tuple(weight.shape)} "
          f"dtype={weight.dtype} device={weight.device}")


def _print_token_row(idx: int, token: str, token_id: int,
                     vector: Optional[torch.Tensor], vector_limit: int) -> None:
    """Print one tokenization result with its id and optional vector.

    Args:
        idx: Position of the token in the sentence (0 for single lookups).
        token: Raw token string.
        token_id: Vocabulary id of the token.
        vector: Embedding vector for the token, or None when skipped.
        vector_limit: Max vector elements to print (0 = all).
    """
    print(f"  [{idx:<3}] token={_display_token(token)!r:<22} id={token_id}")
    if vector is not None:
        print(f"       vector = {_format_vector(vector, vector_limit)}")


def inspect_sentence(tokenizer, model, args: argparse.Namespace) -> None:
    """Print how a sentence is split into tokens plus their embeddings.

    Args:
        tokenizer: The loaded tokenizer.
        model: The loaded model, or None when ``--no_model`` is used.
        args: Parsed CLI arguments.
    """
    print(f"\n{'=' * 60}\nSENTENCE TOKENIZATION\n{'=' * 60}")
    print(f"  Sentence   : {args.sentence!r}")
    pairs = _tokenize_sentence(tokenizer, args.sentence)
    print(f"  Token count: {len(pairs)}")
    embeddings = None if model is None else model.get_input_embeddings()
    for idx, (token, token_id) in enumerate(pairs):
        _print_token_row(idx, token, token_id,
                         _embedding_row(embeddings, token_id), args.vector_limit)


def lookup_token(tokenizer, model, args: argparse.Namespace) -> None:
    """Look up a single token string and print its id and embedding vector.

    Args:
        tokenizer: The loaded tokenizer.
        model: The loaded model, or None when ``--no_model`` is used.
        args: Parsed CLI arguments.
    """
    print(f"\n{'=' * 60}\nTOKEN LOOKUP (string)\n{'=' * 60}")
    token = args.token
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_token = tokenizer.unk_token
    unk_id = tokenizer.unk_token_id
    not_found = (
        token_id is None
        or (unk_id is not None and token_id == unk_id and token != unk_token)
    )
    pieces = []
    if not_found:
        pieces = tokenizer.tokenize(token)
        if len(pieces) == 1:
            token = pieces[0]
            token_id = tokenizer.convert_tokens_to_ids(token)
            not_found = False
    if not_found:
        print(f"  Token {args.token!r} was not found as a single vocabulary entry.")
        print(f"  It splits into {len(pieces)} tokens instead; "
              f"use --sentence {args.token!r} to inspect them.")
        return
    embeddings = None if model is None else model.get_input_embeddings()
    _print_token_row(0, token, token_id,
                     _embedding_row(embeddings, token_id), args.vector_limit)


def lookup_token_id(tokenizer, model, args: argparse.Namespace) -> None:
    """Look up a token id and print its token string and embedding vector.

    Args:
        tokenizer: The loaded tokenizer.
        model: The loaded model, or None when ``--no_model`` is used.
        args: Parsed CLI arguments.
    """
    print(f"\n{'=' * 60}\nTOKEN LOOKUP (id)\n{'=' * 60}")
    token_str = tokenizer.convert_ids_to_tokens(args.token_id)
    if token_str is None or token_str == "":
        print(f"  Token id {args.token_id} has no assigned token string.")
        return
    embeddings = None if model is None else model.get_input_embeddings()
    _print_token_row(0, token_str, args.token_id,
                     _embedding_row(embeddings, args.token_id), args.vector_limit)


def main() -> None:
    """Orchestrate tokenizer inspection: parse args, load tokenizer, run lookups."""
    args = parse_args()
    device_map, _, load_in_4bit, compute_dtype = resolve_device(args)

    print(f"Loading tokenizer from {args.model_id}...")
    tokenizer = load_tokenizer(args.model_id)
    print(f"Tokenizer class: {tokenizer.__class__.__name__}, "
          f"vocab size: {len(tokenizer)}")

    model = None
    if not args.no_model:
        print(f"\nLoading model from {args.model_id}...")
        model = load_base_model(args.model_id, load_in_4bit, compute_dtype, device_map)
        embeddings = model.get_input_embeddings()
        if embeddings is None:
            print("  Warning: model exposes no input embeddings; "
                  "vectors will be skipped.")
            model = None
        else:
            _print_embedding_info(embeddings)
    else:
        print("  Skipping model load (--no_model); vectors will not be shown.")

    if args.sentence:
        inspect_sentence(tokenizer, model, args)
    if args.token is not None:
        lookup_token(tokenizer, model, args)
    if args.token_id is not None:
        lookup_token_id(tokenizer, model, args)


if __name__ == "__main__":
    main()
