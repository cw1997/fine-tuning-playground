"""
Data loading and formatting utilities for fine-tuning Qwen3-4B.

Supports Hugging Face datasets, local JSON/JSONL files,
and multiple input formats (ChatML, Alpaca, plain text).
"""

import json
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset


def load_dataset_from_hub(
    path: str,
    split: str = "train",
    subset: Optional[str] = None,
) -> Dataset:
    """
    Load a dataset from the Hugging Face Hub.

    Args:
        path: Dataset identifier on the Hub (e.g. "databricks/databricks-dolly-15k").
        split: Which split to load (e.g. "train", "test").
        subset: Optional subset or configuration name.

    Returns:
        A Hugging Face Dataset object.
    """
    if subset is None:
        return load_dataset(path, split=split)
    return load_dataset(path, subset, split=split)


def load_dataset_from_json(filepath: str) -> Dataset:
    """
    Load a dataset from a local JSON or JSONL file.

    Expects a list of records. Auto-detects JSON vs JSONL by file extension.

    Args:
        filepath: Path to the local file.

    Returns:
        A Hugging Face Dataset object.
    """
    if filepath.endswith(".jsonl"):
        records: List[Dict] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return Dataset.from_list(records)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return Dataset.from_list(data)
        elif isinstance(data, dict):
            return Dataset.from_dict(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {filepath}")


def convert_to_chat_format(
    example: Dict[str, str],
    fmt: str = "chat",
) -> List[Dict[str, str]]:
    """
    Convert a single data record into the ChatML messages format.

    Supported input formats:
        - chat:    Already in [{"role": ..., "content": ...}, ...] form.
        - alpaca:  Has "instruction", "input", "output" keys.
        - text:    Has a single "text" key (user/assistant split on separator).

    Args:
        example: A dictionary representing one training example.
        fmt:     The format of the input data ("chat", "alpaca", or "text").

    Returns:
        A list of message dicts with "role" and "content" keys.
    """
    if fmt == "chat":
        messages = example.get("messages", [])
        if not messages:
            raise ValueError("Chat format examples must include a non-empty 'messages' list.")
        return messages

    elif fmt == "alpaca":
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        if inp:
            user_content = f"{instruction}\n\n{inp}"
        else:
            user_content = instruction
        return [
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": output.strip()},
        ]

    elif fmt == "text":
        text = example.get("text", "")
        parts = text.split("<|im_end|>\n<|im_start|>")
        messages: List[Dict[str, str]] = []
        for i, part in enumerate(parts):
            # Clean up ChatML markers that may remain
            part = part.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
            # The pattern is alternation: user / assistant
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": part})
        return messages

    else:
        raise ValueError(f"Unsupported dataset format: {fmt}")


def format_chat_template(
    messages: List[Dict[str, str]],
    tokenizer,
    enable_thinking: bool = False,
) -> str:
    """
    Apply the model's chat template to a list of messages.

    Args:
        messages:       List of dicts with "role" and "content".
        tokenizer:      The model's tokenizer (must have apply_chat_template).
        enable_thinking: Whether to enable Qwen3's thinking mode.

    Returns:
        A string formatted with ChatML delimiters, ready for tokenization.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
