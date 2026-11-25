"""Shared model loading and device utilities for training and inference."""

import os
from typing import Optional

import certifi
import torch
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


def fix_ssl_certificates(verbose: bool = False) -> None:
    """Repair broken SSL CA bundle environment variables on Windows / conda.

    Args:
        verbose: When True, print a note for each repaired variable.
    """
    for ssl_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        ssl_val = os.environ.get(ssl_var)
        if ssl_val and not os.path.isfile(ssl_val):
            os.environ[ssl_var] = certifi.where()
            if verbose:
                print(f"  Note: {ssl_var} pointed to missing file; using certifi bundle.")


def load_tokenizer(model_id: str):
    """Load a tokenizer and ensure ``pad_token`` is set.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        Loaded tokenizer with ``pad_token`` set to ``eos_token`` when missing.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_quantization_config(
    load_in_4bit: bool,
    compute_dtype: torch.dtype,
) -> Optional[BitsAndBytesConfig]:
    """Build a 4-bit NF4 quantization config when requested.

    Args:
        load_in_4bit: Whether to enable 4-bit loading.
        compute_dtype: Computation dtype for quantized matmuls.

    Returns:
        ``BitsAndBytesConfig`` when ``load_in_4bit`` is True, otherwise None.
    """
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_base_model(
    model_id: str,
    load_in_4bit: bool,
    compute_dtype: torch.dtype,
    device_map: dict[str, str],
    *,
    prepare_kbit: bool = False,
    gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """Load a causal language model, optionally with 4-bit quantization.

    Args:
        model_id: Hugging Face model identifier.
        load_in_4bit: Whether to apply 4-bit quantization.
        compute_dtype: Target computation dtype.
        device_map: Device placement mapping.
        prepare_kbit: When True, call ``prepare_model_for_kbit_training`` after load.
        gradient_checkpointing: Whether PEFT should enable checkpointing during
            k-bit preparation. Ignored when preparation is disabled.

    Returns:
        PreTrainedModel: Loaded base model.

    Raises:
        OSError: If the model files cannot be loaded.
        ValueError: If the model, device map or quantization config is unsupported.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_quantization_config(load_in_4bit, compute_dtype),
        dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if load_in_4bit and prepare_kbit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing,
        )
    return model
