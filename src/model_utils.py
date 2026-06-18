"""Shared model loading and device utilities for training and inference."""

import argparse
import os
import shutil
from typing import Optional, Tuple

import certifi
import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


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


def resolve_device(
    args: argparse.Namespace,
) -> Tuple[dict, bool, bool, torch.dtype]:
    """Detect compute device and adjust quantization / dtype for the platform.

    Args:
        args: Parsed CLI arguments with ``device``, ``load_in_4bit``, and
            ``torch_dtype`` attributes.

    Returns:
        Tuple of ``(device_map, use_cuda, load_in_4bit, compute_dtype)``.
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
                print(
                    "  Hint: The current PyTorch is a CPU build. "
                    "Install CUDA-enabled PyTorch for GPU training."
                )
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

    compute_dtype = _DTYPE_MAP[dtype_str]
    return device_map, use_cuda, load_in_4bit, compute_dtype


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
    device_map: dict,
    *,
    prepare_kbit: bool = False,
):
    """Load a causal language model, optionally with 4-bit quantization.

    Args:
        model_id: Hugging Face model identifier.
        load_in_4bit: Whether to apply 4-bit quantization.
        compute_dtype: Target computation dtype.
        device_map: Device placement mapping.
        prepare_kbit: When True, call ``prepare_model_for_kbit_training`` after load.

    Returns:
        Loaded base model.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_quantization_config(load_in_4bit, compute_dtype),
        dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if load_in_4bit and prepare_kbit:
        model = prepare_model_for_kbit_training(model)
    return model
