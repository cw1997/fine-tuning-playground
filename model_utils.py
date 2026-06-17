"""
Model and tokenizer loading utilities for fine-tuning Qwen3-4B.

Handles quantized model loading, LoRA adapter configuration,
and parameter statistics reporting.
"""

import shutil
from typing import List, Optional, Tuple, Union

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

DeviceMap = Union[str, dict]


def resolve_compute_device(device: Optional[str] = None) -> Tuple[torch.device, DeviceMap, bool]:
    """
    Detect available hardware and pick the compute device.

    Args:
        device: Device preference ("gpu", "cpu", or None for auto).
                Auto prefers GPU when CUDA is available.

    Returns:
        Tuple of (torch.device, device_map, use_cuda).
    """
    if device == "cpu":
        return torch.device("cpu"), {"": "cpu"}, False

    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU was requested via --device gpu but CUDA is not available. "
                "Install a CUDA-enabled PyTorch wheel or use --device cpu."
            )
        return torch.device("cuda:0"), {"": 0}, True

    # Auto: prefer GPU when available
    if torch.cuda.is_available():
        return torch.device("cuda:0"), {"": 0}, True
    return torch.device("cpu"), {"": "cpu"}, False


def print_compute_device(device: Optional[str] = None) -> Tuple[torch.device, DeviceMap, bool]:
    """
    Print the selected compute device and return device metadata.

    Args:
        device: Device preference ("gpu", "cpu", or None for auto).

    Returns:
        Tuple of (torch.device, device_map, use_cuda).
    """
    device_obj, device_map, use_cuda = resolve_compute_device(device)
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"Using device: cuda:0 ({gpu_name}), {gpu_count} GPU(s) available")
    elif device == "cpu":
        print("Using device: cpu (--device cpu specified)")
    else:
        print("Using device: cpu (CUDA unavailable in current PyTorch runtime)")
        if torch.version.cuda is None or not torch.backends.cuda.is_built():
            print(
                "  Hint: current PyTorch build is CPU-only. "
                "Install a CUDA-enabled PyTorch wheel to train on GPU."
            )
        elif shutil.which("nvidia-smi"):
            print(
                "  Hint: NVIDIA driver is present but CUDA is not available to PyTorch. "
                "Check driver/CUDA compatibility."
            )
        else:
            print("  Hint: no CUDA-capable GPU driver was detected on this machine.")
    return device_obj, device_map, use_cuda


def adapt_settings_for_device(
    use_4bit: bool,
    torch_dtype: str,
    device: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Adjust quantization and dtype when only CPU is available.

    4-bit quantization via bitsandbytes requires CUDA; CPU falls back to float32.
    """
    _, _, use_cuda = resolve_compute_device(device)
    if use_cuda:
        return use_4bit, torch_dtype

    if use_4bit:
        print("  Note: 4-bit quantization requires GPU; using full-precision on CPU.")
    if torch_dtype != "float32":
        print(f"  Note: {torch_dtype} on CPU is slow; using float32 instead.")
    return False, "float32"


def load_tokenizer(model_id: str) -> PreTrainedTokenizer:
    """
    Load the tokenizer for a Qwen3 model and configure padding.

    Sets pad_token to eos_token to avoid tokenizer warnings during training.

    Args:
        model_id: Hugging Face model identifier (e.g. "Qwen/Qwen3-4B").

    Returns:
        A configured PreTrainedTokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_quantized_model(
    model_id: str,
    torch_dtype: str = "bfloat16",
    use_4bit: bool = True,
    device: Optional[str] = None,
) -> PreTrainedModel:
    """
    Load a Qwen model with optional 4-bit quantization (QLoRA).

    Args:
        model_id:   Hugging Face model identifier.
        torch_dtype: Computation dtype ("bfloat16", "float16", or "float32").
        use_4bit:   Enable 4-bit NF4 quantization via bitsandbytes.
        device:     Device preference ("gpu", "cpu", or None for auto).

    Returns:
        A causal LM model (e.g. Qwen3 or Qwen3.5), loaded in the specified precision.
    """
    _, device_map, _ = resolve_compute_device(device)
    use_4bit, torch_dtype = adapt_settings_for_device(use_4bit, torch_dtype, device)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model


def setup_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """
    Create a LoRA configuration for Qwen3 causal LM.

    Args:
        r:               LoRA rank.
        alpha:           LoRA scaling factor.
        dropout:         Dropout probability for LoRA layers.
        target_modules:  List of module names to attach adapters to.
                         Defaults to all attention + MLP projection layers.

    Returns:
        A LoraConfig instance ready for use with get_peft_model.
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def print_trainable_params(model) -> None:
    """
    Print total parameter count, trainable parameter count, and percentage.

    Args:
        model: A PEFT-wrapped model with trainable adapter parameters.
    """
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %:          {100 * trainable_params / total_params:.4f}%")
