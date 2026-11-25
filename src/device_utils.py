"""Resolve PyTorch backends, precision, and 4-bit support for the CLIs."""

import argparse
from dataclasses import dataclass

import torch

GPU_BACKENDS = ("cuda", "rocm", "mps", "xpu")
BACKEND_HINTS = {
    "cuda": "Install cu128 PyTorch and check the NVIDIA driver.",
    "rocm": "Install AMD's ROCm PyTorch wheel for your GPU, OS, and driver.",
    "mps": "Use native arm64 Python and a macOS wheel with MPS support.",
    "xpu": "Install PyTorch from the XPU index and check the Intel GPU driver.",
}
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class DeviceConfig:
    """Resolved backend, PyTorch device, model dtype, and NF4 setting."""

    backend: str
    device: torch.device
    dtype: torch.dtype
    load_in_4bit: bool

    @property
    def device_map(self) -> dict[str, str]:
        """Place all model weights on the selected device.

        Returns:
            dict[str, str]: Hugging Face device map for the entire model.
        """
        return {"": str(self.device)}

    @property
    def dtype_name(self) -> str:
        """Return the effective precision for logs and serialized results.

        Returns:
            str: The CLI name for the effective dtype.
        """
        return str(self.dtype).removeprefix("torch.")

    @property
    def device_name(self) -> str:
        """Identify the device in logs and metrics.

        Returns:
            str: The GPU name when PyTorch provides it, or a backend label.
        """
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        if self.device.type == "xpu":
            return torch.xpu.get_device_name(self.device)
        return "Apple Metal" if self.backend == "mps" else "cpu"


def add_device_argument(parser: argparse.ArgumentParser, default: str = "auto") -> None:
    """Expose the same device selection flags across the project.

    Args:
        parser (argparse.ArgumentParser): Parser that accepts the device choice.
        default (str): Device choice when the user omits the flag.

    Returns:
        None.
    """
    parser.add_argument(
        "--device", choices=("auto", "gpu", *GPU_BACKENDS, "cpu"), default=default,
        help=f"Compute backend (default: {default}); gpu requires an available GPU",
    )


def parse_4bit(value: str) -> bool:
    """Parse the explicit 4-bit choice used by the model CLI flags.

    Args:
        value (str): Case-insensitive ``True`` or ``False``.

    Returns:
        bool: Whether the user requested 4-bit loading.

    Raises:
        argparse.ArgumentTypeError: If the value is neither ``True`` nor ``False``.
    """
    normalized = value.lower()
    if normalized not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected True or False.")
    return normalized == "true"


def select_device(requested: str) -> tuple[str, torch.device]:
    """Select an available PyTorch device for an explicit backend or automatic mode.

    Args:
        requested (str): Backend, ``auto``, or ``gpu``.

    Returns:
        tuple[str, torch.device]: Backend label and the corresponding PyTorch device.

    Raises:
        RuntimeError: If the requested accelerator is unavailable.
        ValueError: If the selection is unknown.
    """
    if requested == "cpu":
        return "cpu", torch.device("cpu")
    if requested not in ("auto", "gpu", *GPU_BACKENDS):
        raise ValueError(f"Unknown device: {requested}")

    candidates = GPU_BACKENDS if requested in ("auto", "gpu") else (requested,)
    hip = bool(getattr(torch.version, "hip", None))
    for backend in candidates:
        if backend in ("cuda", "rocm"):
            if hip == (backend == "rocm") and torch.cuda.is_available():
                return backend, torch.device("cuda:0")
        elif backend == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return backend, torch.device("mps")
        else:
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                return backend, torch.device("xpu:0")

    if requested == "auto":
        return "cpu", torch.device("cpu")
    hint = BACKEND_HINTS.get(requested, "Install PyTorch for CUDA, ROCm, MPS, or XPU.")
    raise RuntimeError(
        f"--device {requested} is unavailable. {hint} Or use --device cpu."
    )


def _resolve_dtype(
    backend: str, device: torch.device, requested: str, training: bool,
) -> torch.dtype:
    """Resolve model precision against the selected backend's capabilities.

    Args:
        backend (str): Selected PyTorch backend label.
        device (torch.device): Device used for runtime capability checks.
        requested (str): Requested dtype name.
        training (bool): Whether the dtype must support Trainer mixed precision.

    Returns:
        torch.dtype: Effective model dtype.

    Raises:
        KeyError: If the requested dtype name is unknown.
        torch.cuda.OutOfMemoryError: If the ROCm precision probe runs out of memory.
    """
    dtype = DTYPES[requested]
    if backend == "cpu" or dtype == torch.float32:
        return torch.float32
    if backend == "mps":
        # The installer imports device selection before installing shared dependencies.
        from packaging.version import Version

        if dtype == torch.bfloat16 and not torch.backends.mps.is_macos_or_newer(14, 0):
            dtype = torch.float16
        minimum = "2.6" if dtype == torch.bfloat16 else "2.5"
        if training and Version(torch.__version__) < Version(minimum):
            return torch.float32
    elif dtype == torch.bfloat16 and backend == "rocm":
        # PyTorch's CUDA capability helper reports BF16 for every HIP build.
        try:
            matrix = torch.ones((16, 16), dtype=dtype, device=device)
            (matrix @ matrix)[0, 0].item()
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError:
            return torch.float16
    elif dtype == torch.bfloat16:
        accelerator = torch.xpu if backend == "xpu" else torch.cuda
        if not accelerator.is_bf16_supported():
            return torch.float16
    return dtype


def validate_rocm_4bit(config: DeviceConfig, *, training: bool) -> None:
    """Exercise the installed ROCm NF4 kernels before loading a large model.

    Args:
        config (DeviceConfig): Selected ROCm device and compute dtype.
        training (bool): Whether to verify gradients through the quantized layer.

    Returns:
        None.

    Raises:
        RuntimeError: If the bitsandbytes install or GPU kernels are incompatible.
    """
    try:
        from bitsandbytes.nn import Linear4bit

        with torch.random.fork_rng(devices=[]):
            layer = Linear4bit(
                64, 64, bias=False, compute_dtype=config.dtype,
                compress_statistics=True, quant_type="nf4",
            ).requires_grad_(False).to(config.device)
        inputs = torch.ones(
            (2, 64), device=config.device, dtype=config.dtype, requires_grad=training,
        )
        with torch.set_grad_enabled(training):
            output = layer(inputs)
            if training:
                output.sum().backward()
        if not torch.isfinite(output).all().item():
            raise RuntimeError("NF4 produced non-finite values.")
        if training and (
            inputs.grad is None or not torch.isfinite(inputs.grad).all().item()
        ):
            raise RuntimeError("NF4 produced missing or non-finite gradients.")
        torch.cuda.synchronize(config.device)
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as error:
        raise RuntimeError(
            f"ROCm NF4 failed on {config.device_name} (HIP {torch.version.hip}, "
            f"{config.dtype_name}). Install bitsandbytes>=0.50.0 with kernels for "
            "your ROCm release and GPU, or use --load_in_4bit False."
        ) from error


def resolve_device(args: argparse.Namespace, *, training: bool = False) -> DeviceConfig:
    """Resolve CLI device, precision, and quantization options before model loading.

    Args:
        args (argparse.Namespace): Device, torch_dtype, and optional NF4 choice.
        training (bool): Whether the configuration will be used for training.

    Returns:
        DeviceConfig: Effective device and model loading settings.

    Raises:
        RuntimeError: If the device or explicitly requested GPU NF4 is unavailable.
    """
    backend, device = select_device(args.device or "auto")
    load_in_4bit = args.load_in_4bit
    if load_in_4bit is None:
        load_in_4bit = backend == "cuda"
    if load_in_4bit and backend in ("mps", "xpu"):
        raise RuntimeError(
            f"4-bit loading is unavailable on {backend}. Use --load_in_4bit False."
        )
    if load_in_4bit and backend == "cpu":
        print("  Note: Using unquantized weights on CPU.")
        load_in_4bit = False

    dtype = _resolve_dtype(backend, device, args.torch_dtype, training)
    config = DeviceConfig(backend, device, dtype, load_in_4bit)
    quantization = "NF4" if config.load_in_4bit else "unquantized"
    description = "cpu" if backend == "cpu" else (
        f"{backend} ({device}, {config.device_name})"
    )
    print(f"Using device: {description}, {config.dtype_name}, {quantization}")
    if config.dtype_name != args.torch_dtype:
        print(f"  Note: Using {config.dtype_name} instead of {args.torch_dtype}.")
    if backend == "rocm" and load_in_4bit:
        validate_rocm_4bit(config, training=training)
    return config
