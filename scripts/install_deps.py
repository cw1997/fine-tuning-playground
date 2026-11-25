"""Install project dependencies using the active Python and a selected backend."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from platform import machine, system

ROOT = Path(__file__).resolve().parents[1]
COMMON_REQUIREMENTS = ROOT / "requirements-common.txt"
PYPI_INDEX = "https://pypi.org/simple"
TORCH_WHEEL_INDEXES = {
    "cuda": "https://download.pytorch.org/whl/cu128",
    "mps": PYPI_INDEX,
    "xpu": "https://download.pytorch.org/whl/xpu",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
TORCH_PROBE = """
import json
import sys
from importlib.metadata import version

import torch

from device_utils import select_device

backend, device = select_device(sys.argv[1])
print(json.dumps({
    "torch": version("torch"), "backend": backend, "device": str(device),
    "cuda": torch.version.cuda, "hip": torch.version.hip,
    "verified": (torch.ones(1, device=device) + 1).item() == 2,
}))
"""


def pip_install(*args: str) -> None:
    """Install packages into the environment running this script.

    Args:
        *args (str): Arguments for pip install.

    Returns:
        None.

    Raises:
        subprocess.CalledProcessError: If pip fails to install the packages.
    """
    subprocess.run([sys.executable, "-m", "pip", "install", *args], check=True)


def verify_torch(backend: str) -> str:
    """Verify a tensor runs on the selected backend and obtain the wheel version.

    Args:
        backend (str): Backend that must be available to PyTorch.

    Returns:
        str: Installed PyTorch distribution version, including its local tag.

    Raises:
        RuntimeError: If the selected backend cannot execute the probe.
    """
    probe = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", TORCH_PROBE, backend],
        cwd=ROOT / "src", capture_output=True, text=True, encoding="utf-8",
    )
    if probe.returncode:
        raise RuntimeError(
            f"PyTorch verification failed for {backend}. {probe.stderr.strip()}"
        )
    details = json.loads(probe.stdout.splitlines()[-1])
    if not details["verified"]:
        raise RuntimeError(f"PyTorch produced an unexpected result on {backend}.")
    print(f"Verified PyTorch {details['torch']} on {backend} ({details['device']}) "
          f"with CUDA={details['cuda']}, HIP={details['hip']}", flush=True)
    return details["torch"]


def main() -> None:
    """Install the selected PyTorch build and the project's dependencies.

    Returns:
        None.

    Raises:
        SystemExit: If the requested installation cannot be completed.
    """
    is_macos = system() == "Darwin"
    is_apple_silicon = is_macos and machine() == "arm64"
    default = ("mps" if is_apple_silicon else "cpu") if is_macos else "cuda"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "rocm", "mps", "xpu", "cpu"),
                        default=default, help=f"PyTorch backend (default: {default})")
    parser.add_argument("--rocm-4bit", action="store_true",
                        help="Install ROCm-compatible bitsandbytes for optional NF4")
    args = parser.parse_args()
    if args.rocm_4bit and args.backend != "rocm":
        parser.error("--rocm-4bit requires --backend rocm.")
    if args.backend == "mps" and not is_apple_silicon:
        parser.error("MPS requires Apple Silicon macOS and native arm64 Python.")

    try:
        if args.backend != "rocm":
            index = TORCH_WHEEL_INDEXES[args.backend]
            if is_macos and args.backend == "cpu":
                index = PYPI_INDEX
            pip_install("--index-url", index, "-c", str(COMMON_REQUIREMENTS), "torch")
        torch_version = verify_torch(args.backend)
        requirements = ROOT / "requirements.txt" if args.backend == "cuda" else (
            COMMON_REQUIREMENTS
        )
        # Keep the chosen vendor wheel when pip resolves transitive dependencies.
        dependencies = ["-r", str(requirements), f"torch=={torch_version}"]
        if args.rocm_4bit:
            dependencies.append("bitsandbytes>=0.50.0")
        pip_install(*dependencies)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
