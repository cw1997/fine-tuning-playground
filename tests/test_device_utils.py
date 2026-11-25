"""Regression tests for backend selection without accelerator hardware."""

from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from types import ModuleType
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from src import device_utils


class TestDeviceSelection(TestCase):
    """Verify the CLI device and precision policies with simulated GPUs."""

    def setUp(self) -> None:
        """Start with a CPU build and stub the available GPU APIs.

        Returns:
            None.
        """
        out_of_memory_error = torch.cuda.OutOfMemoryError
        self.cuda = self.enterContext(patch.object(torch, "cuda", Mock()))
        self.cuda.OutOfMemoryError = out_of_memory_error
        self.mps = self.enterContext(patch.object(torch.backends, "mps", Mock()))
        self.xpu = self.enterContext(patch.object(torch, "xpu", Mock(), create=True))
        self.torch_version = self.enterContext(
            patch.object(torch, "version", Mock(hip=None)),
        )
        for accelerator in (self.cuda, self.mps, self.xpu):
            accelerator.is_available.return_value = False
            accelerator.get_device_name.return_value = "Test GPU"
            accelerator.is_bf16_supported.return_value = True
        self.mps.is_macos_or_newer.return_value = True
        self.enterContext(redirect_stdout(StringIO()))

    def test_auto_and_gpu_prefer_available_accelerators(self) -> None:
        """Distinguish CUDA from HIP and then try MPS, XPU, and CPU.

        Returns:
            None.
        """
        for accelerator in (self.cuda, self.mps, self.xpu):
            accelerator.is_available.return_value = True
        self.assertEqual(device_utils.select_device("auto")[0], "cuda")
        self.torch_version.hip = "7.2"
        self.assertEqual(
            device_utils.select_device("gpu"), ("rocm", torch.device("cuda:0")),
        )
        self.cuda.is_available.return_value = False
        self.assertEqual(device_utils.select_device("auto")[0], "mps")
        self.mps.is_available.return_value = False
        self.assertEqual(
            device_utils.select_device("auto"), ("xpu", torch.device("xpu:0")),
        )
        self.xpu.is_available.return_value = False
        self.assertEqual(device_utils.select_device("auto")[0], "cpu")

    def test_forced_cpu_does_not_probe_gpu(self) -> None:
        """Keep explicit CPU selection even on a GPU system.

        Returns:
            None.
        """
        self.cuda.is_available.return_value = True
        args = Namespace(device="cpu", torch_dtype="float16", load_in_4bit=True)
        config = device_utils.resolve_device(args, training=True)
        self.assertEqual(config.dtype, torch.float32)
        self.assertFalse(config.load_in_4bit)
        self.cuda.is_available.assert_not_called()

    def test_unavailable_gpu_or_vendor_is_an_error(self) -> None:
        """Require a GPU or the exact backend when requested explicitly.

        Returns:
            None.
        """
        for backend in ("gpu", "cuda", "rocm", "mps", "xpu"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(RuntimeError, f"--device {backend}"):
                    device_utils.select_device(backend)
        self.cuda.is_available.return_value = True
        self.torch_version.hip = "7.2"
        with self.assertRaisesRegex(RuntimeError, "--device cuda"):
            device_utils.select_device("cuda")
        self.torch_version.hip = None
        with self.assertRaisesRegex(RuntimeError, "--device rocm"):
            device_utils.select_device("rocm")

    def test_cuda_nf4_default_and_effective_precision(self) -> None:
        """Keep the CUDA NF4 default while adapting to an older GPU's precision.

        Returns:
            None.
        """
        self.cuda.is_available.return_value = True
        self.cuda.is_bf16_supported.return_value = False
        args = Namespace(device="cuda", torch_dtype="bfloat16", load_in_4bit=None)
        config = device_utils.resolve_device(args, training=True)
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.dtype, torch.float16)
        self.assertEqual(config.dtype_name, "float16")
        self.assertEqual(config.device_map, {"": "cuda:0"})

    def test_rocm_nf4_opt_in_uses_selected_precision(self) -> None:
        """Use FP16 when ROCm cannot execute BF16 and validate explicit NF4.

        Returns:
            None.
        """
        self.cuda.is_available.return_value = True
        self.torch_version.hip = "7.2"
        args = Namespace(device="rocm", torch_dtype="bfloat16", load_in_4bit=None)
        with (
            patch.object(torch, "ones", side_effect=RuntimeError("no BF16 kernel")),
            patch.object(device_utils, "validate_rocm_4bit") as probe,
        ):
            config = device_utils.resolve_device(args, training=True)
            self.assertFalse(config.load_in_4bit)
            self.assertEqual(config.dtype, torch.float16)
            probe.assert_not_called()
            args.load_in_4bit = True
            config = device_utils.resolve_device(args, training=True)
            self.assertTrue(config.load_in_4bit)
            self.assertEqual(config.dtype, torch.float16)
            probe.assert_called_once_with(config, training=True)

    def test_mps_and_xpu_load_unquantized_weights(self) -> None:
        """Exercise the portable defaults and reject an explicit NF4 request.

        Returns:
            None.
        """
        self.mps.is_available.return_value = True
        self.xpu.is_available.return_value = True
        for backend in ("mps", "xpu"):
            with self.subTest(backend=backend):
                args = Namespace(
                    device=backend, torch_dtype="bfloat16", load_in_4bit=None,
                )
                self.assertFalse(device_utils.resolve_device(args).load_in_4bit)
                args.load_in_4bit = True
                with self.assertRaisesRegex(RuntimeError, "--load_in_4bit False"):
                    device_utils.resolve_device(args)

    def test_mps_training_uses_precision_supported_by_pytorch(self) -> None:
        """Resolve the macOS BF16 and PyTorch mixed precision requirements.

        Returns:
            None.
        """
        self.mps.is_available.return_value = True
        args = Namespace(device="mps", torch_dtype="bfloat16", load_in_4bit=None)
        with patch.object(torch, "__version__", "2.5.1"):
            self.assertEqual(device_utils.resolve_device(args).dtype, torch.bfloat16)
            self.assertEqual(
                device_utils.resolve_device(args, training=True).dtype, torch.float32,
            )
            self.mps.is_macos_or_newer.return_value = False
            self.assertEqual(
                device_utils.resolve_device(args, training=True).dtype, torch.float16,
            )

    def test_rocm_probe_reports_bad_native_library(self) -> None:
        """Surface the root cause and an explicit unquantized fallback.

        Returns:
            None.
        """
        self.torch_version.hip = "7.2"
        bnb = ModuleType("bitsandbytes")
        bnb_nn = ModuleType("bitsandbytes.nn")
        bnb_nn.Linear4bit = Mock(side_effect=OSError("bad native library"))
        config = device_utils.DeviceConfig(
            "rocm", torch.device("cuda:0"), torch.float16, True,
        )
        modules = {"bitsandbytes": bnb, "bitsandbytes.nn": bnb_nn}
        with patch.dict("sys.modules", modules):
            with self.assertRaisesRegex(RuntimeError, "--load_in_4bit False") as error:
                device_utils.validate_rocm_4bit(config, training=False)
        self.assertIsInstance(error.exception.__cause__, OSError)

    def test_rocm_preflight_runs_backward_for_training(self) -> None:
        """Exercise the preflight with a stand-in linear layer on CPU.

        Returns:
            None.
        """
        bnb = ModuleType("bitsandbytes")
        bnb_nn = ModuleType("bitsandbytes.nn")
        layer = torch.nn.Linear(64, 64, bias=False)
        bnb_nn.Linear4bit = Mock(return_value=layer)
        config = device_utils.DeviceConfig(
            "rocm", torch.device("cpu"), torch.float32, True,
        )
        modules = {"bitsandbytes": bnb, "bitsandbytes.nn": bnb_nn}
        with patch.dict("sys.modules", modules):
            device_utils.validate_rocm_4bit(config, training=True)
        self.assertFalse(layer.weight.requires_grad)
        self.assertEqual(bnb_nn.Linear4bit.call_args.kwargs["quant_type"], "nf4")
        self.cuda.synchronize.assert_called_once_with(config.device)
