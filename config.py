"""
Configuration module for fine-tuning Qwen3-4B.

Defines FinetuneConfig, a dataclass that holds all hyperparameters
for model, data, LoRA, training, and inference settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FinetuneConfig:
    """
    Central configuration container for the Qwen3-4B fine-tuning pipeline.

    Attributes:
        model_id: Hugging Face model identifier.
        load_in_4bit: Whether to use 4-bit QLoRA quantization.
        torch_dtype: Computation dtype (bfloat16, float16, or float32).
        lora_r: LoRA rank dimension.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to attach LoRA adapters to.
        dataset_path: Path to local dataset file or Hugging Face dataset name.
        dataset_format: Format of the dataset (chat, alpaca, or text).
        test_split: Fraction of data reserved for evaluation.
        max_seq_length: Maximum sequence length for tokenization.
        learning_rate: Peak learning rate for the optimizer.
        num_epochs: Number of training epochs.
        per_device_batch_size: Batch size per GPU.
        gradient_accum_steps: Gradient accumulation steps.
        warmup_ratio: Fraction of training steps for linear warmup.
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        output_dir: Directory to save model checkpoints and adapter.
        push_to_hub: Whether to upload the adapter to Hugging Face Hub.
        hub_model_id: Target hub repository name (required if push_to_hub is True).
        max_new_tokens: Maximum tokens to generate during inference.
        temperature: Sampling temperature for generation.
        top_p: Nucleus sampling probability threshold.
        use_thinking: Enable Qwen3 thinking mode in chat template.
        device: Compute device preference ("gpu", "cpu", or None for auto).
    """

    # --- Model settings ---
    model_id: str = "Qwen/Qwen3.5-0.8B" # use "Qwen/Qwen3.5-4B" on GPU, "Qwen/Qwen3.5-0.8B" on CPU
    load_in_4bit: bool = True
    torch_dtype: str = "bfloat16"

    # --- LoRA settings ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    # --- Data settings ---
    dataset_path: str = ""
    dataset_format: str = "chat"
    test_split: float = 0.05
    max_seq_length: int = 4096

    # --- Training settings ---
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accum_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    output_dir: str = "./models/qwen3-4b-finetuned"

    # --- Hub settings ---
    push_to_hub: bool = False
    hub_model_id: str = ""

    # --- Inference settings ---
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    use_thinking: bool = False

    # --- Hardware settings ---
    device: Optional[str] = None

    def __post_init__(self) -> None:
        """Set default target modules if none provided."""
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

    @classmethod
    def from_args(cls, args) -> "FinetuneConfig":
        """
        Build a FinetuneConfig from an argparse.Namespace.

        Args:
            args: Namespace returned by argparse.parse_args().

        Returns:
            A FinetuneConfig instance populated with CLI arguments.
        """
        kwargs = {k: v for k, v in vars(args).items() if v is not None}
        return cls(**kwargs)
