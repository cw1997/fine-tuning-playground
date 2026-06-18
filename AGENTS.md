# Project Rules

This file defines conventions and rules for AI agents working on this project.

## Language & Comments

- All code comments MUST be in **standard English**
- Inline comments should explain *why*, not *what* (the code itself shows *what*)
- All functions MUST have a **docblock comment** (Google-style or NumPy-style) describing:
  - Purpose of the function
  - All arguments (name, type, description)
  - Return value and type
  - Any exceptions raised

## Python Best Practices

### Style & Formatting
- Follow **PEP 8** conventions (4-space indentation, line length ≤ 88)
- Use **type hints** for all function signatures
- Prefer `pathlib.Path` over `os.path` for path manipulation
- Use f-strings for string formatting (not `%` or `.format()`)

### Imports
- Order: standard library → third-party → local modules (groups separated by blank line)
- Avoid `from x import *`
- Import specific names, not modules, when using frequently (e.g., `from pathlib import Path`)

### Functions & Variables
- Use descriptive, snake_case names for functions and variables
- Use UPPER_CASE for module-level constants
- Prefer local variables over global state
- Keep functions focused on a single responsibility

### Error Handling
- Raise specific exception types (not bare `raise Exception`)
- Provide informative error messages
- Use `warnings.filterwarnings` to suppress known benign warnings (e.g., `_check_is_size`, `triton not found`)

### Dependencies
- Pin minimum versions in `requirements.txt`, not exact versions
- Prefer `>=` constraints to allow patch updates
- Use the PyTorch index (`https://download.pytorch.org/whl/cu128`) for CUDA-enabled wheels

## Known Deprecations & Fixes

### `torch_dtype` → `dtype`
In `transformers` `from_pretrained()` calls, use `dtype=compute_dtype` instead of `torch_dtype=compute_dtype`:
```python
# Correct
model = AutoModelForCausalLM.from_pretrained(..., dtype=compute_dtype)

# Wrong
model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=compute_dtype)
```

### `_check_is_size` FutureWarning
Suppress with:
```python
warnings.filterwarnings("ignore", message=".*_check_is_size.*")
```

### Triton not found
Suppress FLOP counter warning on systems without Triton:
```python
warnings.filterwarnings("ignore", message=".*triton not found.*")
```

## Project Architecture

```
fine-tuning-playground/
├── src/                    # Source code
│   ├── train.py            # SFT training pipeline (single-file, CLI-driven)
│   └── inference.py        # Inference script (base / finetuned / compare modes)
├── data/                   # Datasets and generators
│   ├── generate_ntnu_dataset.py  # Dataset generator script
│   ├── ntnu_extended_records.py  # Extended Q&A records module
│   └── ntnu_dataset.jsonl       # Bundled dataset (161+ ChatML records)
├── scripts/                # Installation helpers (shell/PowerShell)
└── models/                 # Fine-tuned LoRA adapters (gitignored)
```

## Tech Stack

- **Model**: Qwen3 / Qwen3.5 (decoder-only, Hugging Face)
- **Training**: `transformers` + `trl.SFTTrainer` + `peft` (LoRA/QLoRA)
- **Quantization**: `bitsandbytes` (4-bit NF4)
- **Hardware**: CUDA GPU (recommended) / CPU fallback
- **Packages**: See `requirements.txt`

## Adding New Features

1. Maintain the single-file philosophy unless complexity warrants modularization
2. Add CLI argument for every new hyperparameter
3. Write Google-style docstrings for all new functions
4. Keep comments in English only
