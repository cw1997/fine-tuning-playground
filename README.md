# Fine-Tuning Playground

A teaching-oriented codebase for supervised fine-tuning (SFT) of **Qwen3 / Qwen3.5** decoder-only models (e.g. **Qwen3-4B**, **Qwen3.5-4B**) using **QLoRA / LoRA**. Built with Hugging Face `transformers`, `peft`, `trl`, and `bitsandbytes`, the entire training pipeline lives in a single top-to-bottom script (`src/train.py`) with all settings passed as CLI flags.

---

## Table of Contents

- [Fine-Tuning Playground](#fine-tuning-playground)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Hardware Requirements](#hardware-requirements)
  - [Installation](#installation)
    - [Option 1 — Miniconda (Recommended)](#option-1--miniconda-recommended)
    - [Option 2 — venv](#option-2--venv)
    - [Helper Scripts](#helper-scripts)
  - [Quick Start](#quick-start)
  - [Project Structure](#project-structure)
    - [File Descriptions](#file-descriptions)
  - [Configuration](#configuration)
  - [Recommended Training Presets](#recommended-training-presets)
    - [Platform & OS](#platform--os)
    - [VRAM and Model Size (QLoRA)](#vram-and-model-size-qlora)
    - [Dataset Size Guidelines](#dataset-size-guidelines)
    - [Named Presets](#named-presets)
    - [Preset Scripts](#preset-scripts)
    - [Manual Commands by Preset](#manual-commands-by-preset)
  - [Data Formats](#data-formats)
    - [ChatML Format](#chatml-format)
    - [Alpaca Format](#alpaca-format)
    - [Text Format](#text-format)
  - [Usage](#usage)
    - [Basic Training](#basic-training)
    - [Training with Alpaca Dataset](#training-with-alpaca-dataset)
    - [Training from Hugging Face Hub](#training-from-hugging-face-hub)
    - [Inference — Interactive (Default)](#inference--interactive-default)
    - [Inference — One-Shot (`--no_interactive`)](#inference--one-shot---no_interactive)
    - [Inference with Thinking Mode](#inference-with-thinking-mode)
  - [Advanced Usage](#advanced-usage)
    - [Custom LoRA Target Modules](#custom-lora-target-modules)
    - [Push to Hugging Face Hub](#push-to-hugging-face-hub)
    - [Full vs QLoRA](#full-vs-qlora)
    - [Enable Thinking Mode During Training](#enable-thinking-mode-during-training)
  - [Output Artifacts](#output-artifacts)
  - [Tips \& Best Practices](#tips--best-practices)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)

---

## Overview

**Qwen3** and **Qwen3.5** are dense language models by Alibaba's Qwen team. The 4B variants support long context, grouped-query attention, and an optional chain-of-thought **thinking mode** (via `<think>` tags in the chat template).

This project provides a complete fine-tuning pipeline that lets you adapt these models to your own tasks and domains. It uses:

- **QLoRA** (4-bit NormalFloat quantization) to reduce memory footprint to ~6 GB VRAM, making fine-tuning accessible on consumer GPUs like the RTX 3060 / 4060 / 4070.
- **`trl.SFTTrainer`** for supervised fine-tuning with chat template integration and evaluation loop.
- **All 7 linear projection layers** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) as LoRA targets for optimal adaptation on a 4B-scale model.

---

## Features

- **QLoRA / LoRA support** — train with 4-bit quantization to fit on 6 GB GPUs, or disable quantization for full FP16 LoRA.
- **Multiple data formats** — ChatML (native), Alpaca (instruction/input/output), and plain text.
- **Hugging Face Hub integration** — load datasets directly from the Hub, and push trained adapters back.
- **Thinking mode** — optionally enable Qwen3's `<think>` reasoning tags during training and inference.
- **CLI-first design** — all hyperparameters configurable via command-line flags; no YAML files needed.
- **CPU fallback** — when CUDA is unavailable, training and inference automatically fall back to full-precision CPU (slow; intended for smoke tests).
- **Single-file training script** — `src/train.py` runs the full SFT pipeline sequentially, ideal for walking through each step when teaching fine-tuning.
- **Pre / post fine-tuning comparison** — test the base model before fine-tuning, then compare responses side by side with the fine-tuned model to measure improvement.
- **Interactive inference** — `src/inference.py` keeps the model loaded and accepts prompts in a loop by default; use `--no_interactive` for one-shot batch runs.
- **Evaluation split** — automatic train/test split with best-model selection based on eval loss.
- **Gradient checkpointing** — enabled by default to reduce memory usage during training.

---

## Hardware Requirements

| Configuration | Minimum VRAM | Typical GPU |
|---|---|---|
| QLoRA 4B (4-bit, batch size 1–2) | ~6–8 GB | RTX 3060 / 4060 / 4070, T4 |
| QLoRA 9B (4-bit, batch size 1) | ~10–12 GB | RTX 4070 Ti / 5070 Ti (mobile), 3080 12 GB |
| QLoRA 9B (4-bit, batch size 2) | ~20–24 GB | RTX 4090 / 3090 24 GB, A5000 |
| LoRA 4B (16-bit, batch size 1) | ~12 GB | RTX 3080 / 4070 Ti / 4080, A10 |
| CPU (no quantization, float32) | — | Any CPU (very slow; smoke tests only) |

GPU training is strongly recommended. On machines without CUDA, the pipeline auto-disables 4-bit quantization and uses `float32` on CPU. Use `--device gpu` or `--device cpu` to override auto-detection. See [Recommended Training Presets](#recommended-training-presets) for ready-made commands per GPU tier.

---

## Installation

Use **Python 3.11 or 3.12** for best CUDA wheel compatibility. Newer Python versions may only receive CPU-only PyTorch wheels from PyPI.

### Option 1 — Miniconda (Recommended)

```bash
# Clone the repository
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

# Create and activate a conda environment
conda create -n finetune python=3.11 -y
conda activate finetune

# Install dependencies
pip install -r requirements.txt
```

### Option 2 — venv

```bash
# Clone the repository
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux / macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Helper Scripts

```bash
# Auto-installs CUDA-enabled dependencies
bash scripts/install_deps.sh                 # Linux / macOS / Git Bash
powershell -File scripts/install_deps.ps1    # Windows PowerShell

# List training presets (VRAM / model / dataset size)
bash scripts/train_preset.sh --list          # Linux / macOS / Git Bash
powershell -File scripts/train_preset.ps1 -List
```

Verify GPU is visible to PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`requirements.txt` includes CUDA PyTorch (cu128) and all project dependencies. For CPU-only fallback (very slow for 4B models), install PyTorch from PyPI first, then the remaining packages without `bitsandbytes`.

Core packages:

| Package | Minimum Version | Purpose |
|---|---|---|
| `torch` | >= 2.4.0, < 2.12 | Deep learning framework (CUDA cu128 via `requirements.txt`) |
| `transformers` | >= 4.47.0 | Model loading and tokenization (Qwen3 support) |
| `peft` | >= 0.13.0 | LoRA / QLoRA adapter configuration |
| `trl` | >= 0.9.0 | SFTTrainer for supervised fine-tuning |
| `accelerate` | >= 1.0.0 | Distributed training support |
| `bitsandbytes` | >= 0.43.0 | 4-bit quantization (QLoRA) |
| `datasets` | >= 3.0.0 | Dataset loading and processing |

---

## Quick Start

Set up the environment and start training in a few steps:

```bash
# 1. Clone and enter the repo
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

# 2. Create and activate a conda environment (or use venv — see Installation)
conda create -n finetune python=3.11 -y
conda activate finetune

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify GPU is available
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# 5. Run training with the bundled NTNU dataset (~1.2K ChatML examples)
bash scripts/train_preset.sh gpu-12gb-4b
# Or manually:
# python src/train.py --model_id Qwen/Qwen3.5-4B --dataset_path ./data/ntnu_dataset.jsonl --output_dir ./models/ntnu/gpu-12gb-4b

# 6. Run inference with the base model (before fine-tuning) — save the output for comparison
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B --prompt "台灣師范大學的地址是什麼" --no_interactive

# 7. Run inference with the fine-tuned model — compare with step 6
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B --adapter_path ./models/ntnu/gpu-12gb-4b --prompt "台灣師范大學的地址是什麼" --no_interactive

# 8. Interactive chat with the fine-tuned model (type quit to exit)
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B --adapter_path ./models/ntnu/gpu-12gb-4b
```

See the [Usage](#usage) section for detailed training options and dataset formats.

---

## Project Structure

```
fine-tuning-playground/
├── requirements.txt          # Python package dependencies
├── src/                      # Source scripts
│   ├── train.py              # Full SFT pipeline (single file, top-to-bottom)
│   └── inference.py          # Interactive inference + base/finetuned/compare modes
├── data/                     # Training datasets and generators
│   ├── ntnu_dataset.jsonl    # Bundled NTNU ChatML dataset (~1.2K examples)
│   ├── generate_ntnu_dataset.py
│   ├── ntnu_extended_records.py
│   └── ntnu_massive_records{1-4}.py
├── models/                   # Fine-tuned LoRA adapters (gitignored)
├── scripts/                  # install_deps.* / train_preset.*
├── README.md                 # This file
└── README.zh-Hant.md         # Traditional Chinese documentation
```

### File Descriptions

**`src/train.py`** — The complete supervised fine-tuning pipeline in one file. Runs sequentially: SSL fix → parse CLI args → detect device → load tokenizer → load base model (optional 4-bit) → apply LoRA → load & format dataset → configure `SFTTrainer` → train → save adapter. All hyperparameters are CLI flags; no separate config module.

**`src/inference.py`** — Standalone inference script for base model, fine-tuned model, or side-by-side comparison. By default, enters an interactive prompt loop after loading (`You:` prompt; type `quit` / `exit` / `q` to stop). CLI flags: `--mode` (`base` / `finetuned` / `compare`), `--model_id`, `--adapter_path`, `--prompt`, `--no_interactive`, `--load_in_4bit`, `--use_thinking`, `--max_new_tokens`, `--temperature`, `--top_p`, `--device`, `--torch_dtype`.

**`data/`** — Contains the bundled `ntnu_dataset.jsonl` (~1,252 ChatML records about National Taiwan Normal University) and scripts to regenerate it (`generate_ntnu_dataset.py`, `ntnu_extended_records.py`, `ntnu_massive_records{1-4}.py`).

**`scripts/train_preset.sh` / `scripts/train_preset.ps1`** — Named training presets for common GPU tiers and model sizes. Override the dataset with `DATASET_PATH=./data/my.jsonl`.

---

## Configuration

### Training (`src/train.py`)

| Category | Parameter | Default | Description |
|---|---|---|---|
| **Model** | `model_id` | `Qwen/Qwen3.5-0.8B` | Hugging Face model identifier (use `Qwen/Qwen3.5-4B` or `Qwen/Qwen3-4B` on GPU) |
| | `load_in_4bit` | `True` | Enable 4-bit NF4 quantization (auto-disabled on CPU) |
| | `torch_dtype` | `bfloat16` | Computation dtype (falls back to `float32` on CPU) |
| **LoRA** | `lora_r` | `16` | LoRA rank |
| | `lora_alpha` | `32` | LoRA scaling factor (commonly 2 × r) |
| | `lora_dropout` | `0.05` | Dropout for LoRA layers |
| | `target_modules` | All 7 linear layers | Comma-separated module names (e.g. `q_proj,k_proj,v_proj,o_proj`) |
| **Data** | `dataset_path` | _(required)_ | Local file or HF dataset name (local paths must start with `./`, `.`, or `/`) |
| | `dataset_format` | `chat` | Input format: `chat`, `alpaca`, or `text` |
| | `test_split` | `0.05` | Fraction for evaluation (0 = no eval) |
| | `max_seq_length` | `4096` | Max token length for each example |
| | `use_thinking` | `False` | Enable thinking mode in chat template |
| **Training** | `learning_rate` (`--lr`) | `2e-4` | Peak learning rate |
| | `num_epochs` (`--epochs`) | `3` | Training epochs |
| | `per_device_batch_size` (`--batch_size`) | `2` | Batch size per device |
| | `gradient_accum_steps` | `8` | Gradient accumulation steps |
| | `warmup_ratio` | `0.03` | Linear warmup fraction |
| | `logging_steps` | `10` | Log metrics every N steps |
| | `save_steps` | `200` | Save checkpoint every N steps |
| | `output_dir` | `./models/qwen3-4b-finetuned` | Output directory |
| **Hardware** | `device` | `None` (auto) | Compute device: `gpu`, `cpu`, or auto-detect |
| **Hub** | `push_to_hub` | `False` | Push adapter to HF Hub |
| | `hub_model_id` | `""` | Target repository on Hub |

### Inference (`src/inference.py`)

| Category | Parameter | Default | Description |
|---|---|---|---|
| **Model** | `model_id` | `Qwen/Qwen3.5-0.8B` | Base model (must match the model you fine-tuned) |
| | `mode` | `base` | `base`, `finetuned`, or `compare` |
| | `adapter_path` | — | LoRA adapter directory (required for `finetuned` / `compare`) |
| | `load_in_4bit` | `True` | 4-bit quantization for inference |
| | `torch_dtype` | `bfloat16` | Computation dtype |
| | `device` | `None` (auto) | `gpu`, `cpu`, or auto-detect |
| **Generation** | `prompt` | — | Single test prompt (omit to skip batch test) |
| | `max_new_tokens` | `2048` | Max generation tokens |
| | `temperature` | `0.7` | Sampling temperature |
| | `top_p` | `0.9` | Nucleus sampling threshold |
| | `use_thinking` | `False` | Enable thinking mode in chat template |
| | `no_interactive` | `False` | Exit after batch inference instead of entering the prompt loop |

> **Note:** Both `src/train.py` and `src/inference.py` default `--model_id` to `Qwen/Qwen3.5-0.8B` (CPU smoke tests). For GPU training, pass `--model_id Qwen/Qwen3.5-4B` or `Qwen/Qwen3-4B`, and use the **same** `--model_id` at inference. In `src/inference.py`, omitting `--prompt` skips batch tests and goes straight to the interactive loop; with `--no_interactive` and no `--prompt`, two built-in NTNU test prompts are used.

---

## Recommended Training Presets

These presets assume **QLoRA (4-bit)**, **`bfloat16`**, **ChatML format** (`--dataset_format chat`), and an **effective batch size of 16** (`per_device_batch_size × gradient_accum_steps`). All presets enable a 5% eval split and save the best checkpoint by eval loss.

The bundled NTNU dataset (`./data/ntnu_dataset.jsonl`) contains **~1,252** examples — presets default to **3 epochs** and **lr = 2e-4** for this size. Use the [dataset-size adjustments](#dataset-size-guidelines) below when training on larger custom data.

### Platform & OS

| Platform | Install | List presets | Run training | Notes |
|---|---|---|---|---|
| **Linux** | `bash scripts/install_deps.sh` | `bash scripts/train_preset.sh --list` | `bash scripts/train_preset.sh gpu-12gb-4b` | Native CUDA recommended |
| **macOS** | `bash scripts/install_deps.sh` | same as Linux | `bash scripts/train_preset.sh cpu-smoke` | No NVIDIA CUDA; use CPU smoke or external GPU |
| **Windows (PowerShell)** | `powershell -File scripts/install_deps.ps1` | `powershell -File scripts/train_preset.ps1 -List` | `powershell -File scripts/train_preset.ps1 -Preset gpu-12gb-4b` | Set `$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` if OOM |
| **Windows (Git Bash)** | `bash scripts/install_deps.sh` | same as Linux | same as Linux | Same commands as Linux |

Override the dataset, output directory, or hyperparameters without editing scripts:

```bash
# Custom dataset (Linux / Git Bash)
DATASET_PATH=./data/my_dataset.jsonl bash scripts/train_preset.sh gpu-12gb-4b

# Larger dataset — fewer epochs (30K+ examples)
NUM_EPOCHS=2 LEARNING_RATE=1.5e-4 DATASET_PATH=./data/large.jsonl bash scripts/train_preset.sh gpu-24gb-9b

# Custom output directory
OUTPUT_DIR=./models/my-run bash scripts/train_preset.sh gpu-12gb-4b
```

```powershell
# Custom dataset (Windows PowerShell)
$env:DATASET_PATH = ".\data\my_dataset.jsonl"
powershell -File scripts/train_preset.ps1 -Preset gpu-12gb-4b

# Larger dataset — fewer epochs
$env:NUM_EPOCHS = "2"
$env:LEARNING_RATE = "1.5e-4"
$env:DATASET_PATH = ".\data\large.jsonl"
powershell -File scripts/train_preset.ps1 -Preset gpu-24gb-9b
```

### VRAM and Model Size (QLoRA)

Starting points before applying [dataset-size](#dataset-size-guidelines) tuning. Reduce `--max_seq_length` first if you hit OOM; then try `--target_modules q_proj,k_proj,v_proj,o_proj`.

| VRAM | Model | Batch | Accum | Eff. Batch | Max Seq | LoRA r | Epochs | LR | Typical GPU |
|---|---|---|---|---|---|---|---|---|---|
| 6–8 GB | Qwen3.5-4B | 1 | 16 | 16 | 2048 | 16 | 3 | 2e-4 | RTX 3060 / 4060 / 4070 |
| 12 GB | Qwen3.5-4B | 2 | 8 | 16 | 2048 | 16 | 3 | 2e-4 | RTX 3060 12 GB / 4070 |
| 12 GB | Qwen3.5-9B | 1 | 16 | 16 | 1536 | 16 | 3 | 2e-4 | RTX 5070 Ti laptop |
| 12 GB | Qwen3.5-9B | 1 | 16 | 16 | 2048 | 16 | 3 | 2e-4 | Attention-only LoRA if OOM |
| 16 GB | Qwen3.5-9B | 1 | 16 | 16 | 2048 | 32 | 3 | 2e-4 | RTX 4080 16 GB / 4070 Ti Super |
| 24 GB | Qwen3.5-4B | 2 | 8 | 16 | 4096 | 32 | 3 | 2e-4 | RTX 3090 / 4090 |
| 24 GB | Qwen3.5-9B | 2 | 8 | 16 | 2048 | 32 | 3 | 2e-4 | RTX 4090 / A5000 |
| CPU | Qwen3.5-0.8B | 1 | 4 | 4 | 512 | 8 | 1 | 2e-4 | Smoke test only; `--device cpu` |

### Dataset Size Guidelines

Apply these on top of a [hardware preset](#vram-and-model-size-qlora). Effective batch size should stay in the **16–32** range.

| Examples | Epochs | Learning Rate | LoRA r | test_split | Preset override | Notes |
|---|---|---|---|---|---|---|
| < 500 | 3–5 | 1e-4 | 16 | 0.10 | `LEARNING_RATE=1e-4 NUM_EPOCHS=5` | High overfit risk; monitor eval loss |
| 500 – 5,000 | 3 | 2e-4 | 16 | 0.05 | _(preset defaults)_ | **Bundled NTNU (~1.2K)**; good default for small custom sets |
| 5,000 – 30,000 | 2–3 | 1.5e-4 – 2e-4 | 32 | 0.05 | `NUM_EPOCHS=2 LEARNING_RATE=1.5e-4` | Higher rank helps domain adaptation |
| 30,000+ | 1–2 | 1.5e-4 | 16–32 | 0.05 | `NUM_EPOCHS=2 LEARNING_RATE=1.5e-4` | Avoid 3+ epochs; watch for overfitting |

**Combined example — 12 GB laptop + bundled NTNU (~1.2K):**

```bash
bash scripts/train_preset.sh gpu-12gb-4b
# Equivalent manual command — see [Manual Commands](#manual-commands-by-preset)
```

**Combined example — 24 GB desktop + 50K custom examples:**

```bash
NUM_EPOCHS=2 LEARNING_RATE=1.5e-4 DATASET_PATH=./data/large.jsonl \
  bash scripts/train_preset.sh gpu-24gb-9b
```

### Named Presets

| Preset | Target Hardware | Model | Purpose |
|---|---|---|---|
| `smoke-4b` | Any GPU | Qwen3.5-4B | 1-epoch pipeline check |
| `smoke-9b` | 12 GB+ GPU | Qwen3.5-9B | 1-epoch pipeline check (12 GB friendly) |
| `cpu-smoke` | CPU only | Qwen3.5-0.8B | Verify install without GPU |
| `gpu-8gb-4b` | 6–8 GB | Qwen3.5-4B | Entry-level consumer GPU |
| `gpu-12gb-4b` | 12 GB | Qwen3.5-4B | Balanced 4B on 12 GB |
| `gpu-12gb-9b` | 12 GB mobile/desktop | Qwen3.5-9B | **Recommended for RTX 5070 Ti 12 GB laptop** |
| `gpu-12gb-9b-long` | 12 GB | Qwen3.5-9B | Longer answers (2048 seq, attention-only LoRA) |
| `gpu-16gb-9b` | 16 GB | Qwen3.5-9B | Full 7-layer LoRA, r=32 |
| `gpu-24gb-4b` | 24 GB | Qwen3.5-4B | High-quality 4B, long context |
| `gpu-24gb-9b` | 24 GB | Qwen3.5-9B | **Recommended desktop 9B training** |

### Preset Scripts

Optional: reduce CUDA fragmentation on Windows / Git Bash before training:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Linux / macOS / Git Bash:**

```bash
# List all presets
bash scripts/train_preset.sh --list

# Smoke test (recommended before a full run)
bash scripts/train_preset.sh smoke-9b

# Full training — 12 GB laptop with Qwen3.5-9B (bundled NTNU dataset)
bash scripts/train_preset.sh gpu-12gb-9b

# Custom dataset
DATASET_PATH=./data/my_dataset.jsonl bash scripts/train_preset.sh gpu-24gb-9b

# Custom output directory
OUTPUT_DIR=./models/my-run DATASET_PATH=./data/my.jsonl bash scripts/train_preset.sh gpu-12gb-4b
```

**Windows PowerShell:**

```powershell
# List all presets
powershell -File scripts/train_preset.ps1 -List

# Smoke test
powershell -File scripts/train_preset.ps1 -Preset smoke-9b

# Full training — 12 GB laptop
powershell -File scripts/train_preset.ps1 -Preset gpu-12gb-9b

# Custom dataset
$env:DATASET_PATH = ".\data\my_dataset.jsonl"
powershell -File scripts/train_preset.ps1 -Preset gpu-24gb-9b
```

### Manual Commands by Preset

Use these when you prefer a single copy-paste command without the helper script.

**Smoke test — Qwen3.5-4B (any GPU):**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-4B \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 1 \
  --per_device_batch_size 1 \
  --gradient_accum_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 \
  --test_split 0.05 \
  --save_steps 200 \
  --output_dir ./models/ntnu/smoke-4b
```

**12 GB — Qwen3.5-4B + NTNU (~1.2K):**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-4B \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 3 \
  --per_device_batch_size 2 \
  --gradient_accum_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 \
  --lora_dropout 0.05 \
  --warmup_ratio 0.03 \
  --test_split 0.05 \
  --logging_steps 10 \
  --save_steps 200 \
  --output_dir ./models/ntnu/gpu-12gb-4b
```

**12 GB laptop — Qwen3.5-9B + NTNU (~1.2K):**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-9B \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 1536 \
  --num_epochs 3 \
  --per_device_batch_size 1 \
  --gradient_accum_steps 16 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 \
  --lora_dropout 0.05 \
  --warmup_ratio 0.03 \
  --test_split 0.05 \
  --logging_steps 10 \
  --save_steps 250 \
  --output_dir ./models/ntnu/gpu-12gb-9b
```

**24 GB desktop — Qwen3.5-9B + NTNU (~1.2K):**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-9B \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 3 \
  --per_device_batch_size 2 \
  --gradient_accum_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 32 --lora_alpha 64 \
  --lora_dropout 0.05 \
  --test_split 0.05 \
  --save_steps 300 \
  --output_dir ./models/ntnu/gpu-24gb-9b
```

**24 GB desktop — Qwen3.5-9B + large custom dataset (30K+ examples):**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-9B \
  --dataset_path ./data/large_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 2 \
  --per_device_batch_size 2 \
  --gradient_accum_steps 8 \
  --learning_rate 1.5e-4 \
  --lora_r 32 --lora_alpha 64 \
  --lora_dropout 0.05 \
  --test_split 0.05 \
  --save_steps 300 \
  --output_dir ./models/custom/gpu-24gb-9b
```

**Post-training comparison:**

```bash
python src/inference.py \
  --mode compare \
  --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b \
  --load_in_4bit True \
  --prompt "請簡單介紹國立臺灣師範大學。" \
  --no_interactive
```

---

## Data Formats

The pipeline supports three input formats. All are normalized to the ChatML message format internally.

### ChatML Format

This is the native format for Qwen3. Each JSON object (one per line in JSONL, or as list items in JSON) contains a `"messages"` key:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

The system message is optional. This format requires `--dataset_format chat` (the default).

### Alpaca Format

The Stanford Alpaca format uses three fields: `instruction`, `input`, and `output`. The `input` field may be empty.

```json
{
  "instruction": "Explain the concept of recursion.",
  "input": "",
  "output": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem..."
}
```

```json
{
  "instruction": "Translate the following to French.",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous ?"
}
```

This format requires `--dataset_format alpaca`.

### Text Format

A flat text format where each example is a single `"text"` field containing the full conversation. The pipeline splits on `<|im_end|>\n<|im_start|>` to recover alternating user/assistant turns:

```json
{
  "text": "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\nMachine learning is a subset of AI...<|im_end|>"
}
```

This format requires `--dataset_format text`.

---

## Usage

### Basic Training

Train on a local JSONL file where each line is a ChatML-formatted conversation:

```bash
python src/train.py \
    --model_id Qwen/Qwen3.5-4B \
    --dataset_path ./data/my_dataset.jsonl \
    --output_dir ./my-finetuned-model
```

### Training with Alpaca Dataset

```bash
python src/train.py \
    --dataset_path ./data/alpaca_data.json \
    --dataset_format alpaca \
    --lr 2e-4 \
    --epochs 5 \
    --output_dir ./alpaca-finetuned
```

### Training from Hugging Face Hub

Load a dataset directly from the Hugging Face Hub:

```bash
python src/train.py \
    --dataset_path databricks/databricks-dolly-15k \
    --dataset_format alpaca \
    --max_seq_length 2048 \
    --output_dir ./dolly-finetuned
```

### Inference — Interactive (Default)

By default, `src/inference.py` keeps the model loaded and enters an interactive prompt loop. Type your question at the `You:` prompt; enter `quit`, `exit`, or `q` (or press Ctrl+C) to exit.

- **No `--prompt`**: skips batch tests and goes straight to the input loop.
- **With `--prompt`**: runs that prompt once, then continues interactively.

```bash
# Fine-tuned model — interactive chat
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./models/ntnu/gpu-12gb-4b

# Base model — run one prompt, then continue interactively
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B \
    --prompt "台灣師范大學的地址是什麼"

# Compare mode — each input shows base vs fine-tuned responses
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./models/ntnu/gpu-12gb-4b
```

### Inference — One-Shot (`--no_interactive`)

For scripts, CI, or quick baseline comparisons, pass `--no_interactive` to run batch inference once and exit. Without `--prompt`, two built-in NTNU test prompts are used.

```bash
# Base model baseline (before fine-tuning)
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B \
    --prompt "台灣師范大學的地址是什麼" --no_interactive

# Fine-tuned model (after training)
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./my-finetuned-model \
    --prompt "台灣師范大學的地址是什麼" --no_interactive

# Side-by-side comparison (batch mode, built-in test prompts)
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./models/my-finetuned-model --no_interactive
```

### Inference with Thinking Mode

To enable Qwen3's chain-of-thought reasoning, pass `--use_thinking True`. Unless `--no_interactive` is set, the script continues to the interactive loop after the initial `--prompt` run.

```bash
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
    --adapter_path ./my-finetuned-model \
    --use_thinking True \
    --prompt "台灣師范大學的地址是什麼"
```

---

## Advanced Usage

### Custom LoRA Target Modules

By default, LoRA is applied to all 7 linear projection layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. You can restrict to attention-only modules to save memory:

```bash
python src/train.py \
    --dataset_path ./data/train.jsonl \
    --model_id Qwen/Qwen3.5-4B \
    --target_modules q_proj,k_proj,v_proj,o_proj \
    --output_dir ./attention-only-lora
```

### Push to Hugging Face Hub

To upload the trained adapter directly to the Hugging Face Hub, you must be logged in:

```bash
huggingface-cli login
```

Then add the `--push_to_hub` and `--hub_model_id` flags:

```bash
python src/train.py \
    --dataset_path ./data/train.jsonl \
    --output_dir ./my-adapter \
    --push_to_hub True \
    --hub_model_id your-username/qwen3-4b-finetuned
```

### Full vs QLoRA

QLoRA (4-bit) is the default and requires the least memory. To train in FP16 LoRA without quantization:

```bash
python src/train.py \
    --dataset_path ./data/train.jsonl \
    --load_in_4bit False \
    --per_device_batch_size 1 \
    --output_dir ./fplora-finetuned
```

This requires approximately 12 GB VRAM. Reduce `per_device_batch_size` or `max_seq_length` if you encounter out-of-memory errors.

### Enable Thinking Mode During Training

If your dataset includes reasoning traces (content wrapped in `<think>...</think>`), you can enable the thinking-aware chat template:

```bash
python src/train.py \
    --dataset_path ./data/reasoning_dataset.jsonl \
    --use_thinking True \
    --output_dir ./thinking-finetuned
```

This ensures the tokenizer correctly handles the `<think>` tags during training.

---

## Output Artifacts

After training, the `output_dir` contains:

```
my-finetuned-model/
├── adapter_config.json     # LoRA configuration (rank, alpha, target modules, etc.)
├── adapter_model.safetensors  # Trained LoRA adapter weights
├── tokenizer_config.json   # Tokenizer configuration
├── tokenizer.json          # Tokenizer vocabulary
├── special_tokens_map.json # Special token definitions
└── (checkpoint-XXX/)       # Intermediate training checkpoints (if save_steps was set)
```

The adapter is only ~40–80 MB, making it easy to share and version.

---

## Tips & Best Practices

1. **Start small**: Run `bash scripts/train_preset.sh smoke-4b` (or `smoke-9b` for 9B) before a full training run. Alternatively use `--max_seq_length 2048` and `--num_epochs 1` manually.

2. **Effective batch size**: The effective batch size is `per_device_batch_size * gradient_accum_steps * num_gpus`. Aim for an effective batch size of 16–32. With 1 GPU, `--per_device_batch_size 2 --gradient_accum_steps 8` gives an effective batch size of 16.

3. **Learning rate**: For LoRA/QLoRA, `2e-4` is a robust default. If your dataset is very small (< 500 examples), consider lowering it to `1e-4` to avoid overfitting.

4. **Monitor eval loss**: The trainer automatically tracks eval loss and saves the best checkpoint. Watch for divergence (eval loss going up) as a sign of overfitting or a learning rate that is too high.

5. **Sequence length**: Examples are truncated/padded to `max_seq_length` via `SFTConfig.max_length`. Sequence packing is not enabled in the current trainer configuration.

6. **Mixed precision**: `bfloat16` is preferred over `float16` for training stability on GPU. If your GPU does not support bf16, use `--torch_dtype float16`. On CPU, dtype automatically falls back to `float32`.

7. **Data quality matters**: A clean dataset of 500–1000 high-quality examples often outperforms a noisy dataset of 10,000 examples for instruction tuning.

8. **Match `model_id` at inference**: The base model used in `src/inference.py` (`--model_id`) must match the model you fine-tuned.

9. **Interactive vs one-shot inference**: Use the default interactive loop for manual exploration. Add `--no_interactive` when you need a single batch run (e.g., Quick Start steps 6–7 or shell scripts).

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| Training uses CPU despite having a GPU | CPU-only PyTorch installed (`torch x.x.x+cpu`) | Use Python 3.11–3.12, then `pip install -r requirements.txt` |
| `RuntimeError: GPU was requested via --device gpu` | CUDA not available to PyTorch | Install a CUDA-enabled PyTorch wheel or omit `--device gpu` |
| `OutOfMemoryError` | Batch size too large for your GPU | Reduce `--per_device_batch_size` or `--max_seq_length` |
| `KeyError: 'qwen3'` | Transformers version too old | Upgrade to `transformers>=4.47.0` |
| `bitsandbytes` import error | Missing CUDA or incompatible version | Ensure bitsandbytes matches your CUDA version: `pip install bitsandbytes --force-reinstall` |
| Hub download / SSL errors | Broken `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` on conda/Windows | `src/train.py` and `src/inference.py` auto-fix at startup; or fix the env var manually |
| Training loss is NaN | Learning rate too high or dtype issue | Lower `--learning_rate` (or `--lr`) or switch to `--torch_dtype bfloat16` |
| Model repeats the same phrase | Overfitting or temperature too low | Increase `--temperature` or reduce `--num_epochs` |
| `apply_chat_template` error | Data not in expected message format | Check that your dataset uses the correct format (see [Data Formats](#data-formats)) |
| Local file loaded from Hub by mistake | `dataset_path` missing `./` prefix | Use `./data/train.jsonl` instead of `data/train.jsonl` for local files |
| Eval loss increases during training | Overfitting | Reduce `--num_epochs`, increase `--test_split`, or add more training data |
| `LoRA adapter not found` | Wrong `--adapter_path` | Point to the output dir or a `checkpoint-*` subfolder; inference auto-picks the latest checkpoint |

---

## License

This project is licensed under the [MIT License](LICENSE). Qwen3 / Qwen3.5 models are released under their respective licenses on Hugging Face (typically Apache 2.0).

Traditional Chinese documentation: [README.zh-Hant.md](README.zh-Hant.md)
