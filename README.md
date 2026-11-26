# Fine-Tuning Playground

## Project Introduction

This repository is a teaching-oriented project that demonstrates **supervised fine-tuning (SFT)** end to end. It takes **Qwen3.5-4B** as the example base model and shows how to prepare data, train a LoRA / QLoRA adapter, inspect the model and tokenizer, compare responses before and after training, and evaluate the result.

### What is SFT?

A pretrained language model learns to predict the next token from a very large, general corpus. It has broad knowledge, but it has not been trained to reliably follow a user's instruction, answer in a chosen style, or stay within a domain. **Supervised fine-tuning (SFT)** is the step that bridges that gap: you continue training the model on a curated collection of prompt–answer pairs, so it learns to produce the kind of response you want.

In this project, a training example is usually a ChatML conversation with `user` and `assistant` messages. The chat template turns those messages into the exact text the model sees, and `trl.SFTTrainer` computes the loss on that text. With `--assistant_only_loss`, the loss is restricted to the assistant's answer (and its end-of-turn token); the option `--use_thinking` enables Qwen's `<think>` reasoning tags in the template. After training, you can run `src/inference.py --mode compare` to see the same prompt answered by the untouched base model and by the fine-tuned model side by side.

Instead of updating every weight in the model, the demo uses **LoRA**: small trainable matrices are added to the model's projection layers while the original weights stay frozen. **QLoRA** adds 4-bit NF4 quantization of the base weights, which is why a 4B model can be adapted on a consumer GPU.

### Base model: Qwen3.5-4B

**Qwen3.5-4B** ([model card](https://huggingface.co/Qwen/Qwen3.5-4B)) is a dense, decoder-only language model from Alibaba's Qwen team. The 4B size is a practical middle ground for a fine-tuning demo: it is large enough to give meaningful answers, yet small enough to run locally with 4-bit quantization. Like other Qwen3 / Qwen3.5 models, it supports long context, grouped-query attention, and an optional chain-of-thought **thinking mode** through `<think>` tags in the chat template.

The commands in this README use `Qwen/Qwen3.5-4B` as the example `--model_id`. For fast CPU smoke tests the scripts fall back to `Qwen/Qwen3.5-0.8B` (`DEFAULT_MODEL_ID` in `src/model_utils.py`); when you train a 4B adapter, pass `--model_id Qwen/Qwen3.5-4B` and use the **same** value for inference and evaluation. Always pair a trained adapter with the base model it was trained from.

---

## Installation

Use **Python 3.11 or 3.12** with a compatible PyTorch wheel. Use the Python version specified by AMD for your ROCm release. Create a fresh environment for each PyTorch backend.

### Option 1 — Miniconda (Recommended)

```bash
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

conda create -n finetune python=3.11 -y
conda activate finetune

# Install the platform default, or choose a backend below
python scripts/install_deps.py
```

### Option 2 — venv

```bash
git clone https://github.com/cw1997/fine-tuning-playground.git
cd fine-tuning-playground

python -m venv venv
source venv/bin/activate  # Linux / macOS
# venv\Scripts\activate   # Windows

# Install the platform default, or choose a backend below
python scripts/install_deps.py
```

### Install and verify a backend

The installer runs a small tensor computation and prints the active backend and PyTorch version. Its default is MPS on Apple Silicon macOS, CPU on Intel macOS, and CUDA on Windows and Linux. The Bash wrapper accepts the Python installer's flags. The Windows PowerShell wrapper uses `-Backend` and `-Rocm4Bit`.

| Backend | Install in the active environment | Verify model tooling |
|---|---|---|
| NVIDIA CUDA (Linux / Windows) | `python scripts/install_deps.py --backend cuda` | `python src/inspect_model.py --device cuda --no_load` |
| AMD ROCm (Linux / WSL2 / Windows) | Install the AMD PyTorch wheel, then run `python scripts/install_deps.py --backend rocm` | `python src/inspect_model.py --device rocm --no_load` |
| Apple M series (macOS) | `bash scripts/install_deps.sh --backend mps` using native arm64 Python | `python src/inspect_model.py --device mps --no_load` |
| Intel XPU (Linux / Windows) | `python scripts/install_deps.py --backend xpu` after installing the GPU driver | `python src/inspect_model.py --device xpu --no_load` |
| CPU | `python scripts/install_deps.py --backend cpu` | `python src/inspect_model.py --device cpu --no_load` |

For ROCm, match the driver, GPU, and PyTorch wheel before running the installer. The installer keeps the selected AMD wheel in the environment. For Intel, check the GPU and driver requirements before installing. [AMD Linux installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html) · [AMD WSL2 compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html) · [AMD Windows installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html) · [Intel XPU setup](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)

For AMD NF4, run `python scripts/install_deps.py --backend rocm --rocm-4bit` or `powershell -File scripts/install_deps.ps1 -Backend rocm -Rocm4Bit`. Match the bitsandbytes build to your GPU and ROCm release. [bitsandbytes installation guide](https://huggingface.co/docs/bitsandbytes/installation#amd-rocm)

`python -m pip install -r requirements.txt` remains available for direct NVIDIA CUDA installation. For other backends, the installer uses `requirements-common.txt` after selecting or verifying PyTorch.

Core packages:

| Package | Minimum Version | Purpose |
|---|---|---|
| `torch` | >= 2.4.0, < 2.12 | Deep learning framework (CUDA cu128 via `requirements.txt`) |
| `transformers` | >= 5.12.1 | Qwen model loading, tokenization and training |
| `peft` | >= 0.13.0 | LoRA / QLoRA adapter configuration |
| `trl` | >= 1.6.0 | `SFTTrainer` with assistant labels and chunked loss |
| `accelerate` | >= 1.13.0 | Trainer placement and mixed precision |
| `bitsandbytes` | CUDA >= 0.46.1, ROCm >= 0.50.0 | Optional 4-bit quantization (QLoRA) |
| `datasets` | >= 3.0.0 | Dataset loading and processing |
| `packaging` | >= 21.0 | PyTorch version checks |
| `certifi` | >= 2023.7.22 | CA bundle repair for Hub downloads |

---

## Quick Start

### Cross-platform smoke test

After installing your backend, run one training step on Qwen3.5-0.8B (the scripts' default model) and reload the adapter. Replace `gpu` with `mps`, `rocm`, `xpu`, `cuda`, or `cpu` to set the target explicitly.

```bash
python -c "from pathlib import Path; p=Path('models/backend-smoke'); p.mkdir(parents=True, exist_ok=True); (p/'train.jsonl').write_bytes(Path('data/ntnu_dataset.jsonl').read_bytes().splitlines(keepends=True)[0])"
python src/train.py --device gpu --dataset_path ./models/backend-smoke/train.jsonl --num_epochs 1 --test_split 0 --max_seq_length 64 --per_device_batch_size 1 --gradient_accum_steps 1 --output_dir ./models/backend-smoke/adapter
python src/inference.py --device gpu --mode finetuned --adapter_path ./models/backend-smoke/adapter --prompt "What does a tokenizer do?" --max_new_tokens 32 --no_interactive
```

On a compatible AMD GPU, exercise NF4 by adding `--load_in_4bit True` to both commands.

### Qwen3.5-4B on NVIDIA CUDA

Train on the bundled NTNU dataset (1,252 ChatML examples), then compare the base model with the adapter:

```bash
# Train
python src/train.py \
  --model_id Qwen/Qwen3.5-4B \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --device cuda \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 3 \
  --per_device_batch_size 2 \
  --gradient_accum_steps 8 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 \
  --test_split 0.05 \
  --output_dir ./models/ntnu/gpu-12gb-4b

# Before fine-tuning
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B \
  --prompt "台灣師範大學的地址是什麼" --no_interactive

# After fine-tuning
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b \
  --prompt "台灣師範大學的地址是什麼" --no_interactive

# Optional: interactive chat, or a side-by-side comparison
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b \
  --prompt "請簡單介紹國立臺灣師範大學。" --no_interactive
```

See [Usage](#usage) for more dataset formats, training options, and inference modes.

---

## Project Structure

```
fine-tuning-playground/
├── requirements.txt          # NVIDIA cu128 installation entry point
├── requirements-common.txt   # Shared Python dependencies
├── LICENSE                   # MIT license
├── src/                      # Source scripts
│   ├── train.py              # Full SFT pipeline (single file, top-to-bottom)
│   ├── inference.py          # Interactive inference + base/finetuned/compare modes
│   ├── evaluate_ntnu.py      # Greedy generation and blinded NTNU quality scoring
│   ├── model_utils.py        # Shared tokenizer, model loading and SSL helpers
│   ├── device_utils.py       # Backend selection, precision, and ROCm NF4 validation
│   ├── inspect_model.py      # Model internals inspection (params, vocab, dims, config)
│   └── inspect_tokenizer.py  # Tokenizer inspection (token pieces, ids, embedding vectors)
├── data/                     # Bundled training datasets
│   ├── ntnu_dataset.jsonl    # 1,252 NTNU ChatML examples
│   ├── full_dataset.jsonl    # 1,746 aggregated NTNU + Taiwan records
│   ├── ntnu/                 # Categorized NTNU JSONL files
│   └── taiwan/               # Categorized Taiwan JSONL files
├── evaluation/ntnu_qa.jsonl  # 60 held-out prompts with criteria (fact: + sources)
├── models/                   # LoRA adapters and evaluation output (gitignored)
├── scripts/                  # Dependency installers
│   ├── install_deps.py       # Backend-aware installer and device verification
│   ├── install_deps.sh       # Bash wrapper
│   └── install_deps.ps1      # PowerShell wrapper
├── tests/
│   └── test_device_utils.py  # Backend selection and quantization tests
├── README.md                 # English documentation
└── README.zh-Hant.md         # Traditional Chinese documentation
```

### File Descriptions

**`src/train.py`** — The complete supervised fine-tuning pipeline in one file. Runs sequentially: SSL fix → parse CLI args → resolve device → load tokenizer → load base model (optional 4-bit) → apply LoRA → load and format dataset → configure `SFTTrainer` → train → save adapter. All hyperparameters are CLI flags; there is no separate config module.

**`src/inference.py`** — Standalone inference script for the base model, a fine-tuned model, or a side-by-side comparison. By default it enters an interactive loop after loading (`You:` prompt; type `quit`, `exit`, or `q` to stop). CLI flags: `--mode` (`base` / `finetuned` / `compare`), `--model_id`, `--adapter_path`, `--prompt`, `--load_in_4bit`, `--torch_dtype`, `--device`, `--max_new_tokens`, `--temperature`, `--top_p`, `--use_thinking`, `--no_interactive`.

**`src/evaluate_ntnu.py`** — Generates deterministic answers for the fixed question set with `generate`, shuffles several runs for blinded review with `blind`, and applies the quality gates with `score`. **`evaluation/ntnu_qa.jsonl`** holds 60 held-out cases: 40 fact questions (each with an official source), 10 uncertainty questions, and 10 general questions. Keep these prompts out of training.

**`src/model_utils.py`** — Shared helpers for SSL certificate repair, tokenizer loading, NF4 configuration, and base-model loading. `DEFAULT_MODEL_ID` is `Qwen/Qwen3.5-0.8B` for smoke tests.

**`src/device_utils.py`** — Resolves the backend, effective dtype, and NF4 setting for all model commands, including a preflight check for ROCm 4-bit kernels.

**`src/inspect_model.py`** — Prints a standardized report of model internals: total / trainable / frozen parameter counts, token embedding shape and memory estimate, vocabulary size, layer layout, attention and FFN configuration, RoPE settings, and tokenizer metadata. CLI flags: `--model_id`, `--load_in_4bit`, `--torch_dtype`, `--device`, `--no_load` (configuration and tokenizer only).

**`src/inspect_tokenizer.py`** — Prints how a sentence is split into tokens, each token's vocabulary id, and its embedding vector from the model's input embedding layer. Also supports a single token string (`--token`) or id (`--token_id`). Byte-level tokens are displayed in readable form. CLI flags: `--model_id`, `--sentence`, `--token`, `--token_id`, `--vector_limit` (`0` = all elements), `--load_in_4bit`, `--torch_dtype`, `--device`, `--no_model` (tokenization and ids only).

**`data/`** — Contains `ntnu_dataset.jsonl` (1,252 ChatML records about National Taiwan Normal University), `full_dataset.jsonl` (1,746 records covering NTNU plus Taiwan universities, districts, history, and culture), and categorized records under `ntnu/` and `taiwan/`.

**`scripts/install_deps.py`** — Installs and verifies the chosen PyTorch backend and the shared dependencies; `install_deps.sh` and `install_deps.ps1` forward arguments to it. **`tests/test_device_utils.py`** simulates hardware to test backend and precision selection; run it with `python -m unittest discover -s tests -v`.

---

## Configuration

### Training (`src/train.py`)

| Category | Parameter | Default | Description |
|---|---|---|---|
| **Model** | `model_id` | `Qwen/Qwen3.5-0.8B` | Hugging Face model identifier (use `Qwen/Qwen3.5-4B` for the demo) |
| | `load_in_4bit` | On for CUDA | NF4 QLoRA. Set `True` on compatible ROCm; MPS / XPU reject it; CPU uses unquantized weights |
| | `torch_dtype` | `bfloat16` | Computation dtype (always `float32` on CPU; FP16 when BF16 is unavailable) |
| **LoRA** | `lora_r` | `16` | LoRA rank |
| | `lora_alpha` | `32` | LoRA scaling factor (commonly 2 × r) |
| | `lora_dropout` | `0.05` | Dropout for LoRA layers |
| | `target_modules` | All 7 linear layers | Comma-separated module names (default `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`) |
| **Data** | `dataset_path` | _(required)_ | Local JSON/JSONL file, directory of JSONL files (recursive), or HF dataset name. Existing local paths are detected automatically |
| | `dataset_format` | `chat` | Input format: `chat`, `alpaca`, or `text` |
| | `test_split` | `0.05` | Fraction for evaluation (0 = no eval) |
| | `seed` | `42` | Seed for dataset splitting, adapter initialization and sampling |
| | `max_seq_length` | `4096` | Max token length for each example |
| | `use_thinking` | `False` | Enable thinking mode in the chat template |
| **Training** | `learning_rate` (`--lr`) | `2e-4` | Peak learning rate |
| | `num_epochs` (`--epochs`) | `3` | Training epochs |
| | `per_device_batch_size` (`--batch_size`) | `2` | Batch size per device |
| | `gradient_accum_steps` | `8` | Gradient accumulation steps |
| | `gradient_checkpointing` | `True` | Recompute activations to reduce peak GPU memory |
| | `group_by_length` | off | Batch similarly sized training examples to reduce padding |
| | `assistant_only_loss` | off | For chat data, supervise assistant responses and their end-of-turn token; rejects truncated targets |
| | `loss_type` | TRL default | Choose `nll` or `chunked_nll`. Chunking can reduce vocabulary projection memory |
| | `eval_each_epoch` | off | Evaluate and save each epoch, then load the lowest-loss checkpoint |
| | `eval_batch_size` | `8` | Evaluation batch size per device |
| | `warmup_ratio` | `0.03` | Linear warmup fraction |
| | `logging_steps` | `10` | Log metrics every N steps |
| | `save_steps` | `200` | Save checkpoint every N steps |
| | `output_dir` | `./models/qwen3-4b-finetuned` | Output directory |
| **Benchmark** | `benchmark_steps` | `0` | Timed optimizer steps. A positive value skips evaluation and adapter export |
| | `benchmark_warmup_steps` | `3` | Exclude this many initial steps from timing |
| | `run_metrics` | — | Optional JSON with timings, effective backend, dtype, NF4 state, token lengths, model revision and software versions |
| **Hardware** | `device` | `auto` | `auto`, `gpu`, `cuda`, `rocm`, `mps`, `xpu`, `cpu` |
| **Hub** | `push_to_hub` | `False` | Push adapter to HF Hub |
| | `hub_model_id` | `""` | Target repository on Hub |

### Inference (`src/inference.py`)

| Category | Parameter | Default | Description |
|---|---|---|---|
| **Model** | `model_id` | `Qwen/Qwen3.5-0.8B` | Base model (must match the model you fine-tuned) |
| | `mode` | `base` | `base`, `finetuned`, or `compare` |
| | `adapter_path` | — | LoRA adapter directory (required for `finetuned` / `compare`) |
| | `load_in_4bit` | On for CUDA | NF4 inference. Set `True` on compatible ROCm; MPS / XPU reject it; CPU uses unquantized weights |
| | `torch_dtype` | `bfloat16` | Computation dtype |
| | `device` | `auto` | `auto`, `gpu`, `cuda`, `rocm`, `mps`, `xpu`, `cpu` |
| **Generation** | `prompt` | — | Single test prompt; **required** when `--no_interactive` is set |
| | `max_new_tokens` | `2048` | Max generation tokens |
| | `temperature` | `0.7` | Sampling temperature |
| | `top_p` | `0.9` | Nucleus sampling threshold |
| | `use_thinking` | `False` | Enable thinking mode in the chat template |
| | `no_interactive` | `False` | Run the batch prompt once and exit instead of entering the interactive loop |

### Inspection (`src/inspect_model.py`)

| Parameter | Default | Description |
|---|---|---|
| `model_id` | `Qwen/Qwen3.5-0.8B` | Hugging Face model identifier to inspect |
| `load_in_4bit` | `False` | Load weights in 4-bit quantization (for quantized parameter counts) |
| `torch_dtype` | `float32` | Computation dtype when loading weights |
| `device` | `auto` | `auto`, `gpu`, `cuda`, `rocm`, `mps`, `xpu`, `cpu` |
| `no_load` | `False` | Print configuration and tokenizer report only; do not load weights |

### Tokenizer Inspection (`src/inspect_tokenizer.py`)

| Parameter | Default | Description |
|---|---|---|
| `model_id` | `Qwen/Qwen3.5-0.8B` | Hugging Face model identifier to inspect |
| `sentence` | — | Sentence to tokenize; prints each token piece, its id, and its vector |
| `token` | — | Single token string to look up (index/position and vector) |
| `token_id` | — | Single token id to look up (token string and vector) |
| `vector_limit` | `8` | Vector elements to print per token (`0` prints the entire vector) |
| `load_in_4bit` | `False` | Load the model in 4-bit quantization |
| `torch_dtype` | `float32` | Computation dtype when loading the model |
| `device` | `auto` | `auto`, `gpu`, `cuda`, `rocm`, `mps`, `xpu`, `cpu` |
| `no_model` | `False` | Show tokenization and ids only; do not load the model (no vectors) |

At least one of `--sentence`, `--token`, or `--token_id` is required.

> **Note:** Both `src/train.py` and `src/inference.py` default `--model_id` to `Qwen/Qwen3.5-0.8B` for CPU smoke tests. Pass `--model_id Qwen/Qwen3.5-4B` for the demo, and use the same id at inference and evaluation. In `src/inference.py`, omitting `--prompt` skips the batch run and goes straight to the interactive loop; `--no_interactive` requires a non-empty `--prompt`.

### NTNU evaluation (`src/evaluate_ntnu.py generate`)

| Argument | Default | Description |
|---|---|---|
| `--cases` (global, before `generate`) | `evaluation/ntnu_qa.jsonl` | Compatible JSONL question file |
| `model_id` | _(required)_ | Base model used by the adapter |
| `adapter_path` | _(required)_ | Adapter directory (or a parent containing `checkpoint-*`) |
| `device` | `gpu` | `auto`, `gpu`, `cuda`, `rocm`, `mps`, `xpu`, `cpu` |
| `load_in_4bit` | On for CUDA | Explicit `True` or `False` for NF4. On ROCm, opt in with `True` |
| `bf16_weights` | off | Use unquantized weights with BF16 when supported. Mutually exclusive with `--load_in_4bit` |
| `batch_size` | `4` | Generation batch size |
| `max_new_tokens` | `192` | Answer length cap |
| `output` | _(required)_ | JSON file for resumable answers and run metadata |

Generation uses greedy decoding and writes the effective backend, dtype, and quantization into the output for resume checks. Use a fresh output path when switching hardware backends or precision. See [NTNU evaluation workflow](#ntnu-evaluation-workflow) for the `blind` and `score` steps.

---

## Training Recommendations

This project uses `bfloat16`, ChatML data (`--dataset_format chat`, the default), and an effective batch size of `per_device_batch_size × gradient_accum_steps`. With the default `test_split 0.05`, the trainer evaluates on the held-out split and loads the lowest-eval-loss checkpoint at the end. Reduce `--max_seq_length` first if a run does not fit in memory.

### Dataset Size Guidelines

| Examples | Epochs | Learning Rate | test_split | Notes |
|---|---|---|---|---|
| < 500 | 3–5 | 1e-4 | 0.10 | High overfit risk; monitor eval loss |
| 500 – 5,000 | 3 | 2e-4 | 0.05 | Bundled NTNU (1,252); good default for small custom sets |
| 5,000 – 30,000 | 2–3 | 1.5e-4 – 2e-4 | 0.05 | Higher LoRA rank helps domain adaptation |
| 30,000+ | 1–2 | 1.5e-4 | 0.05 | Avoid 3+ epochs; watch for overfitting |

### Example Commands

These commands call `src/train.py` directly; no preset scripts are required.

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
  --output_dir ./models/ntnu/smoke-4b
```

**12 GB GPU — Qwen3.5-9B:**

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
  --test_split 0.05 \
  --output_dir ./models/ntnu/gpu-12gb-9b
```

**24 GB GPU — Qwen3.5-9B on a larger dataset:**

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-9B \
  --dataset_path ./data/full_dataset.jsonl \
  --device gpu \
  --load_in_4bit True \
  --torch_dtype bfloat16 \
  --max_seq_length 2048 \
  --num_epochs 2 \
  --per_device_batch_size 2 \
  --gradient_accum_steps 8 \
  --learning_rate 1.5e-4 \
  --lora_r 32 --lora_alpha 64 \
  --test_split 0.05 \
  --output_dir ./models/ntnu/gpu-24gb-9b
```

---

## NTNU Evaluation Workflow

The fixed question set contains 60 cases: 40 fact questions (each with an official source), 10 questions about uncertain claims, and 10 general questions. Do not include them in training data.

Generate answers for each adapter you want to compare, then prepare a blinded review. `blind` expects unique `NAME=PATH` arguments and writes a separate key file; `score` needs exactly two baseline names.

```bash
python src/evaluate_ntnu.py generate \
  --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b \
  --output ./models/ntnu/eval/candidate.json

python src/evaluate_ntnu.py blind \
  "existing=./models/ntnu/eval/existing.json" \
  "baseline=./models/ntnu/eval/baseline.json" \
  "candidate=./models/ntnu/eval/candidate.json" \
  --output ./models/ntnu/eval/review.json \
  --key ./models/ntnu/eval/key.json

python src/evaluate_ntnu.py score \
  --review ./models/ntnu/eval/review.json \
  --key ./models/ntnu/eval/key.json \
  --baselines existing baseline \
  --output ./models/ntnu/eval/quality.json
```

Review each answer against every listed criterion. Mark `major` for fabricated material facts such as invented eligibility, deadlines, amounts, addresses, or instructions. Mark `pass` only when every criterion is met and factual answers use Traditional Chinese; fill in both booleans and a note for failures. The `score` command gives a candidate `passes_quality` only when fact ≥ `max(34, both baselines)`, uncertain ≥ 8, general ≥ `best baseline − 1`, and `major == 0`. In addition, compare `elapsed_seconds` from `--run_metrics` runs and run one training or inference workload at a time for comparable timings.

---

## Data Formats

The pipeline supports three input formats. All are normalized into a chat representation before training.

### ChatML Format

This is the native format for Qwen3 / Qwen3.5. Each JSON object (one per line in JSONL, or as a list item in JSON) contains a `messages` key:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

The system message is optional. This format uses `--dataset_format chat` (the default).

### Alpaca Format

The Stanford Alpaca format uses three fields: `instruction`, `input`, and `output`. The `input` field may be empty.

```json
{
  "instruction": "Explain the concept of recursion.",
  "input": "",
  "output": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem..."
}
```

This format uses `--dataset_format alpaca`.

### Text Format

A flat text format where each example is a single `text` field containing the full conversation. The pipeline splits on `<|im_end|>\n<|im_start|>` to recover alternating user/assistant turns:

```json
{
  "text": "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\nMachine learning is a subset of AI...<|im_end|>"
}
```

This format uses `--dataset_format text`.

---

## Usage

### Basic Training

Train on a local JSONL file where each line is a ChatML-formatted conversation:

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-4B \
  --dataset_path ./data/my_dataset.jsonl \
  --output_dir ./models/my-adapter
```

`--dataset_path` also accepts a directory of JSONL files (scanned recursively), so the whole `data/` tree can be passed directly. Existing local paths are recognized automatically; if the path does not exist locally, it is treated as a Hugging Face dataset name.

```bash
python src/train.py \
  --model_id Qwen/Qwen3.5-4B \
  --dataset_path ./data \
  --output_dir ./models/ntnu
```

### Training with an Alpaca Dataset

```bash
python src/train.py \
  --dataset_path ./data/alpaca_data.json \
  --dataset_format alpaca \
  --lr 2e-4 \
  --epochs 5 \
  --output_dir ./models/alpaca-finetuned
```

### Training from Hugging Face Hub

Use a Hub dataset whose fields match the selected format. For example, `tatsu-lab/alpaca` has `instruction`, `input`, and `output` fields:

```bash
python src/train.py \
  --dataset_path tatsu-lab/alpaca \
  --dataset_format alpaca \
  --max_seq_length 2048 \
  --output_dir ./models/alpaca-hub-finetuned
```

### Inference — Interactive (Default)

By default, `src/inference.py` keeps the model loaded and enters an interactive prompt loop. Type your question at the `You:` prompt; enter `quit`, `exit`, or `q` (or press Ctrl+C) to exit.

- **No `--prompt`**: skips the batch run and goes straight to the input loop.
- **With `--prompt`**: runs that prompt once, then continues interactively.

```bash
# Fine-tuned model — interactive chat
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b

# Base model — run one prompt, then continue interactively
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B \
  --prompt "台灣師範大學的地址是什麼"

# Compare mode — each input shows base vs fine-tuned responses
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b
```

### Inference — One-Shot (`--no_interactive`)

For scripts or CI, pass `--no_interactive` to run one prompt and exit. **A non-empty `--prompt` is required**; the script errors out if `--no_interactive` is used without one.

```bash
# Base model baseline (before fine-tuning)
python src/inference.py --mode base --model_id Qwen/Qwen3.5-4B \
  --prompt "台灣師範大學的地址是什麼" --no_interactive

# Fine-tuned model (after training)
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/my-adapter \
  --prompt "台灣師範大學的地址是什麼" --no_interactive

# Side-by-side comparison
python src/inference.py --mode compare --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/ntnu/gpu-12gb-4b \
  --prompt "請簡單介紹國立臺灣師範大學。" --no_interactive
```

### Inference with Thinking Mode

To enable Qwen3 / Qwen3.5 chain-of-thought reasoning, pass `--use_thinking True`. Unless `--no_interactive` is set, the script continues to the interactive loop after the initial `--prompt` run.

```bash
python src/inference.py --mode finetuned --model_id Qwen/Qwen3.5-4B \
  --adapter_path ./models/my-adapter \
  --use_thinking True \
  --prompt "台灣師範大學的地址是什麼"
```

### Inspecting a Model

`src/inspect_model.py` prints a standardized report of a model's internals without needing a dataset.

```bash
# Configuration + tokenizer + weights (default)
python src/inspect_model.py --model_id Qwen/Qwen3.5-4B

# Configuration + tokenizer only (skips weight loading)
python src/inspect_model.py --model_id Qwen/Qwen3.5-4B --no_load

# Load with 4-bit quantization to inspect quantized parameter counts
python src/inspect_model.py --model_id Qwen/Qwen3.5-4B --load_in_4bit True
```

The report covers model configuration (vocabulary size, embedding dimension, layers, attention heads, KV heads, head dimension, FFN size, context length, activation, RoPE theta, tied embeddings), tokenizer metadata, and loaded weights (parameter counts, embedding shape and memory estimate, dtypes, device placement).

### Inspecting the Tokenizer

`src/inspect_tokenizer.py` shows how a sentence is split into tokens and what each token looks like to the model — its vocabulary id and embedding vector. Byte-level tokens are shown in readable form.

```bash
# Tokenize a sentence; print each token piece, id, and embedding vector
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Where is National Taiwan Normal University?"

# Look up a single token string or id
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --token "台"
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --token_id 120573

# Print the entire vector instead of the first 8 elements
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Hello" --vector_limit 0

# Tokenization + ids only (skips model loading, so no vectors)
python src/inspect_tokenizer.py --model_id Qwen/Qwen3.5-0.8B --sentence "Hello world" --no_model
```

---

## Advanced Usage

### Custom LoRA Target Modules

By default, LoRA is applied to all 7 linear projection layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. You can restrict it to attention modules:

```bash
python src/train.py \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --model_id Qwen/Qwen3.5-4B \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --output_dir ./models/attention-only-lora
```

### Push to Hugging Face Hub

Log in first, then pass `--push_to_hub True` and a destination repository:

```bash
huggingface-cli login

python src/train.py \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --output_dir ./models/my-adapter \
  --push_to_hub True \
  --hub_model_id your-username/qwen3-4b-finetuned
```

### Full vs QLoRA

QLoRA (4-bit) is the NVIDIA default. Use unquantized LoRA with `--load_in_4bit False`:

```bash
python src/train.py \
  --dataset_path ./data/ntnu_dataset.jsonl \
  --load_in_4bit False \
  --per_device_batch_size 1 \
  --output_dir ./models/lora-finetuned
```

Unquantized 4B training needs considerably more memory than 4-bit training. Reduce `per_device_batch_size` or `max_seq_length` if you run out of memory.

### Enable Thinking Mode During Training

If your dataset includes reasoning traces wrapped in `<think>...</think>`, enable the thinking-aware chat template:

```bash
python src/train.py \
  --dataset_path ./data/reasoning_dataset.jsonl \
  --use_thinking True \
  --output_dir ./models/thinking-finetuned
```

---

## Output Artifacts

After training, the `output_dir` contains:

```
my-adapter/
├── adapter_config.json        # LoRA configuration (rank, alpha, target modules, etc.)
├── adapter_model.safetensors  # Trained LoRA adapter weights
├── tokenizer_config.json      # Tokenizer configuration
├── tokenizer.json             # Tokenizer vocabulary
├── special_tokens_map.json    # Special token definitions
└── checkpoint-*/              # Intermediate checkpoints (when save_steps is reached)
```

Only the small adapter is saved; the base model is not copied into `output_dir`. A benchmark run (`--benchmark_steps > 0`) records metrics but does not export an adapter.

---

## Tips & Best Practices

1. **Start small**: run the one-example smoke test in [Quick Start](#quick-start), or use `--max_seq_length 2048 --num_epochs 1` before a full run.

2. **Effective batch size**: `per_device_batch_size × gradient_accum_steps × num_gpus`. Aim for 16–32; with one GPU, `--per_device_batch_size 2 --gradient_accum_steps 8` gives 16.

3. **Learning rate**: `2e-4` is a robust LoRA / QLoRA default. For fewer than 500 examples, try `1e-4`.

4. **Monitor eval loss**: with a positive `--test_split`, the trainer tracks eval loss and loads the best checkpoint at the end. Rising eval loss suggests overfitting or a learning rate that is too high.

5. **Sequence length**: examples are truncated/padded via `SFTConfig.max_length`; sequence packing is not enabled.

6. **Mixed precision**: the startup log shows the effective dtype. GPUs without BF16 use FP16, CPU uses FP32; for MPS training, FP16 needs PyTorch 2.5+ and BF16 needs 2.6+.

7. **Data quality matters**: 500–1000 clean examples often beat 10,000 noisy ones for instruction tuning.

8. **Match `model_id` at inference**: `--model_id` must match the base model used for training.

9. **Interactive vs one-shot inference**: use the default loop for manual exploration; add `--no_interactive` together with `--prompt` for scripts and CI.

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| Training uses CPU despite having a GPU | The active PyTorch wheel or driver cannot access the GPU | Follow the backend's [installation instructions](#install-and-verify-a-backend) in a fresh environment and pass an explicit `--device` |
| `--device gpu` or an explicit GPU is unavailable | PyTorch could not use the requested accelerator | Verify the wheel and driver; use `--device cpu` to run on CPU |
| `OutOfMemoryError` | Batch size or sequence length too large | Reduce `--per_device_batch_size` or `--max_seq_length` |
| `KeyError: 'qwen3'` | Transformers version too old | Upgrade to `transformers>=5.12.1` |
| NF4 fails on CUDA or ROCm | Missing bitsandbytes or incompatible GPU kernels | Match bitsandbytes with the GPU and CUDA/ROCm release; use `--load_in_4bit False` for unquantized weights |
| 4-bit loading rejected on MPS or XPU | Those backends do not support NF4 here | Re-run with `--load_in_4bit False` |
| MPS operator is unsupported | PyTorch has no MPS implementation for the operator | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before launching Python |
| Hub download / SSL errors | Broken `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, or `CURL_CA_BUNDLE` | `src/train.py`, `src/inference.py`, and the inspection tools repair these paths at startup; or fix the environment variables manually |
| Training loss is NaN | Learning rate too high or dtype issue | Lower `--learning_rate` (or `--lr`) or use `--torch_dtype bfloat16` |
| Model repeats the same phrase | Overfitting or temperature too low | Increase `--temperature` or reduce `--num_epochs` |
| `apply_chat_template` error | Data not in the expected message format | Check your dataset against [Data Formats](#data-formats) |
| Local file loaded from Hub by mistake | The path does not exist locally, so it was treated as a dataset name | Check the path; `./data/train.jsonl` makes a local file explicit |
| `--no_interactive` fails with an argument error | No `--prompt` was supplied | Provide a non-empty `--prompt` with `--no_interactive` |
| Eval loss increases during training | Overfitting | Reduce `--num_epochs`, increase `--test_split`, or add more training data |
| `LoRA adapter not found` | Wrong `--adapter_path` | Point to the output directory or a parent containing `checkpoint-*` directories; inference picks the latest checkpoint |

---

## License

This project is licensed under the [MIT License](LICENSE). Qwen3 / Qwen3.5 models are released under their respective licenses on Hugging Face (typically Apache 2.0).

Traditional Chinese documentation: [README.zh-Hant.md](README.zh-Hant.md)
