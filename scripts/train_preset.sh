#!/usr/bin/env bash
# Named QLoRA training presets for common GPU tiers and model sizes.
#
# Usage:
#   bash scripts/train_preset.sh --list
#   bash scripts/train_preset.sh gpu-12gb-4b
#   DATASET_PATH=./data/my.jsonl bash scripts/train_preset.sh gpu-24gb-9b
#   OUTPUT_DIR=./models/my-run bash scripts/train_preset.sh smoke-4b
#
# Optional overrides (applied on top of the preset):
#   DATASET_PATH   — default: ./data/ntnu_dataset.jsonl
#   OUTPUT_DIR     — default: ./models/ntnu/<preset-name>
#   NUM_EPOCHS     — override epoch count (e.g. for larger custom datasets)
#   LEARNING_RATE  — override learning rate

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET_PATH="${DATASET_PATH:-./data/ntnu_dataset.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"

list_presets() {
    cat <<'EOF'
Available training presets (QLoRA 4-bit, bfloat16, ChatML, effective batch ≈ 16):

  smoke-4b          Any GPU          Qwen3.5-4B   1-epoch pipeline check
  smoke-9b          12 GB+ GPU       Qwen3.5-9B   1-epoch pipeline check
  cpu-smoke         CPU only         Qwen3.5-0.8B Verify install without GPU
  gpu-8gb-4b        6–8 GB           Qwen3.5-4B   Entry-level consumer GPU
  gpu-12gb-4b       12 GB            Qwen3.5-4B   Balanced 4B on 12 GB
  gpu-12gb-9b       12 GB            Qwen3.5-9B   RTX 5070 Ti 12 GB laptop
  gpu-12gb-9b-long  12 GB            Qwen3.5-9B   Longer seq; attention-only LoRA
  gpu-16gb-9b       16 GB            Qwen3.5-9B   Full 7-layer LoRA, r=32
  gpu-24gb-4b       24 GB            Qwen3.5-4B   High-quality 4B, long context
  gpu-24gb-9b       24 GB            Qwen3.5-9B   Recommended desktop 9B

Environment overrides:
  DATASET_PATH=./data/my.jsonl   Custom dataset
  OUTPUT_DIR=./models/my-run      Custom output directory
  NUM_EPOCHS=2                    Override epochs (see README dataset-size table)
  LEARNING_RATE=1.5e-4            Override learning rate

Bundled NTNU dataset: ./data/ntnu_dataset.jsonl (~1,252 examples → 3 epochs, 2e-4).
For 30K+ examples use NUM_EPOCHS=1–2 and LEARNING_RATE=1.5e-4 instead.
EOF
}

run_train() {
    # shellcheck disable=SC2068
    python src/train.py "$@"
}

apply_overrides() {
    local -n args_ref=$1
    if [[ -n "$NUM_EPOCHS" ]]; then
        args_ref+=(--num_epochs "$NUM_EPOCHS")
    fi
    if [[ -n "$LEARNING_RATE" ]]; then
        args_ref+=(--learning_rate "$LEARNING_RATE")
    fi
}

PRESET="${1:-}"
if [[ -z "$PRESET" || "$PRESET" == "--list" || "$PRESET" == "-l" || "$PRESET" == "list" ]]; then
    list_presets
    exit 0
fi

COMMON=(
    --dataset_path "$DATASET_PATH"
    --dataset_format chat
    --device gpu
    --load_in_4bit True
    --torch_dtype bfloat16
    --lora_dropout 0.05
    --warmup_ratio 0.03
    --test_split 0.05
    --logging_steps 10
)

case "$PRESET" in
    smoke-4b)
        OUT="${OUTPUT_DIR:-./models/ntnu/smoke-4b}"
        ARGS=(
            "${COMMON[@]}"
            --model_id Qwen/Qwen3.5-4B
            --max_seq_length 2048
            --num_epochs 1
            --per_device_batch_size 1
            --gradient_accum_steps 8
            --learning_rate 2e-4
            --lora_r 16 --lora_alpha 32
            --save_steps 200
            --output_dir "$OUT"
        )
        apply_overrides ARGS
        run_train "${ARGS[@]}"
        ;;
    smoke-9b)
        OUT="${OUTPUT_DIR:-./models/ntnu/smoke-9b}"
        ARGS=(
            "${COMMON[@]}"
            --model_id Qwen/Qwen3.5-9B
            --max_seq_length 1536
            --num_epochs 1
            --per_device_batch_size 1
            --gradient_accum_steps 8
            --learning_rate 2e-4
            --lora_r 16 --lora_alpha 32
            --save_steps 200
            --output_dir "$OUT"
        )
        apply_overrides ARGS
        run_train "${ARGS[@]}"
        ;;
    cpu-smoke)
        OUT="${OUTPUT_DIR:-./models/ntnu/cpu-smoke}"
        run_train \
            --model_id Qwen/Qwen3.5-0.8B \
            --dataset_path "$DATASET_PATH" \
            --dataset_format chat \
            --device cpu \
            --load_in_4bit False \
            --torch_dtype float32 \
            --max_seq_length 512 \
            --num_epochs "${NUM_EPOCHS:-1}" \
            --per_device_batch_size 1 \
            --gradient_accum_steps 4 \
            --learning_rate "${LEARNING_RATE:-2e-4}" \
            --lora_r 8 --lora_alpha 16 \
            --lora_dropout 0.05 \
            --test_split 0.05 \
            --logging_steps 5 \
            --save_steps 50 \
            --output_dir "$OUT"
        ;;
    gpu-8gb-4b)
        OUT="${OUTPUT_DIR:-./models/ntnu/gpu-8gb-4b}"
        ARGS=(
            "${COMMON[@]}"
            --model_id Qwen/Qwen3.5-4B
            --max_seq_length 2048
            --num_epochs 3
            --per_device_batch_size 1
            --gradient_accum_steps 16
            --learning_rate 2e-4
            --lora_r 16 --lora_alpha 32
            --save_steps 200
            --output_dir "$OUT"
        )
        apply_overrides ARGS
        run_train "${ARGS[@]}"
        ;;
    gpu-12gb-4b)
        OUT="${OUTPUT_DIR:-./models/ntnu/gpu-12gb-4b}"
        ARGS=(
            "${COMMON[@]}"
            --model_id Qwen/Qwen3.5-4B
            --max_seq_length 2048
            --num_epochs 3
            --per_device_batch_size 2
            --gradient_accum_steps 8
            --learning_rate 2e-4
            --lora_r 16 --lora_alpha 32
            --save_steps 200
            --output_dir "$OUT"
        )
        apply_overrides ARGS
        run_train "${ARGS[@]}"
        ;;
    gpu-12gb-9b)
        OUT="${OUTPUT_DIR:-./models/ntnu/gpu-12gb-9b}"
        ARGS=(
            "${COMMON[@]}"
            --model_id Qwen/Qwen3.5-9B
            --max_seq_length 1536
            --num_epochs 3
            --per_device_batch_size 1
            --gradient_accum_steps 16
            --learning_rate 2e-4
            --lora_r 16 --lora_alpha 32
            --save_steps 250
            --output_dir "$OUT"
        )
        apply_overrides ARGS
        run_train "${ARGS[@]}"
        ;;
    gpu-12gb-9b-long)
        OUT="${OUTPUT_DIR:-./models/ntnu/gpu-12gb-9b-long}"
        ARGS=(
            "${COMMON[@]}"
            --model_id Qwen/Qwen3.5-9B
            --max_seq_length 2048
            --num_epochs 3
            --per_device_batch_size 1
            --gradient_accum_steps 16
            --learning_rate 2e-4
            --lora_r 16 --lora_alpha 32
            --target_modules q_proj,k_proj,v_proj,o_proj
            --save_steps 250
            --output_dir "$OUT"
        )
        apply_overrides ARGS
        run_train "${ARGS[@]}"
        ;;
    gpu-16gb-9b)
        OUT="${OUTPUT_DIR:-./models/ntnu/gpu-16gb-9b}"
        ARGS=(
            "${COMMON[@]}"
            --model_id Qwen/Qwen3.5-9B
            --max_seq_length 2048
            --num_epochs 3
            --per_device_batch_size 1
            --gradient_accum_steps 16
            --learning_rate 2e-4
            --lora_r 32 --lora_alpha 64
            --save_steps 300
            --output_dir "$OUT"
        )
        apply_overrides ARGS
        run_train "${ARGS[@]}"
        ;;
    gpu-24gb-4b)
        OUT="${OUTPUT_DIR:-./models/ntnu/gpu-24gb-4b}"
        ARGS=(
            "${COMMON[@]}"
            --model_id Qwen/Qwen3.5-4B
            --max_seq_length 4096
            --num_epochs 3
            --per_device_batch_size 2
            --gradient_accum_steps 8
            --learning_rate 2e-4
            --lora_r 32 --lora_alpha 64
            --save_steps 300
            --output_dir "$OUT"
        )
        apply_overrides ARGS
        run_train "${ARGS[@]}"
        ;;
    gpu-24gb-9b)
        OUT="${OUTPUT_DIR:-./models/ntnu/gpu-24gb-9b}"
        ARGS=(
            "${COMMON[@]}"
            --model_id Qwen/Qwen3.5-9B
            --max_seq_length 2048
            --num_epochs 3
            --per_device_batch_size 2
            --gradient_accum_steps 8
            --learning_rate 2e-4
            --lora_r 32 --lora_alpha 64
            --save_steps 300
            --output_dir "$OUT"
        )
        apply_overrides ARGS
        run_train "${ARGS[@]}"
        ;;
    *)
        echo "Unknown preset: $PRESET" >&2
        echo "" >&2
        list_presets >&2
        exit 1
        ;;
esac
