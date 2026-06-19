# Named QLoRA training presets for common GPU tiers and model sizes.
#
# Usage:
#   powershell -File scripts/train_preset.ps1 -List
#   powershell -File scripts/train_preset.ps1 -Preset gpu-12gb-4b
#   $env:DATASET_PATH = ".\data\my.jsonl"; powershell -File scripts/train_preset.ps1 -Preset gpu-24gb-9b
#
# Optional overrides:
#   $env:DATASET_PATH, $env:OUTPUT_DIR, $env:NUM_EPOCHS, $env:LEARNING_RATE

param(
    [string]$Preset = "",
    [switch]$List
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not $env:DATASET_PATH) { $env:DATASET_PATH = ".\data\ntnu_dataset.jsonl" }

function Show-Presets {
    @"
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
  `$env:DATASET_PATH, `$env:OUTPUT_DIR, `$env:NUM_EPOCHS, `$env:LEARNING_RATE

Bundled NTNU dataset: .\data\ntnu_dataset.jsonl (~1,252 examples → 3 epochs, 2e-4).
For 30K+ examples use NUM_EPOCHS=1–2 and LEARNING_RATE=1.5e-4 instead.
"@
}

function Invoke-Train {
    param([string[]]$Args)
    & python src/train.py @Args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Add-Overrides {
    param([System.Collections.Generic.List[string]]$Args)
    if ($env:NUM_EPOCHS) {
        $Args.Add("--num_epochs"); $Args.Add($env:NUM_EPOCHS)
    }
    if ($env:LEARNING_RATE) {
        $Args.Add("--learning_rate"); $Args.Add($env:LEARNING_RATE)
    }
}

if ($List -or -not $Preset) {
    Show-Presets
    exit 0
}

$Common = @(
    "--dataset_path", $env:DATASET_PATH,
    "--dataset_format", "chat",
    "--device", "gpu",
    "--load_in_4bit", "True",
    "--torch_dtype", "bfloat16",
    "--lora_dropout", "0.05",
    "--warmup_ratio", "0.03",
    "--test_split", "0.05",
    "--logging_steps", "10"
)

switch ($Preset) {
    "smoke-4b" {
        $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\models\ntnu\smoke-4b" }
        $Args = [System.Collections.Generic.List[string]]@(
            $Common +
            @(
                "--model_id", "Qwen/Qwen3.5-4B",
                "--max_seq_length", "2048",
                "--num_epochs", "1",
                "--per_device_batch_size", "1",
                "--gradient_accum_steps", "8",
                "--learning_rate", "2e-4",
                "--lora_r", "16", "--lora_alpha", "32",
                "--save_steps", "200",
                "--output_dir", $Out
            )
        )
        Add-Overrides $Args
        Invoke-Train $Args
    }
    "smoke-9b" {
        $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\models\ntnu\smoke-9b" }
        $Args = [System.Collections.Generic.List[string]]@(
            $Common +
            @(
                "--model_id", "Qwen/Qwen3.5-9B",
                "--max_seq_length", "1536",
                "--num_epochs", "1",
                "--per_device_batch_size", "1",
                "--gradient_accum_steps", "8",
                "--learning_rate", "2e-4",
                "--lora_r", "16", "--lora_alpha", "32",
                "--save_steps", "200",
                "--output_dir", $Out
            )
        )
        Add-Overrides $Args
        Invoke-Train $Args
    }
    "cpu-smoke" {
        $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\models\ntnu\cpu-smoke" }
        $Epochs = if ($env:NUM_EPOCHS) { $env:NUM_EPOCHS } else { "1" }
        $Lr = if ($env:LEARNING_RATE) { $env:LEARNING_RATE } else { "2e-4" }
        Invoke-Train @(
            "--model_id", "Qwen/Qwen3.5-0.8B",
            "--dataset_path", $env:DATASET_PATH,
            "--dataset_format", "chat",
            "--device", "cpu",
            "--load_in_4bit", "False",
            "--torch_dtype", "float32",
            "--max_seq_length", "512",
            "--num_epochs", $Epochs,
            "--per_device_batch_size", "1",
            "--gradient_accum_steps", "4",
            "--learning_rate", $Lr,
            "--lora_r", "8", "--lora_alpha", "16",
            "--lora_dropout", "0.05",
            "--test_split", "0.05",
            "--logging_steps", "5",
            "--save_steps", "50",
            "--output_dir", $Out
        )
    }
    "gpu-8gb-4b" {
        $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\models\ntnu\gpu-8gb-4b" }
        $Args = [System.Collections.Generic.List[string]]@(
            $Common +
            @(
                "--model_id", "Qwen/Qwen3.5-4B",
                "--max_seq_length", "2048",
                "--num_epochs", "3",
                "--per_device_batch_size", "1",
                "--gradient_accum_steps", "16",
                "--learning_rate", "2e-4",
                "--lora_r", "16", "--lora_alpha", "32",
                "--save_steps", "200",
                "--output_dir", $Out
            )
        )
        Add-Overrides $Args
        Invoke-Train $Args
    }
    "gpu-12gb-4b" {
        $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\models\ntnu\gpu-12gb-4b" }
        $Args = [System.Collections.Generic.List[string]]@(
            $Common +
            @(
                "--model_id", "Qwen/Qwen3.5-4B",
                "--max_seq_length", "2048",
                "--num_epochs", "3",
                "--per_device_batch_size", "2",
                "--gradient_accum_steps", "8",
                "--learning_rate", "2e-4",
                "--lora_r", "16", "--lora_alpha", "32",
                "--save_steps", "200",
                "--output_dir", $Out
            )
        )
        Add-Overrides $Args
        Invoke-Train $Args
    }
    "gpu-12gb-9b" {
        $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\models\ntnu\gpu-12gb-9b" }
        $Args = [System.Collections.Generic.List[string]]@(
            $Common +
            @(
                "--model_id", "Qwen/Qwen3.5-9B",
                "--max_seq_length", "1536",
                "--num_epochs", "3",
                "--per_device_batch_size", "1",
                "--gradient_accum_steps", "16",
                "--learning_rate", "2e-4",
                "--lora_r", "16", "--lora_alpha", "32",
                "--save_steps", "250",
                "--output_dir", $Out
            )
        )
        Add-Overrides $Args
        Invoke-Train $Args
    }
    "gpu-12gb-9b-long" {
        $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\models\ntnu\gpu-12gb-9b-long" }
        $Args = [System.Collections.Generic.List[string]]@(
            $Common +
            @(
                "--model_id", "Qwen/Qwen3.5-9B",
                "--max_seq_length", "2048",
                "--num_epochs", "3",
                "--per_device_batch_size", "1",
                "--gradient_accum_steps", "16",
                "--learning_rate", "2e-4",
                "--lora_r", "16", "--lora_alpha", "32",
                "--target_modules", "q_proj,k_proj,v_proj,o_proj",
                "--save_steps", "250",
                "--output_dir", $Out
            )
        )
        Add-Overrides $Args
        Invoke-Train $Args
    }
    "gpu-16gb-9b" {
        $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\models\ntnu\gpu-16gb-9b" }
        $Args = [System.Collections.Generic.List[string]]@(
            $Common +
            @(
                "--model_id", "Qwen/Qwen3.5-9B",
                "--max_seq_length", "2048",
                "--num_epochs", "3",
                "--per_device_batch_size", "1",
                "--gradient_accum_steps", "16",
                "--learning_rate", "2e-4",
                "--lora_r", "32", "--lora_alpha", "64",
                "--save_steps", "300",
                "--output_dir", $Out
            )
        )
        Add-Overrides $Args
        Invoke-Train $Args
    }
    "gpu-24gb-4b" {
        $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\models\ntnu\gpu-24gb-4b" }
        $Args = [System.Collections.Generic.List[string]]@(
            $Common +
            @(
                "--model_id", "Qwen/Qwen3.5-4B",
                "--max_seq_length", "4096",
                "--num_epochs", "3",
                "--per_device_batch_size", "2",
                "--gradient_accum_steps", "8",
                "--learning_rate", "2e-4",
                "--lora_r", "32", "--lora_alpha", "64",
                "--save_steps", "300",
                "--output_dir", $Out
            )
        )
        Add-Overrides $Args
        Invoke-Train $Args
    }
    "gpu-24gb-9b" {
        $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { ".\models\ntnu\gpu-24gb-9b" }
        $Args = [System.Collections.Generic.List[string]]@(
            $Common +
            @(
                "--model_id", "Qwen/Qwen3.5-9B",
                "--max_seq_length", "2048",
                "--num_epochs", "3",
                "--per_device_batch_size", "2",
                "--gradient_accum_steps", "8",
                "--learning_rate", "2e-4",
                "--lora_r", "32", "--lora_alpha", "64",
                "--save_steps", "300",
                "--output_dir", $Out
            )
        )
        Add-Overrides $Args
        Invoke-Train $Args
    }
    default {
        Write-Error "Unknown preset: $Preset"
        Show-Presets
        exit 1
    }
}
