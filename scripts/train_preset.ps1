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
    [switch]$List,
    [string]$PythonExecutable = "python",
    [ValidateRange(0, 2147483647)][int]$BenchmarkSteps = 0,
    [ValidateRange(0, 2147483647)][int]$BenchmarkWarmupSteps = 3,
    [ValidateSet(1, 2, 4, 8, 16)][int]$MicroBatch,
    [ValidateSet("True", "False")][string]$Checkpointing,
    [switch]$AssistantOnly,
    [ValidateSet("nll", "chunked_nll")][string]$LossType,
    [string]$RunMetrics
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not $env:DATASET_PATH) { $env:DATASET_PATH = ".\data\ntnu_dataset.jsonl" }

function Show-Presets {
    @"
Training presets (GPU QLoRA requires CUDA or compatible ROCm):

  smoke-4b          CUDA / ROCm      Qwen3.5-4B   1-epoch pipeline check
  smoke-9b          12 GB+ GPU       Qwen3.5-9B   1-epoch pipeline check
  cpu-smoke         CPU only         Qwen3.5-0.8B Verify install without GPU
  gpu-8gb-4b        6–8 GB           Qwen3.5-4B   Entry-level consumer GPU
  gpu-12gb-4b       12 GB            Qwen3.5-4B   Balanced 4B on 12 GB
  gpu-12gb-9b       12 GB            Qwen3.5-9B   RTX 5070 Ti 12 GB laptop
  gpu-12gb-9b-long  12 GB            Qwen3.5-9B   Longer seq; attention-only LoRA
  gpu-16gb-9b       16 GB            Qwen3.5-9B   Full 7-layer LoRA, r=32
  gpu-24gb-4b       24 GB            Qwen3.5-4B   High-quality 4B, long context
  gpu-24gb-9b       24 GB            Qwen3.5-9B   Recommended desktop 9B

Native Windows NTNU comparison (12 GB, BF16 compute):
  ntnu-win-baseline    Qwen3.5-4B NF4, 2 x 8, 3 epochs
  ntnu-win-qwen35-4b   Qwen3.5-4B NF4, 4 x 4, 2 epochs
  ntnu-win-qwen3-4b    Qwen3-4B-Instruct-2507 NF4, 4 x 4, 2 epochs
  ntnu-win-qwen35-2b   Qwen3.5-2B BF16, 8 x 2, 2 epochs

Windows comparison options:
  -PythonExecutable, -BenchmarkSteps, -BenchmarkWarmupSteps, -MicroBatch, -Checkpointing,
  -AssistantOnly, -LossType, -RunMetrics

Environment overrides:
  `$env:DATASET_PATH, `$env:OUTPUT_DIR, `$env:NUM_EPOCHS, `$env:LEARNING_RATE

Bundled NTNU dataset: .\data\ntnu_dataset.jsonl (~1,252 examples → 3 epochs, 2e-4).
For 30K+ examples use NUM_EPOCHS=1–2 and LEARNING_RATE=1.5e-4 instead.
"@
}

function Invoke-Train {
    param([string[]]$Args)
    & $PythonExecutable -u src/train.py @Args
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

if ($Preset -like "ntnu-win-*") {
    $Settings = switch ($Preset) {
        "ntnu-win-baseline"  { @("Qwen/Qwen3.5-4B", "True", 2, 3) }
        "ntnu-win-qwen35-4b" { @("Qwen/Qwen3.5-4B", "True", 4, 2) }
        "ntnu-win-qwen3-4b"  { @("Qwen/Qwen3-4B-Instruct-2507", "True", 4, 2) }
        "ntnu-win-qwen35-2b" { @("Qwen/Qwen3.5-2B", "False", 8, 2) }
        default { throw "Unknown Windows NTNU preset: $Preset" }
    }
    $IsBaseline = $Preset -eq "ntnu-win-baseline"
    $Batch = if ($MicroBatch) { $MicroBatch } else { $Settings[2] }
    $Accum = 16 / $Batch
    $Epochs = if ($env:NUM_EPOCHS) { $env:NUM_EPOCHS } else { $Settings[3] }
    $Lr = if ($env:LEARNING_RATE) { $env:LEARNING_RATE } else { "2e-4" }
    $Length = if ($IsBaseline) { "2048" } else { "512" }
    if (-not $Checkpointing) { $Checkpointing = $IsBaseline.ToString() }
    if (-not $LossType) { $LossType = "nll" }

    $RunName = if ($BenchmarkSteps) { "$Preset-short" } else { $Preset }
    $Out = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else {
        Join-Path ".\models\ntnu\windows-comparison" $RunName
    }
    if (-not $RunMetrics) { $RunMetrics = Join-Path $Out "metrics.json" }
    $TrainArguments = @(
        "--model_id", $Settings[0], "--load_in_4bit", $Settings[1],
        "--dataset_path", $env:DATASET_PATH, "--device", "gpu",
        "--num_epochs", $Epochs, "--learning_rate", $Lr, "--max_seq_length", $Length,
        "--per_device_batch_size", $Batch, "--gradient_accum_steps", $Accum,
        "--gradient_checkpointing", $Checkpointing, "--output_dir", $Out,
        "--loss_type", $LossType, "--run_metrics", $RunMetrics
    )
    if (-not $IsBaseline) {
        $TrainArguments += @("--group_by_length", "--eval_each_epoch", "--eval_batch_size", $Batch)
    }
    if ($AssistantOnly) { $TrainArguments += "--assistant_only_loss" }
    if ($BenchmarkSteps) {
        $TrainArguments += @(
            "--benchmark_steps", $BenchmarkSteps,
            "--benchmark_warmup_steps", $BenchmarkWarmupSteps
        )
    }
    & $PythonExecutable -u src/train.py @TrainArguments
    exit $LASTEXITCODE
}

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
