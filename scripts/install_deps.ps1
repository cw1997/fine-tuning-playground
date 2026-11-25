param(
    [ValidateSet("cuda", "rocm", "mps", "xpu", "cpu")][string]$Backend,
    [switch]$Rocm4Bit
)

$ErrorActionPreference = "Stop"
$InstallArgs = @()
if ($Backend) { $InstallArgs += @("--backend", $Backend.ToLowerInvariant()) }
if ($Rocm4Bit) { $InstallArgs += "--rocm-4bit" }
& python (Join-Path $PSScriptRoot "install_deps.py") @InstallArgs
exit $LASTEXITCODE
