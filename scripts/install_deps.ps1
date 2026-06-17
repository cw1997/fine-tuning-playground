$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Installing dependencies (CUDA PyTorch by default)..."
pip install -r requirements.txt

python -c @"
import torch
print(f'torch={torch.__version__}, cuda_built={torch.backends.cuda.is_built()}')
if torch.cuda.is_available():
    print(f'cuda_available=True, device={torch.cuda.get_device_name(0)}')
else:
    print('cuda_available=False')
    if torch.version.cuda is None:
        print('Hint: reinstall with: pip install -r requirements.txt')
"@
