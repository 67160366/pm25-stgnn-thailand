# install_native_deps.ps1
#
# Install PyG native extensions that can NOT be in pyproject.toml because
# they require `torch` to be already installed in the same environment
# at build/install time (uv resolves deps before installing, which breaks
# this).
#
# Run this AFTER `uv sync` succeeds. Re-running is idempotent (pip skips
# packages already installed).
#
# Verified working: 2026-05-18 on Windows 11, Python 3.11, RTX 3060 Ti.

$ErrorActionPreference = "Stop"

Write-Host "=== PyG native extensions installer ===" -ForegroundColor Cyan
Write-Host ""

# --- Verify torch is installed ---
Write-Host "Checking torch is installed..."
$torchInfo = python -c "import torch; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "torch not installed in current environment. Run 'uv sync' first."
    exit 1
}
Write-Host "  Found torch: $torchInfo" -ForegroundColor Green

# --- Detect torch + CUDA tag for PyG wheel index ---
$torchVersion = python -c "import torch; print(torch.__version__.split('+')[0])"
$cudaTag = python -c @"
import torch
if torch.cuda.is_available():
    cuda_str = torch.version.cuda.replace('.', '')
    print(f'cu{cuda_str}')
else:
    print('cpu')
"@

Write-Host "  Torch base version: $torchVersion"
Write-Host "  CUDA tag: $cudaTag"

$wheelIndex = "https://data.pyg.org/whl/torch-${torchVersion}+${cudaTag}.html"
Write-Host "  PyG wheel index: $wheelIndex"
Write-Host ""

# --- Install torch-scatter and torch-sparse from PyG wheel index ---
Write-Host "Installing torch-scatter and torch-sparse..." -ForegroundColor Yellow
python -m pip install torch-scatter torch-sparse --index-url $wheelIndex
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install torch-scatter/torch-sparse. Verify wheels exist at $wheelIndex"
    exit 1
}
Write-Host "  torch-scatter and torch-sparse installed" -ForegroundColor Green
Write-Host ""

# --- Install torch-geometric-temporal AFTER torch-sparse is available ---
# torch-geometric-temporal depends on torch-sparse at runtime/import time
Write-Host "Installing torch-geometric-temporal..." -ForegroundColor Yellow
python -m pip install "torch-geometric-temporal>=0.54"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install torch-geometric-temporal"
    exit 1
}
Write-Host "  torch-geometric-temporal installed" -ForegroundColor Green
Write-Host ""

# --- Final verification ---
Write-Host "Running verification..." -ForegroundColor Cyan
python -c @"
import torch, torch_scatter, torch_sparse
import torch_geometric, torch_geometric_temporal

print(f'  torch:                    {torch.__version__}')
print(f'  CUDA available:           {torch.cuda.is_available()}')
print(f'  torch_scatter:            {torch_scatter.__version__}')
print(f'  torch_sparse:             {torch_sparse.__version__}')
print(f'  torch_geometric:          {torch_geometric.__version__}')
print(f'  torch_geometric_temporal: {torch_geometric_temporal.__version__}')

# Smoke test: scatter on CUDA
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
idx = torch.tensor([0, 1, 0, 1])
if torch.cuda.is_available():
    x = x.cuda()
    idx = idx.cuda()
out = torch_scatter.scatter_add(x, idx, dim=0)
expected = [4.0, 6.0]
actual = out.cpu().tolist()
assert actual == expected, f'scatter test failed: got {actual}, expected {expected}'
print(f'  scatter smoke test:       PASS ({actual})')
"@

Write-Host ""
Write-Host "=== Native deps installed successfully ===" -ForegroundColor Green
