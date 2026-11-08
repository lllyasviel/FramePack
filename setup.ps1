# AMD AI Strix Point Setup Script for FramePack
# Targeted for Ryzen 9 AI + Radeon 890M

$VENV_NAME = ".venv"
$PYTHON_EXEC = "python"

Write-Host "--- Initializing AMD AI Studio Environment ---" -ForegroundColor Cyan

# 1. Create Virtual Environment
if (!(Test-Path $VENV_NAME)) {
    Write-Host "Creating virtual environment..."
    & $PYTHON_EXEC -m venv $VENV_NAME
}

$VENV_PYTHON = "$VENV_NAME\Scripts\python.exe"
$VENV_PIP = "$VENV_NAME\Scripts\pip.exe"

# 2. Upgrade Pip
& $VENV_PIP install --upgrade pip

# 3. Install Core Torch with DirectML support
# Note: we use torch-directml which acts as a provider for torch
Write-Host "Installing torch-directml and dependencies..."
& $VENV_PIP install torch-directml torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install FramePack Requirements
Write-Host "Installing FramePack requirements..."
& $VENV_PIP install -r requirements.txt

# 5. Install ONNX Runtime for NPU/DirectML offloading
Write-Host "Installing ONNX Runtime for AMD NPU..."
& $VENV_PIP install onnxruntime-directml

# 6. Install specific FP8/GGUF support (bitsandbytes for AMD or alternative)
# For AMD, we often use 'bitsandbytes' patched for ROCm/DML or custom loaders
Write-Host "Installing quantization support..."
& $VENV_PIP install bitsandbytes-windows  # Basic support, might need specific AMD patch

# 7. Set Environment Variables for Bypass
[System.Environment]::SetEnvironmentVariable("USE_DIRECTML", "1", "Process")
[System.Environment]::SetEnvironmentVariable("MIOPEN_ENABLE_LOGGING", "0", "Process")

Write-Host "--- Setup Complete! ---" -ForegroundColor Green
Write-Host "To activate, run: .\$VENV_NAME\Scripts\Activate.ps1"
