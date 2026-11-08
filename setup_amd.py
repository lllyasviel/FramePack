import os
import subprocess
import sys

def run_command(command):
    print(f"Executing: {command}")
    subprocess.check_call(command, shell=True)

def setup_amd_env():
    print("--- AMD Strix Point Native Setup for FramePack ---")
    
    # 1. Create Virtual Environment
    if not os.path.exists(".venv"):
        run_command(f"{sys.executable} -m venv .venv")
    
    venv_python = os.path.abspath(os.path.join(".venv", "Scripts", "python.exe"))
    venv_pip = os.path.abspath(os.path.join(".venv", "Scripts", "pip.exe"))

    # 2. Upgrade Pip
    run_command(f'"{venv_pip}" install --upgrade pip')

    # 3. Install DirectML and AMD-specific tensor libraries
    # We force torch-directml and skip CUDA requirements
    print("Installing torch-directml and onnxruntime-directml...")
    run_command(f'"{venv_pip}" install torch-directml torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu')
    run_command(f'"{venv_pip}" install onnxruntime-directml')

    # 4. Install FramePack requirements (filtering out CUDA/NVIDIA specifics)
    print("Installing FramePack dependencies...")
    run_command(f'"{venv_pip}" install -r requirements.txt')

    # 5. Set Environment Variables for DirectML bypass
    os.environ["USE_DIRECTML"] = "1"
    os.environ["ORT_DIRECTML_DEVICE_ID"] = "0" # Radeon 890M usually ID 0
    
    print("--- AMD Native Setup Complete ---")
    print("Test with: .venv\\Scripts\\python.exe setup_amd.py --test-dml")

def test_dml():
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Success! DirectML Device detected: {device}")
        # Test a small tensor operation
        x = torch_directml.torch.tensor([1.0, 2.0], device=device)
        print(f"Tensor on DML: {x}")
    except Exception as e:
        print(f"DML Test Failed: {e}")

if __name__ == "__main__":
    if "--test-dml" in sys.argv:
        test_dml()
    else:
        setup_amd_env()
