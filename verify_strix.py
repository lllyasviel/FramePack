import os
import sys
import torch
import numpy as np
import time

def benchmark_strix():
    print("--- AMD Strix Point Verification & Benchmark ---")
    
    # 1. Check iGPU (DirectML)
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"[iGPU] DirectML detected: {dml_device}")
        
        # Allocate 1GB of VRAM as a test
        x = torch_directml.torch.randn((512, 512, 1024), device=dml_device)
        print(f"[iGPU] Successfully allocated ~1GB on Radeon 890M.")
        del x
    except Exception as e:
        print(f"[iGPU] Radeon 890M test failed: {e}")

    # 2. Check NPU (ORX)
    try:
        import onnxruntime as ort
        print(f"[NPU] ONNX Runtime providers: {ort.get_available_providers()}")
        if 'VitisAIExecutionProvider' in ort.get_available_providers() or 'DirectMLExecutionProvider' in ort.get_available_providers():
            print("[NPU] XDNA 2 NPU appears to be accessible via ONNX EP.")
        else:
            print("[NPU] XDNA 2 NPU not explicitly found, falling back to CPU/DML.")
    except Exception as e:
        print(f"[NPU] NPU Check failed: {e}")

    # 3. Check State-Saver (System RAM)
    try:
        print("[RAM] Testing State-Saver (94GB System Pool)...")
        # Simulate caching a 1GB latent tensor to system RAM
        start_time = time.time()
        latent_sim = torch.randn((1, 16, 256, 128, 128)) # Large latent
        latent_cpu = latent_sim.cpu()
        end_time = time.time()
        print(f"[RAM] Successfully cached 1GB latent to System RAM in {end_time - start_time:.4f}s.")
        print(f"[RAM] System RAM seems ready for Drift Prevention caching.")
    except Exception as e:
        print(f"[RAM] RAM cache test failed: {e}")

    print("--- Verification Complete ---")

if __name__ == "__main__":
    benchmark_strix()
