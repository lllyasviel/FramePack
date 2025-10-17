import os
import torch
import numpy as np
from diffusers_helper.memory import get_mps_free_memory_gb, mps

# 设置MPS回退环境变量，以处理未实现的操作
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def test_mps_basic():
    """
    Test basic MPS functionality without requiring model downloads
    """
    print("Testing basic MPS support for FramePack")

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("MPS is not available on this device")
        return False

    print(f"MPS is available. Device: {mps}")
    print(f"Free MPS memory: {get_mps_free_memory_gb(mps):.2f} GB")

    # Test creating tensors on MPS
    try:
        # Create a tensor on CPU
        cpu_tensor = torch.randn(1000, 1000)
        print(f"CPU tensor created with shape {cpu_tensor.shape}")

        # Move tensor to MPS
        mps_tensor = cpu_tensor.to(mps)
        print(f"Tensor moved to MPS: {mps_tensor.device}")

        # Perform some operations on MPS
        result = mps_tensor @ mps_tensor.t()
        print(f"Matrix multiplication on MPS completed with shape {result.shape}")

        # Test different dtypes
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            try:
                dtype_tensor = torch.ones(100, 100, dtype=dtype).to(mps)
                print(f"Successfully created tensor with dtype {dtype} on MPS")
            except Exception as e:
                print(f"Failed to create tensor with dtype {dtype} on MPS: {e}")

        return True
    except Exception as e:
        print(f"Failed to test MPS: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mps_basic()
    print(f"\nMPS basic support test {'PASSED' if success else 'FAILED'}")
