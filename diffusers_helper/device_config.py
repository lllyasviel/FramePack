import torch
import os

def get_device():
    """Get the optimal device for the current system."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def configure_m3_optimizations():
    """Configure M3-specific optimizations."""
    if torch.backends.mps.is_available():
        # Enable Metal Performance Shaders
        torch.backends.mps.enable_fallback_kernels = True
        # Set conservative memory settings
        high_vram = False
        gpu_memory_preservation = 8
        return high_vram, gpu_memory_preservation
    return None, None

def get_available_memory():
    """Get available memory for the current device."""
    if torch.backends.mps.is_available():
        # MPS doesn't provide direct memory info
        # Return conservative estimate for M3
        return 32  # GB
    elif torch.cuda.is_available():
        from diffusers_helper.memory import get_cuda_free_memory_gb
        return get_cuda_free_memory_gb(torch.device("cuda"))
    return 8  # Conservative CPU estimate

def handle_device_specific_errors(e, device):
    """Handle device-specific errors."""
    if device.type == "mps":
        if "out of memory" in str(e):
            # Handle MPS OOM
            torch.mps.empty_cache()
            return True
    return False

def get_optimal_batch_size(device):
    """Get optimal batch size for the current device."""
    if device.type == "mps":
        return 4  # Conservative batch size for M3
    return 8  # Default batch size

def get_model_dtype():
    """Get optimal model dtype for the current device."""
    if torch.backends.mps.is_available():
        return torch.float16  # M3 prefers float16
    return torch.bfloat16  # Default to bfloat16 