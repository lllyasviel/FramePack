import os
import torch
import argparse

# 设置MPS回退环境变量，以处理未实现的操作
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import get_mps_free_memory_gb, mps

def test_mps_support():
    print("Testing MPS support for FramePack")

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("MPS is not available on this device")
        return False

    print(f"MPS is available. Device: {mps}")
    print(f"Free MPS memory: {get_mps_free_memory_gb(mps):.2f} GB")

    # Test creating a small tensor on MPS
    try:
        test_tensor = torch.ones(10, 10).to(mps)
        print(f"Test tensor created on MPS: {test_tensor.device}")
    except Exception as e:
        print(f"Failed to create tensor on MPS: {e}")
        return False

    # Test loading a small model
    try:
        print("Loading transformer model...")
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            'lllyasviel/FramePackI2V_HY',
            torch_dtype=torch.bfloat16
        ).to(mps)
        print(f"Model loaded successfully on {transformer.device}")

        # Test forward pass with a small input
        print("Testing forward pass with small input...")
        batch_size = 1
        channels = 16
        frames = 4
        height = 64
        width = 64

        # Create dummy inputs
        hidden_states = torch.randn(batch_size, channels, frames, height, width).to(mps, dtype=torch.bfloat16)
        timestep = torch.tensor([1000]).to(mps)
        encoder_hidden_states = torch.randn(batch_size, 10, 4096).to(mps, dtype=torch.bfloat16)
        encoder_attention_mask = torch.ones(batch_size, 10).to(mps, dtype=torch.bool)
        pooled_projections = torch.randn(batch_size, 768).to(mps, dtype=torch.bfloat16)
        guidance = torch.tensor([10000.0]).to(mps, dtype=torch.bfloat16)
        image_embeddings = torch.randn(batch_size, 10, 1152).to(mps, dtype=torch.bfloat16)

        # Run forward pass
        with torch.no_grad():
            output = transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                pooled_projections=pooled_projections,
                guidance=guidance,
                image_embeddings=image_embeddings
            )

        print("Forward pass successful!")
        print(f"Output shape: {output.sample.shape}")

        return True
    except Exception as e:
        print(f"Failed to test model on MPS: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mps_support()
    print(f"\nMPS support test {'PASSED' if success else 'FAILED'}")
