import os
import torch
import numpy as np
import time

# 设置MPS回退环境变量，以处理未实现的操作
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from diffusers_helper.memory import get_mps_free_memory_gb, mps
from diffusers_helper.models.hunyuan_video_packed import chunked_attention_bfloat16, mps_attn_varlen_func

def test_mps_attention():
    """
    Test MPS attention mechanisms implemented in PR #65
    """
    print("Testing MPS attention mechanisms for FramePack")

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("MPS is not available on this device")
        return False

    print(f"MPS is available. Device: {mps}")
    print(f"Free MPS memory: {get_mps_free_memory_gb(mps):.2f} GB")

    try:
        # Test parameters
        batch_size = 1
        seq_len = 64
        hidden_size = 4096
        num_heads = 32
        head_dim = hidden_size // num_heads

        # Create test tensors
        print(f"Creating test tensors with shape [batch={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]")
        query = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16).to(mps)
        key = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16).to(mps)
        value = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16).to(mps)

        # Test regular attention
        print("Testing regular attention...")
        start_time = time.time()

        # Reshape for multi-head attention
        q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        regular_time = time.time() - start_time
        print(f"Regular attention completed in {regular_time:.4f} seconds")
        print(f"Output shape: {attn_output.shape}")

        # Test chunked attention
        print("\nTesting chunked attention...")
        start_time = time.time()

        # Reshape for multi-head attention format expected by chunked_attention_bfloat16
        q_reshaped = query.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k_reshaped = key.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v_reshaped = value.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # Test with different chunk sizes
        for chunk_size in [16, 32]:
            print(f"Testing with chunk_size={chunk_size}")
            chunked_output_reshaped = chunked_attention_bfloat16(
                q_reshaped, k_reshaped, v_reshaped, chunk_size=chunk_size
            )

            # Convert back to original shape
            chunked_output = chunked_output_reshaped.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)

            # Verify output shape
            assert chunked_output.shape == (batch_size, seq_len, hidden_size), f"Expected shape {(batch_size, seq_len, hidden_size)}, got {chunked_output.shape}"

            # Verify output values (should be close to regular attention)
            if chunk_size == 32:  # Only check for the larger chunk size
                error = torch.abs(chunked_output - attn_output).mean().item()
                print(f"Mean absolute error between regular and chunked attention: {error:.6f}")
                assert error < 1e-2, f"Error too large: {error}"

        chunked_time = time.time() - start_time
        print(f"Chunked attention tests completed in {chunked_time:.4f} seconds")

        # Test MPS variable length attention
        print("\nTesting MPS variable length attention...")
        start_time = time.time()

        # Create variable length inputs
        var_seq_lens = [32, 48, 64]
        max_seq_len = max(var_seq_lens)

        batch_size = len(var_seq_lens)
        var_query = torch.randn(batch_size, max_seq_len, num_heads, head_dim, dtype=torch.bfloat16).to(mps)
        var_key = torch.randn(batch_size, max_seq_len, num_heads, head_dim, dtype=torch.bfloat16).to(mps)
        var_value = torch.randn(batch_size, max_seq_len, num_heads, head_dim, dtype=torch.bfloat16).to(mps)

        # Test MPS variable length attention
        var_output = mps_attn_varlen_func(
            var_query, var_key, var_value, chunk_size=32
        )

        # Verify output shape
        expected_shape = (batch_size, max_seq_len, num_heads, head_dim)
        assert var_output.shape == expected_shape, f"Expected shape {expected_shape}, got {var_output.shape}"

        var_time = time.time() - start_time
        print(f"Variable length attention completed in {var_time:.4f} seconds")

        print("\nAll MPS attention tests passed!")
        return True
    except Exception as e:
        print(f"Failed to test MPS attention: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mps_attention()
    print(f"\nMPS attention test {'PASSED' if success else 'FAILED'}")
