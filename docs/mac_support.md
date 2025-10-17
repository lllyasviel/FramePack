# FramePack on Apple Silicon Macs

This document provides instructions for running FramePack on Apple Silicon Macs (M1, M2, M3, M4 series).

## Requirements

- Apple Silicon Mac (M1, M2, M3, M4 series)
- macOS 12.0 or later
- Python 3.10 or later
- At least 16GB RAM (32GB+ recommended for better performance)

## Installation

1. Install PyTorch with MPS support:

```bash
pip install torch torchvision
```

2. Clone the FramePack repository:

```bash
git clone https://github.com/lllyasviel/FramePack.git
cd FramePack
```

3. Install dependencies:

```bash
pip install -e .
```

## Running FramePack on MPS

FramePack automatically detects and uses MPS (Metal Performance Shaders) when running on Apple Silicon Macs. No additional configuration is needed.

### Verifying MPS Support

You can verify that MPS is working correctly by running the test scripts:

```bash
python test_mps_basic.py
python test_mac_support.py
```

If the tests pass, your system is ready to use FramePack with MPS acceleration.

## Performance Considerations

- **Memory Usage**: Apple Silicon Macs share memory between CPU and GPU. Monitor your memory usage to avoid performance degradation.
- **Batch Size**: You may need to use smaller batch sizes compared to NVIDIA GPUs.
- **Model Size**: Larger models may require more memory. Start with smaller models if you encounter memory issues.

## Known Limitations

- Some operations may be slower on MPS compared to CUDA.
- Not all PyTorch operations are optimized for MPS yet.
- TeaCache may not provide the same performance benefits as on NVIDIA GPUs.

## Troubleshooting

If you encounter issues:

1. **Memory Errors**: Reduce batch size or model size.
2. **Performance Issues**: Try disabling TeaCache with `transformer.initialize_teacache(enable_teacache=False)`.
3. **Compatibility Issues**: Ensure you're using the latest version of PyTorch with MPS support.
4. **Missing Operator Errors**: If you see errors like `NotImplementedError: The operator 'aten::xxx' is not currently implemented for the MPS device`, set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` before running the script:

   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   python demo_gradio.py
   ```

   Or use the provided convenience script:

   ```bash
   ./run_demo.sh
   ```

   This will make PyTorch fall back to CPU for operations not yet implemented on MPS.

## Contributing

If you find ways to improve FramePack's performance on Apple Silicon, please consider contributing to the project by submitting a pull request.
