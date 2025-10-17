#!/bin/bash
# 设置MPS回退环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 运行演示程序
uv run python demo_gradio.py "$@"
