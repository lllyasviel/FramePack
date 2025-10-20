#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p logs
log_file="./logs/${env_name}_$(date +%Y%m%d_%H%M%S).log"
echo $log_file

torch_compile_mode="max-autotune-no-cudagraphs"
torchrun --nproc_per_node=2 --master_port=29509 inference.py --ulysses_degree 2 --ring_degree 1 --port 8098 --torch_compile_mode ${torch_compile_mode} 1>$log_file 2>&1 &

