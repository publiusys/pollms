#!/bin/bash

models=(
  "Qwen/Qwen2-7B"
  "Qwen/Qwen2-72B"
)

RESULTS_DIR="/workspace/results"
mkdir -p "$RESULTS_DIR"

for model in "${models[@]}"; do
  model_name=$(echo "$model" | tr '/' '_')
  for tensor_parallel in 1 2 4; do
    for batch_size in 1 16 32 64; do
      for input_output_length in 128 256 512 1024 2048; do

        combo="${model_name}_tp${tensor_parallel}_io${input_output_length}_bs${batch_size}"
        power_log="${RESULTS_DIR}/power_${combo}.csv"

        # Start nvidia-smi logging in background
        nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,utilization.gpu \
          --format=csv,nounits -l 1 > "$power_log" &
        NVIDIA_PID=$!

        # Run benchmark
        vllm bench latency \
          --batch-size=$batch_size \
          --tensor-parallel-size=$tensor_parallel \
          --input-len=$input_output_length \
          --output-len=$input_output_length \
          --model=$model \
          --output-json="${RESULTS_DIR}/results_${combo}.json"

        # Stop power logging
        kill $NVIDIA_PID

      done
    done
  done
done