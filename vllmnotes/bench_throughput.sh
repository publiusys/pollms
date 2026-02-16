#!/bin/bash -l 

# export HF_TOKEN="YOUR_HF_TOKEN"
# export HF_HOME="/vast/users/sraskar/mi250/hf/hub"
# export HF_DATASETS_CACHE="/vast/users/sraskar/mi250/hf/hub"

# module load cuda/12.3.0
# source ~/.init_conda_x86.sh
# conda activate h100_vllm

# cd /vast/users/sraskar/h100/llm_research/vllm/benchmarks

# models list that dont have readily available models for vllm
# "meta-llama/Llama-2-7b-hf"
# "meta-llama/Meta-Llama-3-8B"
# "meta-llama/Llama-2-70b-hf"
# "meta-llama/Meta-Llama-3-70B"
# "mistralai/Mistral-7B-v0.1"
# "mistralai/Mixtral-8x7B-v0.1"
models=(
"Qwen/Qwen2-7B"
"Qwen/Qwen2-72B"
)

for model in "${models[@]}"; do
    for tensor_parallel in 1 2 4; do
        for batch_size in 1 16 32 64; do
            for input_output_length in 128 256 512 1024 2048; do
                vllm bench latency \
                    --batch-size=$batch_size \
                    --tensor-parallel-size=$tensor_parallel \
                    --input-len=$input_output_length \
                    --output-len=$input_output_length \
                    --model=$model \
                    --output-json="results_latency_${model_name}_tp${tensor_parallel}_io${input_output_length}_bs${batch_size}.json"
            done
        done
    done
done