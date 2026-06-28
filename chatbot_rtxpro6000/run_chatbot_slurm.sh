#!/bin/bash
#SBATCH -N 1
#SBATCH -w cs-hpc-node-6
#SBATCH -t 30-00:00:00
#SBATCH --mem=124GB
#SBATCH --cpus-per-task=25
#SBATCH --gres=gpu:1

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd ~/pollms/chatbot_rtxpro6000

export CTX_SIZE="262144"
export MODEL_PATH="/hpc/hdong/models/Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
export LLAMA_BIN="./llama.cpp/build/bin/llama-server"

# Restrict to only GPU 0 (TBD need to be fix via SLURM automatic binding)
export CUDA_VISIBLE_DEVICES=1

# Start llama-server directly
$LLAMA_BIN -m "$MODEL_PATH" --host 127.0.0.1 --port 8080 -c "$CTX_SIZE" -ngl 99 -fa on --parallel 1 --reasoning-format auto &
#LLAMA_PID=$!

# Wait for it to be ready
echo "Waiting for $LLAMA_BIN to be ready...."
sleep 10

# Start the proxy server
python3 server_rtxpro6000.py
echo "Started proxy server..."

# Cleanup
echo "Cleaning up!"
kill $LLAMA_PID

