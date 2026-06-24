#!/bin/bash
#SBATCH -N 1
#SBATCH -w cs-hpc-node-1
#SBATCH -t 30-00:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:1

cd ~/pollms/chatbot_rtx5070ti

export CTX_SIZE="32768"
export MODEL_PATH="/hpc/hdong/models/Qwen3.5-27B-Q3_K_S.gguf"
export LLAMA_BIN="./llama.cpp/build/bin/llama-server"

# Start llama-server directly
$LLAMA_BIN -m "$MODEL_PATH" --host 127.0.0.1 --port 8080 -c "$CTX_SIZE" -ngl 99 -fa on --parallel 1 --reasoning-format auto &
LLAMA_PID=$!

# Wait for it to be ready
echo "Waiting for $LLAMA_BIN to be ready...."
sleep 10

# Start the proxy server
python3 server_rtx5070ti.py
echo "Started proxy server..."

# Cleanup
echo "Cleaning up!"
kill $LLAMA_PID

