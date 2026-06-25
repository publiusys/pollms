#!/bin/bash
#SBATCH -N 1
#SBATCH -w gpu0004
#SBATCH --partition=GPU
#SBATCH -t 1:30:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

cd ~/pollms/chatbot_h200

#/home/hdong/pollms/chatbot_h200/llama.cpp/build/bin/llama-server -m /home/hdong/pollms/chatbot_h200/models/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 262144 -fa on --parallel 1 --reasoning-format auto

export CTX_SIZE="262144"
export MODEL_PATH="/home/hdong/pollms/chatbot_h200/models/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
export LLAMA_BIN="./llama.cpp/build/bin/llama-server"

# Start llama-server directly
$LLAMA_BIN -m "$MODEL_PATH" --host 127.0.0.1 --port 8080 -c "$CTX_SIZE" -ngl 99 -fa on --parallel 1 --reasoning-format auto &
LLAMA_PID=$!

# Wait for it to be ready
echo "Waiting for $LLAMA_BIN to be ready...."
sleep 10

# Start the proxy server
python3 server_h200.py
echo "Started proxy server..."

# Cleanup
echo "Cleaning up!"
kill $LLAMA_PID

