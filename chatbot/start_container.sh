#!/bin/bash
# Run this INSIDE the vllm-docker container
# Usage: bash start_container.sh

MODEL=${1:-"mistralai/Ministral-3-14B-Instruct-2512"}

echo ""
echo "=== vLLM Chat Startup ==="
echo "Model: $MODEL"
echo ""

# Kill existing screens if they exist
screen -S vllm -X quit 2>/dev/null
screen -S webui -X quit 2>/dev/null
sleep 1

# Start vLLM in a screen
echo "[1/2] Starting vLLM..."
screen -dmS vllm bash -c "vllm serve '$MODEL' --host 0.0.0.0 --port 8000 2>&1 | tee /workspace/vllm.log"

# Wait for vLLM to be ready
echo "      Waiting for vLLM to be ready (this takes a minute)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "      vLLM is up!"
        break
    fi
    sleep 5
    echo "      ... still waiting ($((i*5))s)"
done

# Start Flask proxy in a screen
echo "[2/2] Starting Flask proxy..."
screen -dmS webui bash -c "
    source /workspace/venv/bin/activate
    cd /workspace/webui
    VLLM_BASE=http://localhost:8000 python3 server.py 2>&1 | tee /workspace/webui.log
"

sleep 2

echo ""
echo "=== Done! ==="
echo ""
echo "Screens running:"
screen -ls
echo ""
echo "To check logs:"
echo "  vLLM:  screen -r vllm"
echo "  Flask: screen -r webui"
echo ""
echo "Flask is on: http://172.17.0.2:8080"
echo "(Run start_host.sh on the DGX host to expose to network)"
echo ""
