#!/bin/bash
# Run this on the DGX HOST (sorenle@spark-d0bd), NOT inside the container
# Usage: bash start_host.sh

CONTAINER="vllm-container"
CONTAINER_IP="172.17.0.2"
HOST_PORT="8080"
MODEL=${1:-"mistralai/Ministral-3-14B-Instruct-2512"}

echo ""
echo "=== DGX Host Startup ==="
echo ""

# Step 1 — copy startup script into container and run it
echo "[1/2] Starting services inside container..."
docker cp start_container.sh $CONTAINER:/workspace/start_container.sh
docker exec -it $CONTAINER bash /workspace/start_container.sh "$MODEL"

# Step 2 — kill any existing socat on that port and restart
echo "[2/2] Exposing port $HOST_PORT to network..."
pkill -f "socat TCP-LISTEN:$HOST_PORT" 2>/dev/null
sleep 1
nohup socat TCP-LISTEN:$HOST_PORT,fork,reuseaddr TCP:$CONTAINER_IP:$HOST_PORT > /tmp/socat.log 2>&1 &
echo "      socat running (pid $!)"

# Step 3 — open firewall if ufw is active
if sudo ufw status | grep -q "Status: active"; then
    echo "      Opening port $HOST_PORT in firewall..."
    sudo ufw allow $HOST_PORT > /dev/null
fi

echo ""
echo "=== All done! ==="
echo ""
HOST_IP=$(hostname -I | awk '{print $1}')
echo "  Chat UI available at: http://$HOST_IP:$HOST_PORT"
echo ""
echo "  Share this with anyone on the same WiFi ^"
echo ""
