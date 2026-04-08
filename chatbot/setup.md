# 1. on your laptop — copy files to DGX

scp index.html server.py start_host.sh start_container.sh dgx1:~

# 2. SSH in and put the webui files in place

ssh dgx1
sudo docker exec -it vllm-container mkdir -p /workspace/webui
sudo docker cp index.html vllm-container:/workspace/webui/index.html
sudo docker cp server.py vllm-container:/workspace/webui/server.py

# 3. run the startup script (does everything else)

bash start_host.sh
