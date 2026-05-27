# 1. on your laptop — copy files to DGX
```
scp index.html server.py start_host.sh start_container.sh dgx1:~
```

```
sudo docker run -d -p 2003:22 --hostname vllm-docker -e USERNAME=sorenle -v /tmp/authorized_keys:/tmp/authorized_keys:ro --name vllm-container --gpus=all nvcr.io/nvidia/vllm:26.01-py3 /bin/bash -c "sleep infinity"

```
# 2. SSH in and put the webui files in place

```
ssh dgx1
sudo docker exec -it vllm-container mkdir -p /workspace/webui
sudo docker cp index.html vllm-container:/workspace/webui/index.html
sudo docker cp server.py vllm-container:/workspace/webui/server.py
```
# 3. run the startup script (does everything else)
```
bash start_host.sh
```