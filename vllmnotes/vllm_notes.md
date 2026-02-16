sudo docker run -d -p 2003:22 --hostname vllm-docker -e USERNAME=sorenle -v /tmp/authorized_keys:/tmp/authorized_keys:ro --name vllm-container --gpus=all nvcr.io/nvidia/vllm:26.01-py3 /bin/bash -c "sleep infinity"

vllm bench latency --batch-size=32 --tensor-parallel-size=1 --input-len=32 --output-len=32 --model="openai/gpt-oss-20b" --num-iters=60 --output-json=results_latency.json

power doesnt exist in new cli, manually do it with utils every second?
run throughput bench

docker exec -it vllm-container /bin/bash

watch -n 1 "nvidia-smi -q -d POWER" (see if theres a way to run this in the background that nvidia-smi provides)
grep the appropriate stuff
have a bash script that runs this ^
do it in two separate windows
gather separate power over different runs

https://medium.com/@anveshkumarchavidi/installing-vllm-on-nvidia-dgx-spark-from-source-4dde137ff3ef
https://docs.vllm.ai/en/latest/cli/bench/latency/?h=bench#arguments
https://build.nvidia.com/spark/vllm/instructions
https://docs.vllm.ai/en/latest/cli/bench/throughput/
