# Version 4

## Install docker
```
TODO
```

## Launching docker container (WIP, not complete)
```
$ sudo docker run -d -p 2002:22 --hostname mochiai-docker -e USERNAME=mochiai -v /tmp/authorized_keys:/tmp/authorized_keys:ro --name mochiai-container --gpus=all nvidia-ssh-image
```
/tmp/authorized_keys - contains ssh keys to access docker container

-p flag to allow outside connections 

## Access container
```
$ docker exec -it mochiai-container bash
```

## Run INSIDE container
```
$ sudo apt install iproute2 -y
$ sudo apt install -y git clang cmake libcurl4-openssl-dev libssl-dev
$ sudo apt install lm-sensors
$ sudo apt install emacs
$ sudo apt install curl
$ sudo apt install screen

$ git clone https://github.com/publiusys/pollms.git
$ cd pollms/chatbot_v4
```

### Clone and build llama.cpp
Follow instructions here: https://build.nvidia.com/spark/llama-cpp/instructions

### Gather power numbers
We're using this: https://github.com/antheas/spark_hwmon, on DGX Spark, disable secureboot

### Download Qwen model
Search on hugging face for a Qwen 3.5 model in GGUF format, will need to find different parameter sizes, i.e. https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/main


### Run llama.cpp server
The following command uses `screen` to run the llama-server in the background. `llama-server` exposes an API at 127.0.0.1:8080 for anyone to make an API call to evaluate a prompt.

For example, `curl -sN http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3.5-27B-Q3_K_S.gguf","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"hi"}],"temperature":0.7,"max_tokens":1024,"stream":true,"stream_options":{"include_usage":true}}'`, to manually send a prompt.

```
$ llama-server -m Qwen3.5-27B-Q3_K_S.gguf --host 127.0.0.1 --port 8080 -c 16384 -ngl 99 -fa on --parallel 1 --reasoning-format auto
OR 
$ screen -dmS llama bash -c "llama-server -m Qwen3.5-27B-Q3_K_S.gguf --host 127.0.0.1 --port 8080 -c 16384 -ngl 99 -fa on --parallel 1 --reasoning-format auto 2>&1 | tee llama.log
```

### Run server_dgx.py
The following command runs server_dgxspark.py in the background. This server serves `index.html` and also collects power numbers from `spark_hwmon`. This code also acts as a proxy by forwarding all user prompts from `index.html` to `llama-server` API and then displays the LLM generated output in `index.html`
```
$ screen -dmS webui bash -c "python3 server_dgxspark.py 2>&1 | tee webui.log"
```

For example if you run it manually, the following shows that it is serving all requests at `0.0.0.0:8000`, `0.0.0.0` means it will listen and respond on all interfaces, effectively exposing the Docker container to anyone and `8000` is the port it is listening on.
```
hdong1@dgx1-docker:~/pollms/chatbot_v4$ python3 server_dgxspark.py
Serving on http://0.0.0.0:8000  (proxying llama at http://127.0.0.1:8080)
```

## Run OUTSIDE container
```
$ socat TCP-LISTEN:8000,fork,reuseaddr TCP:172.17.0.2:8000
```
`socat` sets up a port forwarder or a network proxy. It listens for incoming traffic on one port and shoves it over to another destination.
Here is the step-by-step breakdown of exactly what each piece of that command is doing:

`TCP-LISTEN:8000`
This tells socat to open up port 8000 on your local machine and listen for any incoming TCP connections.

`,fork`
socat spawns (forks) a new child process to handle it, allowing the main process to keep listening for more connections.

`,reuseaddr`
Prevents annoying "Address already in use" error.

`TCP:172.17.0.2:8000`
Whenever someone connects to your local port 8000, socat will forward all that traffic over TCP to the IP address 172.17.0.2 on port 8000. `172.17.0.2` is the default IP given to Docker containers.

