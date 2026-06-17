# chatbot_dgxspark — local LLM study stack (NVIDIA DGX Spark)

Runs a quantized LLM locally with llama.cpp (CUDA, GB10 GPU) behind a small
Python proxy (`server_dgxspark.py`) that serves the web UI (`index.html`),
forwards requests to the model, and samples board power via `sensors`.

`setup_and_run.sh` fetches the chatbot files, then launches `llama-server` and
`server_dgxspark.py` in **separate detached `screen` sessions** so the two are
independent — stopping the proxy leaves `llama-server` running.

## Hardware

NVIDIA DGX Spark — GB10 Grace-Blackwell superchip: 20-core ARM CPU
(10× Cortex-X925 + 10× Cortex-A725), 128 GB LPDDR5X **unified** memory at
**273 GB/s**, GPU with 48 SMs / 6144 CUDA cores.

## Requirements

| Dependency | Why | Install |
|---|---|---|
| `llama-server` (CUDA build) | The inference server | Follow NVIDIA's guide: https://build.nvidia.com/spark/llama-cpp/instructions |
| `git` | Fetches the chatbot files | `sudo apt install git` |
| `python3` | Runs `server_dgxspark.py` (standard library only) | preinstalled on DGX OS |
| `curl` | Health-check during startup | `sudo apt install curl` |
| `sensors` | The proxy reads board power from `sensors` output | from **lm-sensors** (`sudo apt install lm-sensors`) |
| `screen` | Runs llama-server and the proxy in independent sessions | `sudo apt install screen` |

### Power sensor module

`server_dgxspark.py` parses the `sys_total` line from `sensors`. Exposing that
reading requires the DGX Spark hwmon power-export module:

  https://github.com/handong32/spark_hwmon

Install/load that module per its README so `sensors` reports `sys_total`;
otherwise the `/power` endpoint will report 0 W.

If `llama-server` isn't on your `PATH`, point the script at it:

```bash
LLAMA_BIN=/path/to/llama-server ./setup_and_run.sh
```

## Model

Place a **GGUF** model in a `models/` folder next to the script. By default the
script looks for:

```
models/Qwen3.5-27B-Q3_K_S.gguf
```

If your filename differs, rename it or edit `MODEL_FILE` near the top of
`setup_and_run.sh`.

## Usage

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

Then open **http://localhost:8000** in your browser. The script starts both
processes in detached `screen` sessions and exits, leaving them running.

Manage the sessions:

```bash
screen -ls                          # list running sessions
screen -r dgx-proxy                 # attach to the proxy (Ctrl-A then D to detach)
screen -r llama-server              # attach to llama-server
screen -S dgx-proxy -X quit         # stop ONLY the proxy — llama-server keeps running
screen -S llama-server -X quit      # stop llama-server
```

## llama-server flags & parameter review
```
llama-server -m Qwen3.5-27B-Q3_K_S.gguf --host 127.0.0.1 --port 8080 \
  -c 16384 -ngl 99 -fa on --parallel 1 --reasoning-format auto
```

| Flag | Verdict |
|---|---|
| `-ngl 99` | ✅ Offload all layers to the GPU|
| `-fa on` | ✅ Flash attention on |
| `-c 16384` | ✅ Context window size|
| `--parallel 1` | ✅ Single stream requests  |
| `--reasoning-format auto` | ✅ for parsing Qwen3 style thinking models |

## Layout after setup

```
chatbot_dgxspark/
├── setup_and_run.sh
├── README.md
├── index.html             # fetched from the repo if missing
├── server_dgxspark.py     # fetched from the repo if missing
├── llama-server.log       # llama-server output
└── models/
    └── <your-model>.gguf
```