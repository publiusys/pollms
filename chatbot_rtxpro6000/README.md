# chatbot_rtxpro6000 — LLM study stack on an RTX PRO 6000 Blackwell node

The node has **5× NVIDIA RTX PRO 6000 Blackwell** GPUs (~96 GB VRAM each) and
**2× AMD EPYC 9355** CPUs. We run the model on **one** GPU and report a power
number that fairly represents **one GPU + one CPU**, so it's comparable to the
single-GPU nodes in this study.

Same shape as `chatbot_rtx5070ti`: SSH onto the node, build llama.cpp in
userspace (no sudo), then run `llama-server` + `server_rtxpro6000.py` in
separate detached `screen` sessions.

## Quick start

```bash
cd chatbot_rtxpro6000
chmod +x bootstrap_toolchain.sh setup_and_run.sh

# 1. toolchain (skip if `cmake` + `nvcc` already available via modules)
./bootstrap_toolchain.sh

# 2. put a GGUF model in ./models/ (96 GB/GPU — lots of headroom)

# 3. build + launch on GPU 0 (or pick another)
./setup_and_run.sh
GPU_INDEX=3 ./setup_and_run.sh        # run on GPU 3 instead
```

`CUDA_VISIBLE_DEVICES` pins `llama-server` to the one chosen GPU, and the proxy
is told the same `GPU_INDEX` so its power reading matches.

## Power model — how "1 GPU + 1 CPU" is computed

The proxy samples three sources each second and combines them. The result is at
`/power` as `current_w`, with every raw term exposed for verification.

| Term | Source | Notes |
|---|---|---|
| `gpu_w` | `nvidia-smi -i GPU_INDEX` | **Exact** draw of the one GPU in use. The 4 idle GPUs are excluded. |
| `gpu_all_w` | `nvidia-smi` (all GPUs) | reference only |
| `node_w` | `ipmitool dcmi power reading` | whole-node watts (all 5 GPUs + 2 CPUs + RAM + fans + PSU losses) |
| `cpu_w` | depends on `POWER_MODE` | one CPU socket (see below) |

**`current_w = gpu_w + cpu_w`** — that's the headline figure.

`POWER_MODE` (default `auto`) picks how `cpu_w` is derived:

- **`sensors`** — one CPU socket's package power from `sensors` (the AMD `PPT`
  line). This is *component* power and is the most directly comparable to the
  rtx5070ti node, which also reports GPU(nvidia-smi) + CPU(PPT).
- **`dcmi_split`** — derive one CPU's share from the whole-node BMC reading:

  ```
  cpu_w = (node_w − gpu_all_w) / N_CPU_SOCKETS        # default N=2
  ```

  i.e. take the node total, remove **all** GPU power, and split the remainder
  (2 CPUs + RAM + fans + PSU losses + board) across the two sockets. This
  attributes one CPU *plus its share of shared infrastructure*, so it runs
  higher than the pure component number.
- **`auto`** — use `sensors` if a CPU package reading is found, else fall back
  to `dcmi_split`.

### Which mode for a fair cross-node comparison?

The single-GPU nodes report **component** power (GPU package + CPU package), so
for an apples-to-apples comparison use **`sensors`** here too — then every node
is "GPU draw + CPU package draw", and shared infrastructure (PSU/fans) is
excluded everywhere. Use `dcmi_split` if you instead want a whole-node-aware
number that amortizes infrastructure across the sockets. Either way the raw
terms are in `/power`, so you can recompute whichever definition you prefer.

```bash
POWER_MODE=sensors    ./setup_and_run.sh     # component (recommended for comparison)
POWER_MODE=dcmi_split ./setup_and_run.sh     # node total minus other GPUs, per socket
```

> `ipmitool dcmi` usually needs privilege. If it's root-only, set
> `IPMITOOL_CMD="sudo ipmitool"` (and make sure that sudo is non-interactive),
> otherwise `node_w` stays 0 and `dcmi_split` can't be computed.

## Dependencies

```
nvidia-smi, git, cmake, nvcc, python3, screen, curl, gcc, g++   (build)
sensors        (optional — CPU package power for POWER_MODE=sensors)
ipmitool       (optional — whole-node power / POWER_MODE=dcmi_split)
```

## Tunables (env vars)

| Var | Default | Meaning |
|---|---|---|
| `GPU_INDEX` | `0` | which of the 5 GPUs to use (and read power from) |
| `POWER_MODE` | `auto` | `auto` / `sensors` / `dcmi_split` |
| `N_CPU_SOCKETS` | `2` | divisor for the DCMI split |
| `IPMITOOL_CMD` | `ipmitool` | e.g. `"sudo ipmitool"` |
| `MODEL_DIR` / `MODEL_FILE` | `./models` / `Qwen3.5-27B-Q5_K_M.gguf` | model location |
| `LLAMA_BIN` | built copy | use a prebuilt binary, skip building |
| `CUDA_ARCH` | `120` | Blackwell |
| `N_GPU_LAYERS` / `CTX_SIZE` | `99` / `8192` | inference tuning |
| `BLOCK` | `1` | `0` returns your shell immediately |

## Managing sessions

```bash
screen -ls
screen -r pro-proxy             # attach to proxy (Ctrl-A then D to detach)
screen -r pro-llama             # attach to llama-server
screen -S pro-proxy -X quit     # stop ONLY the proxy
screen -S pro-llama -X quit     # stop llama-server
```

## Layout

```
chatbot_rtxpro6000/
├── setup_and_run.sh
├── bootstrap_toolchain.sh
├── README.md
├── index.html
├── server_rtxpro6000.py    # proxy + 1-GPU/1-CPU power monitor
├── llama-server.log
├── proxy.log
├── toolchain/              # userspace cmake/CUDA (from bootstrap)
├── llama.cpp/              # cloned + built on first run
└── models/
    └── <your-model>.gguf
```
