#!/usr/bin/env bash
#
# setup_and_run.sh  (RTX PRO 6000 Blackwell node — 5 GPUs, 2x EPYC 9355)
# ----------------------------------------------------------------------
# SSH straight onto the node (no SLURM, no sudo needed for the model). It:
#
#   1. Checks tools (nvidia-smi, git, cmake, python3, screen, curl).
#   2. Builds llama.cpp with CUDA in your home/project dir (userspace) — skipped
#      if already built. Run ./bootstrap_toolchain.sh first if cmake/nvcc are
#      missing, or point LLAMA_BIN at a prebuilt binary.
#   3. Pins the model to ONE GPU via CUDA_VISIBLE_DEVICES=$GPU_INDEX.
#   4. Starts llama-server in its own detached `screen` session.
#   5. Waits for the model to load, then starts server_rtxpro6000.py (proxy +
#      power monitor) in a SEPARATE detached `screen` session, telling it which
#      GPU to read and how to compute the "1 GPU + 1 CPU" power number.
#   6. Optionally blocks (BLOCK=1, default) so the screens outlive your SSH.
#
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh                 # uses GPU 0
#   GPU_INDEX=3 ./setup_and_run.sh     # run on GPU 3 instead

set -euo pipefail

# ----------------------------- configuration -----------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
LLAMA_BIN="${LLAMA_BIN:-$LLAMA_DIR/build/bin/llama-server}"

CUDA_ARCH="${CUDA_ARCH:-120}"               # 120 = Blackwell (RTX PRO 6000)

MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-27B-Q5_K_M.gguf}"   # ~96 GB VRAM/GPU: plenty of room
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$MODEL_FILE}"

LLAMA_HOST="127.0.0.1"
LLAMA_PORT="8080"
PROXY_PORT="8000"

LLAMA_SCREEN="pro-llama"
PROXY_SCREEN="pro-proxy"

# --- which GPU to run on (one of the 5) and how to attribute power ---
GPU_INDEX="${GPU_INDEX:-0}"                  # physical GPU index 0..4
N_CPU_SOCKETS="${N_CPU_SOCKETS:-2}"          # for the DCMI power split
POWER_MODE="${POWER_MODE:-dcmi_split}"             # auto | sensors | dcmi_split
IPMITOOL_CMD="${IPMITOOL_CMD:-ipmitool}"     # e.g. "sudo ipmitool" if root-only

N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
CTX_SIZE="${CTX_SIZE:-262144}"
PARALLEL="${PARALLEL:-1}"
BLOCK="${BLOCK:-1}"
# --------------------------------------------------------------------------

log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*" >&2; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

# 1. Sanity checks ----------------------------------------------------------
need_build=1
[[ -x "$LLAMA_BIN" ]] && need_build=0

if [[ -f "$PROJECT_DIR/toolchain/activate.sh" ]]; then
  log "Using local userspace toolchain (./toolchain)."
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/toolchain/activate.sh"
fi

missing=()
for tool in python3 curl screen nvidia-smi; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if (( ${#missing[@]} )); then
  err "Missing tools: ${missing[*]}"
  exit 1
fi
command -v sensors  >/dev/null 2>&1 || warn "sensors not found — CPU package power unavailable (will use DCMI split if possible)."
command -v ipmitool >/dev/null 2>&1 || warn "ipmitool not found — whole-node/DCMI power unavailable."

if (( need_build )); then
  build_missing=()
  for tool in git cmake nvcc; do
    command -v "$tool" >/dev/null 2>&1 || build_missing+=("$tool")
  done
  if (( ${#build_missing[@]} )); then
    err "No prebuilt llama-server and missing build tools: ${build_missing[*]}"
    err "Run ./bootstrap_toolchain.sh once, 'module load cuda cmake', or set LLAMA_BIN."
    exit 1
  fi
fi

cd "$PROJECT_DIR"

# 2. Build llama.cpp with CUDA (userspace) ----------------------------------
if (( need_build )); then
  [[ -d "$LLAMA_DIR" ]] || { log "Cloning llama.cpp ..."; git clone --depth 1 "$LLAMA_REPO" "$LLAMA_DIR"; }
  log "Building llama-server with CUDA (arch sm_$CUDA_ARCH); takes a few minutes ..."
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DLLAMA_CURL=OFF
  cmake --build "$LLAMA_DIR/build" --config Release -j --target llama-server
  log "Build complete: $LLAMA_BIN"
else
  log "Using existing llama-server: $LLAMA_BIN"
fi

# 3. Verify chatbot files ---------------------------------------------------
for f in index.html server_rtxpro6000.py; do
  [[ -f "$PROJECT_DIR/$f" ]] || { err "Missing $f — it should sit next to this script."; exit 1; }
done

# 4. Check the model --------------------------------------------------------
if [[ ! -f "$MODEL_PATH" ]]; then
  err "Model not found: $MODEL_PATH"
  err "Download a GGUF into \$MODEL_DIR (see README.md), or set MODEL_FILE / MODEL_DIR."
  exit 1
fi

log "Targeting GPU $GPU_INDEX of:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /' || true

# 5. Start llama-server on ONE GPU, in its own screen session ---------------
LLAMA_FLAGS=(
  -m "$MODEL_PATH"
  --host "$LLAMA_HOST" --port "$LLAMA_PORT"
  -ngl "$N_GPU_LAYERS"
  -c "$CTX_SIZE"
  -fa on
  --parallel "$PARALLEL"
  --reasoning-format auto
)

if screen -ls 2>/dev/null | grep -q "[.]$LLAMA_SCREEN[[:space:]]"; then
  log "screen session '$LLAMA_SCREEN' already running — leaving it as is."
else
  log "Starting llama-server on GPU $GPU_INDEX in screen '$LLAMA_SCREEN' ..."
  # CUDA_VISIBLE_DEVICES restricts llama-server to the single chosen GPU.
  screen -L -Logfile "$PROJECT_DIR/llama-server.log" -dmS "$LLAMA_SCREEN" \
         env CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$LLAMA_BIN" "${LLAMA_FLAGS[@]}"
fi

# 6. Wait until healthy -----------------------------------------------------
log "Waiting for llama-server to load the model (logs: llama-server.log) ..."
ready=0
for _ in $(seq 1 120); do
  if curl -sf "http://$LLAMA_HOST:$LLAMA_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! screen -ls 2>/dev/null | grep -q "[.]$LLAMA_SCREEN[[:space:]]"; then
    err "llama-server screen ended early. Last 20 log lines:"; tail -n 20 "$PROJECT_DIR/llama-server.log" >&2 || true; exit 1
  fi
  sleep 2
done
(( ready )) || { err "Timed out waiting for llama-server. Check llama-server.log."; exit 1; }
log "llama-server is ready."

# 7. Start the proxy / power monitor in its own screen session --------------
if screen -ls 2>/dev/null | grep -q "[.]$PROXY_SCREEN[[:space:]]"; then
  log "screen session '$PROXY_SCREEN' already running — leaving it as is."
else
  log "Starting server_rtxpro6000.py in screen '$PROXY_SCREEN' (port $PROXY_PORT, power mode '$POWER_MODE') ..."
  screen -L -Logfile "$PROJECT_DIR/proxy.log" -dmS "$PROXY_SCREEN" \
         env GPU_INDEX="$GPU_INDEX" N_CPU_SOCKETS="$N_CPU_SOCKETS" \
             POWER_MODE="$POWER_MODE" IPMITOOL_CMD="$IPMITOOL_CMD" PORT="$PROXY_PORT" \
             python3 "$PROJECT_DIR/server_rtxpro6000.py"
fi

NODE="$(hostname)"
cat <<EOF

$(printf '\033[1;32m[ready]\033[0m') Up on node $NODE, GPU $GPU_INDEX (power mode: $POWER_MODE)
  - llama-server : session '$LLAMA_SCREEN'  (log: llama-server.log)
  - proxy/UI     : session '$PROXY_SCREEN'  (log: proxy.log)

The /power endpoint reports a "1 GPU + 1 CPU" figure (current_w) plus the raw
terms (gpu_w, cpu_w, node_w, gpu_all_w, mode) so you can verify the split.

UI on port $PROXY_PORT of this node. From your laptop:
  ssh -N -L $PROXY_PORT:localhost:$PROXY_PORT <user>@$NODE
then open http://localhost:$PROXY_PORT

Manage: screen -ls ; screen -r $PROXY_SCREEN (Ctrl-A D to detach) ;
        screen -S $PROXY_SCREEN -X quit   # stop only the proxy
        screen -S $LLAMA_SCREEN -X quit   # stop llama-server
EOF

# 8. Optionally block so the screens outlive this launcher ------------------
if [[ "$BLOCK" == "1" ]]; then
  log "Holding in foreground (BLOCK=1). Ctrl-C here does NOT stop the screens."
  while screen -ls 2>/dev/null | grep -q "[.]$LLAMA_SCREEN[[:space:]]"; do sleep 10; done
  log "llama-server screen ended; exiting."
fi
