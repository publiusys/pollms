#!/usr/bin/env bash
#
# setup_and_run.sh  (direct / no-SLURM)
# -------------------------------------
# For when you SSH straight onto a GPU node (no SLURM, no sudo). It:
#
#   1. Checks tools (nvidia-smi, git, cmake, python3, screen, curl).
#   2. Builds llama.cpp with CUDA in your home/project dir (userspace, no sudo)
#      — skipped if already built.
#   3. Verifies the chatbot files (index.html, server_rtx5070ti.py) are present.
#   4. Starts llama-server in its OWN detached `screen` session.
#   5. Waits for the model to load, then starts server_rtx5070ti.py (proxy +
#      nvidia-smi power monitor) in a SEPARATE detached `screen` session.
#   6. Optionally blocks (BLOCK=1, default) so closing your SSH session doesn't
#      matter — the screens keep running regardless.
#
# No root, no sudo, no scheduler. Power comes from nvidia-smi.
#
# Usage (from this folder, on the GPU node):
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh
#
# If cmake/CUDA come from environment modules, load them first, e.g.:
#   module load cuda           # then run this script
# or point at a prebuilt binary:
#   LLAMA_BIN=/path/to/llama-server ./setup_and_run.sh

set -euo pipefail

# ----------------------------- configuration -----------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
# Built binary (this script builds it here unless you override LLAMA_BIN).
LLAMA_BIN="${LLAMA_BIN:-$LLAMA_DIR/build/bin/llama-server}"

# CUDA architecture: 120 = Blackwell consumer (RTX 50-series / RTX PRO 6000
# Blackwell). Override if your card differs:  CUDA_ARCH=89 ./setup_and_run.sh
CUDA_ARCH="${CUDA_ARCH:-120}"

MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-4B-UD-Q4_K_M.gguf}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$MODEL_FILE}"

LLAMA_HOST="127.0.0.1"
LLAMA_PORT="8080"
PROXY_PORT="8000"

LLAMA_SCREEN="rtx-llama"
PROXY_SCREEN="rtx-proxy"

N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
CTX_SIZE="${CTX_SIZE:-8192}"
PARALLEL="${PARALLEL:-1}"

BLOCK="${BLOCK:-1}"
# --------------------------------------------------------------------------

log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*" >&2; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

# 1. Sanity checks ----------------------------------------------------------
need_build=1
[[ -x "$LLAMA_BIN" ]] && need_build=0

# If a userspace toolchain was set up via bootstrap_toolchain.sh, use it.
if [[ -f "$PROJECT_DIR/toolchain/activate.sh" ]]; then
  log "Using local userspace toolchain (./toolchain)."
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/toolchain/activate.sh"
fi

# Runtime tools always needed:
missing=()
for tool in python3 curl screen nvidia-smi; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if (( ${#missing[@]} )); then
  err "Missing tools: ${missing[*]}"
  err "nvidia-smi missing usually means you're not on a GPU node."
  exit 1
fi

# Build tools only needed if we have to compile llama.cpp:
if (( need_build )); then
  build_missing=()
  for tool in git cmake nvcc; do
    command -v "$tool" >/dev/null 2>&1 || build_missing+=("$tool")
  done
  if (( ${#build_missing[@]} )); then
    err "No prebuilt llama-server and missing build tools: ${build_missing[*]}"
    err ""
    err "You have no sudo, so install a userspace toolchain once, then re-run:"
    err "    ./bootstrap_toolchain.sh && ./setup_and_run.sh"
    err ""
    err "Alternatives:"
    err "  - 'module load cuda cmake' if your site provides modules, then re-run."
    err "  - point at a prebuilt CUDA binary: LLAMA_BIN=/path/to/llama-server ./setup_and_run.sh"
    exit 1
  fi
fi

cd "$PROJECT_DIR"

# 2. Build llama.cpp with CUDA (userspace) ----------------------------------
if (( need_build )); then
  if ! command -v nvcc >/dev/null 2>&1; then
    warn "nvcc not on PATH — if the CUDA build fails, 'module load cuda' (or add"
    warn "your CUDA toolkit's bin/ to PATH) and re-run."
  fi
  [[ -d "$LLAMA_DIR" ]] || { log "Cloning llama.cpp ..."; git clone --depth 1 "$LLAMA_REPO" "$LLAMA_DIR"; }
  log "Building llama-server with CUDA (arch sm_$CUDA_ARCH); this takes a few minutes ..."
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

# 3. Verify chatbot files are present --------------------------------------
for f in index.html server_rtx5070ti.py; do
  if [[ ! -f "$PROJECT_DIR/$f" ]]; then
    err "Missing $f — it should sit next to this script."
    exit 1
  fi
done

# 4. Check the model --------------------------------------------------------
if [[ ! -f "$MODEL_PATH" ]]; then
  err "Model not found: $MODEL_PATH"
  err "Download a GGUF into \$MODEL_DIR (see README.md), or set MODEL_FILE / MODEL_DIR."
  exit 1
fi

log "GPU(s) visible:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /' || true

# 5. Start llama-server in its own detached screen session ------------------
LLAMA_FLAGS=(
  -m "$MODEL_PATH"
  --host "$LLAMA_HOST" --port "$LLAMA_PORT"
  -c "$CTX_SIZE"
  -ngl "$N_GPU_LAYERS"
  -fa on
  --parallel "$PARALLEL"
  --reasoning-format auto
  # -b 2048 -ub 2048   # uncomment to speed up long-prompt prefill
)

if screen -ls 2>/dev/null | grep -q "[.]$LLAMA_SCREEN[[:space:]]"; then
  log "screen session '$LLAMA_SCREEN' already running — leaving it as is."
else
  log "Starting llama-server in screen '$LLAMA_SCREEN' (http://$LLAMA_HOST:$LLAMA_PORT) ..."
  screen -L -Logfile "$PROJECT_DIR/llama-server.log" \
         -dmS "$LLAMA_SCREEN" "$LLAMA_BIN" "${LLAMA_FLAGS[@]}"
fi

# 6. Wait until the server is healthy ---------------------------------------
log "Waiting for llama-server to load the model (logs: llama-server.log) ..."
ready=0
for _ in $(seq 1 120); do
  if curl -sf "http://$LLAMA_HOST:$LLAMA_PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! screen -ls 2>/dev/null | grep -q "[.]$LLAMA_SCREEN[[:space:]]"; then
    err "llama-server screen ended early. Last 20 log lines:"
    tail -n 20 "$PROJECT_DIR/llama-server.log" >&2 || true
    exit 1
  fi
  sleep 2
done
if (( ! ready )); then
  err "Timed out waiting for llama-server. Check llama-server.log."
  exit 1
fi
log "llama-server is ready."

# 7. Start the proxy / power monitor in its own detached screen session -----
if screen -ls 2>/dev/null | grep -q "[.]$PROXY_SCREEN[[:space:]]"; then
  log "screen session '$PROXY_SCREEN' already running — leaving it as is."
else
  log "Starting server_rtx5070ti.py in screen '$PROXY_SCREEN' (port $PROXY_PORT) ..."
  screen -L -Logfile "$PROJECT_DIR/proxy.log" \
         -dmS "$PROXY_SCREEN" python3 "$PROJECT_DIR/server_rtx5070ti.py"
fi

NODE="$(hostname)"
cat <<EOF

$(printf '\033[1;32m[ready]\033[0m') Both services are up in separate screen sessions on node: $NODE
  - llama-server : session '$LLAMA_SCREEN'  (log: llama-server.log)
  - proxy/UI     : session '$PROXY_SCREEN'  (log: proxy.log)

The UI is on port $PROXY_PORT of this node. From your laptop, open an SSH tunnel:
  ssh -N -L $PROXY_PORT:localhost:$PROXY_PORT <user>@$NODE
(or hop via your login host if the node isn't directly reachable), then browse to
  http://localhost:$PROXY_PORT

Manage sessions:
  screen -ls
  screen -r $PROXY_SCREEN      (Ctrl-A then D to detach)
  screen -r $LLAMA_SCREEN
  screen -S $PROXY_SCREEN -X quit     # stop only the proxy
  screen -S $LLAMA_SCREEN -X quit     # stop llama-server
EOF

# 8. Optionally block; the screens survive regardless of this launcher ------
if [[ "$BLOCK" == "1" ]]; then
  log "Holding in the foreground (BLOCK=1). Ctrl-C here does NOT stop the screens."
  log "Set BLOCK=0 to return to your shell immediately."
  while screen -ls 2>/dev/null | grep -q "[.]$LLAMA_SCREEN[[:space:]]"; do
    sleep 10
  done
  log "llama-server screen has ended; exiting."
fi
