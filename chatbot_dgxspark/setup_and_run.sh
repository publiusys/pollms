#!/usr/bin/env bash
#
# setup_and_run.sh
# ----------------
# Launcher for the DGX Spark LLM study stack (NVIDIA GB10, DGX OS / Linux):
#
#   1. Checks required tools (incl. an already-installed llama-server).
#   2. Fetches the chatbot files (index.html, server_dgxspark.py) if they
#      aren't already next to this script.
#   3. Starts llama-server in its OWN detached `screen` session (CUDA, tuned
#      flags).
#   4. Waits until the model is loaded, then starts server_dgxspark.py (the
#      proxy + power monitor) in a SEPARATE detached `screen` session.
#
# Because each process lives in its own screen session, stopping or detaching
# from one does NOT affect the other — quitting the proxy leaves llama-server
# running (and vice versa). The script itself exits once both are up.
#
# llama-server is NOT built by this script. Install it first by following
# NVIDIA's instructions:  https://build.nvidia.com/spark/llama-cpp/instructions
#
# Run from the directory where you want models/, index.html and
# server_dgxspark.py to live:
#
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh
#
# Manage the sessions afterwards:
#   screen -ls                       # list running sessions
#   screen -r llama-server           # attach to llama-server (Ctrl-A D to detach)
#   screen -r dgx-proxy              # attach to the proxy
#   screen -S dgx-proxy -X quit      # stop ONLY the proxy (llama-server keeps running)
#   screen -S llama-server -X quit   # stop llama-server

set -euo pipefail

# ----------------------------- configuration -----------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHATBOT_REPO="https://github.com/publiusys/pollms.git"
CHATBOT_SUBDIR="chatbot_dgxspark"

MODEL_DIR="$PROJECT_DIR/models"
MODEL_FILE="Qwen3.5-27B-Q3_K_S.gguf"   # <-- change to match the file you downloaded
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

# Set this if llama-server is not on your PATH (e.g. a full path to the binary).
LLAMA_BIN="${LLAMA_BIN:-llama-server}"

LLAMA_HOST="127.0.0.1"
LLAMA_PORT="8080"
PROXY_PORT="8000"                       # server_dgxspark.py listens here

LLAMA_SCREEN="llama-server"             # screen session name for the model server
PROXY_SCREEN="dgx-proxy"                # screen session name for the proxy

# Inference tuning (see README.md for rationale)
N_GPU_LAYERS="99"                       # offload all layers to the GPU
CTX_SIZE="16384"                        # context window
PARALLEL="1"                            # single sequence slot (single-stream study)
# --------------------------------------------------------------------------

log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*" >&2; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

# 1. Sanity checks ----------------------------------------------------------
missing=()
for tool in git python3 curl sensors screen; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if (( ${#missing[@]} )); then
  err "Missing dependencies: ${missing[*]}"
  err "See README.md for how to install them (sensors comes from lm-sensors)."
  exit 1
fi

if ! command -v "$LLAMA_BIN" >/dev/null 2>&1 && [[ ! -x "$LLAMA_BIN" ]]; then
  err "llama-server not found (looked for: $LLAMA_BIN)."
  err "Install it per NVIDIA's guide: https://build.nvidia.com/spark/llama-cpp/instructions"
  err "Then re-run, or set LLAMA_BIN to the binary path: LLAMA_BIN=/path/to/llama-server ./setup_and_run.sh"
  exit 1
fi

cd "$PROJECT_DIR"

# 2. Fetch chatbot files if missing ----------------------------------------
if [[ ! -f "$PROJECT_DIR/index.html" || ! -f "$PROJECT_DIR/server_dgxspark.py" ]]; then
  log "Fetching chatbot files from $CHATBOT_REPO ..."
  tmp="$(mktemp -d)"
  git clone --depth 1 --filter=blob:none --sparse "$CHATBOT_REPO" "$tmp/pollms"
  ( cd "$tmp/pollms" && git sparse-checkout set "$CHATBOT_SUBDIR" )
  cp -n "$tmp/pollms/$CHATBOT_SUBDIR/index.html"          "$PROJECT_DIR/" 2>/dev/null || true
  cp -n "$tmp/pollms/$CHATBOT_SUBDIR/server_dgxspark.py"  "$PROJECT_DIR/" 2>/dev/null || true
  rm -rf "$tmp"
  log "Chatbot files ready."
else
  log "Chatbot files already present — skipping clone."
fi

# 3. Check the model --------------------------------------------------------
if [[ ! -f "$MODEL_PATH" ]]; then
  err "Model not found: $MODEL_PATH"
  err "Download a GGUF model into ./models/ (see README.md), or edit MODEL_FILE at the"
  err "top of this script to match your filename."
  exit 1
fi

# 4. Start llama-server in its own detached screen session ------------------
# Flags live in an array so individual options can be commented out cleanly.
LLAMA_FLAGS=(
  -m "$MODEL_PATH"
  --host "$LLAMA_HOST" --port "$LLAMA_PORT"
  -c "$CTX_SIZE"
  -ngl "$N_GPU_LAYERS"
  -fa on
  --parallel "$PARALLEL"
  --reasoning-format auto
  # -b 2048 -ub 2048   # uncomment to speed up long-prompt prefill (compute headroom)
)

if screen -ls 2>/dev/null | grep -q "[.]$LLAMA_SCREEN[[:space:]]"; then
  log "screen session '$LLAMA_SCREEN' already running — leaving it as is."
else
  log "Starting llama-server in screen session '$LLAMA_SCREEN' (http://$LLAMA_HOST:$LLAMA_PORT) ..."
  # -L -Logfile captures the session's output to a file; -dm starts it detached.
  screen -L -Logfile "$PROJECT_DIR/llama-server.log" \
         -dmS "$LLAMA_SCREEN" "$LLAMA_BIN" "${LLAMA_FLAGS[@]}"
fi

# 5. Wait until the server is healthy ---------------------------------------
log "Waiting for llama-server to load the model (logs: llama-server.log) ..."
ready=0
for _ in $(seq 1 90); do
  if curl -sf "http://$LLAMA_HOST:$LLAMA_PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! screen -ls 2>/dev/null | grep -q "[.]$LLAMA_SCREEN[[:space:]]"; then
    err "llama-server screen session ended early. Last 20 log lines:"
    tail -n 20 "$PROJECT_DIR/llama-server.log" >&2 || true
    exit 1
  fi
  sleep 2
done
if (( ! ready )); then
  err "Timed out waiting for llama-server. Check llama-server.log (session '$LLAMA_SCREEN' may still be loading)."
  exit 1
fi
log "llama-server is ready."

# 6. Start the proxy / power monitor in its own detached screen session -----
if screen -ls 2>/dev/null | grep -q "[.]$PROXY_SCREEN[[:space:]]"; then
  log "screen session '$PROXY_SCREEN' already running — leaving it as is."
else
  log "Starting server_dgxspark.py in screen session '$PROXY_SCREEN' (http://localhost:$PROXY_PORT) ..."
  screen -L -Logfile "$PROJECT_DIR/proxy.log" \
         -dmS "$PROXY_SCREEN" python3 "$PROJECT_DIR/server_dgxspark.py"
fi

# Both services are now running independently; this script can exit safely.
cat <<EOF

$(printf '\033[1;32m[ready]\033[0m') Both services are up in separate screen sessions:
  - llama-server : session '$LLAMA_SCREEN'  (log: llama-server.log)
  - proxy/UI     : session '$PROXY_SCREEN'  (log: proxy.log)

Open http://localhost:$PROXY_PORT in your browser.

Manage them:
  screen -ls                          list sessions
  screen -r $PROXY_SCREEN                attach to the proxy   (Ctrl-A then D to detach)
  screen -r $LLAMA_SCREEN             attach to llama-server
  screen -S $PROXY_SCREEN -X quit        stop ONLY the proxy (llama-server keeps running)
  screen -S $LLAMA_SCREEN -X quit     stop llama-server
EOF
