#!/usr/bin/env bash
#
# setup_and_run.sh
# ----------------
# One-shot setup + launcher for the M4 LLM study stack on macOS (Apple Silicon):
#
#   1. Checks required tools.
#   2. Clones and builds llama.cpp with the Metal (GPU) backend.
#   3. Fetches the chatbot files (index.html, server_m4.py) if they aren't
#      already next to this script.
#   4. Starts llama-server in the background (Metal-accelerated, tuned flags).
#   5. Waits until the model is loaded, then runs server_m4.py (the proxy +
#      powermetrics monitor, which needs sudo).
#
# Run it from the directory where you want llama.cpp/, models/, index.html and
# server_m4.py to live:
#
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh
#
# Ctrl-C stops everything (llama-server is killed on exit).

set -euo pipefail
# ----------------------------- configuration -----------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
CHATBOT_REPO="https://github.com/publiusys/pollms.git"
CHATBOT_SUBDIR="chatbot_m4"

MODEL_DIR="$PROJECT_DIR/models"
MODEL_FILE="Qwen3.5-4B-UD-IQ2_XXS.gguf"   # <-- change to match the file you downloaded
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

LLAMA_HOST="127.0.0.1"
LLAMA_PORT="8080"
PROXY_PORT="8000"                          # server_m4.py listens here

# Inference tuning (see README.md for rationale)
N_GPU_LAYERS="99"                          # offload all layers to the Metal GPU
CTX_SIZE="8192"                            # context window
THREADS="4"                                # M4 performance-core count
# --------------------------------------------------------------------------

log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*" >&2; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

# 1. Sanity checks ----------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  err "This script targets macOS (Apple Silicon). Detected: $(uname -s)."
  exit 1
fi

missing=()
for tool in git cmake python3 curl; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if (( ${#missing[@]} )); then
  err "Missing dependencies: ${missing[*]}"
  err "Install them first — see README.md (e.g. xcode-select --install && brew install cmake git)."
  exit 1
fi

cd "$PROJECT_DIR"

# 2. Build llama.cpp with Metal --------------------------------------------
LLAMA_BIN="$PROJECT_DIR/llama.cpp/build/bin/llama-server"
if [[ ! -x "$LLAMA_BIN" ]]; then
  if [[ ! -d "$PROJECT_DIR/llama.cpp" ]]; then
    log "Cloning llama.cpp ..."
    git clone --depth 1 "$LLAMA_REPO" "$PROJECT_DIR/llama.cpp"
  fi
  log "Building llama-server with the Metal backend (this can take a few minutes) ..."
  cmake -S "$PROJECT_DIR/llama.cpp" -B "$PROJECT_DIR/llama.cpp/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DLLAMA_CURL=OFF
  cmake --build "$PROJECT_DIR/llama.cpp/build" --config Release -j --target llama-server
  log "Build complete: $LLAMA_BIN"
else
  log "llama-server already built — skipping. (Delete llama.cpp/build to force a rebuild.)"
fi

# 3. Fetch chatbot files if missing ----------------------------------------
if [[ ! -f "$PROJECT_DIR/index.html" || ! -f "$PROJECT_DIR/server_m4.py" ]]; then
  log "Fetching chatbot files from $CHATBOT_REPO ..."
  tmp="$(mktemp -d)"
  git clone --depth 1 --filter=blob:none --sparse "$CHATBOT_REPO" "$tmp/pollms"
  ( cd "$tmp/pollms" && git sparse-checkout set "$CHATBOT_SUBDIR" )
  cp -n "$tmp/pollms/$CHATBOT_SUBDIR/index.html"   "$PROJECT_DIR/" 2>/dev/null || true
  cp -n "$tmp/pollms/$CHATBOT_SUBDIR/server_m4.py" "$PROJECT_DIR/" 2>/dev/null || true
  rm -rf "$tmp"
  log "Chatbot files ready."
else
  log "Chatbot files already present — skipping clone."
fi

# 4. Check the model --------------------------------------------------------
if [[ ! -f "$MODEL_PATH" ]]; then
  err "Model not found: $MODEL_PATH"
  err "Download a GGUF model into ./models/ (see README.md), or edit MODEL_FILE at the"
  err "top of this script to match your filename."
  exit 1
fi

# 5. Start llama-server in the background -----------------------------------
log "Starting llama-server on http://$LLAMA_HOST:$LLAMA_PORT ..."
# Build the flag list as an array so individual options can be commented
# out cleanly. (You cannot put '#' comments inside a '\'-continued command —
# doing so detaches the redirection/'&' and llama-server runs in the
# foreground, hanging the script before step 7.)
LLAMA_FLAGS=(
  -m "$MODEL_PATH"
  --host "$LLAMA_HOST" --port "$LLAMA_PORT" --no-ui
  -ngl "$N_GPU_LAYERS"
  -fa on
  -c "$CTX_SIZE"
  -t "$THREADS"
  # --mlock              # uncomment to pin weights in RAM
  # -ctk q8_0 -ctv q8_0  # uncomment for an 8-bit KV cache
)
"$LLAMA_BIN" "${LLAMA_FLAGS[@]}" > "$PROJECT_DIR/llama-server.log" 2>&1 &
LLAMA_PID=$!

cleanup() {
  log "Shutting down llama-server (pid $LLAMA_PID) ..."
  kill "$LLAMA_PID" 2>/dev/null || true
  wait "$LLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 6. Wait until the server is healthy ---------------------------------------
log "Waiting for llama-server to load the model (logs: llama-server.log) ..."
ready=0
for _ in $(seq 1 60); do
  if curl -sf "http://$LLAMA_HOST:$LLAMA_PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    err "llama-server exited early. Last 20 log lines:"
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

# 7. Run the proxy / power monitor (needs sudo for powermetrics) ------------
log "Starting server_m4.py on http://localhost:$PROXY_PORT  (sudo needed for powermetrics)."
log "Open http://localhost:$PROXY_PORT in your browser. Press Ctrl-C to stop everything."
sudo python3 "$PROJECT_DIR/server_m4.py"
