# chatbot_m4 — local LLM study stack (Apple Silicon)

Runs a LLM locally with [llama.cpp](https://github.com/ggml-org/llama.cpp)
(Metal/GPU accelerated) behind a small Python proxy (`server_m4.py`) that serves
the web UI (`index.html`), forwards requests to the model, and
samples package power with `powermetrics`.

`setup_and_run.sh` automates the whole thing: it builds llama.cpp, fetches the
chatbot files, launches `llama-server` in the background, and then starts
`server_m4.py`.

## Requirements

This stack is built for **macOS on Apple Silicon** (tested target: M4, 16 GB).

Install the following before running the script:

| Dependency | Why | Install |
|---|---|---|
| Xcode Command Line Tools | Provides `clang` and the Metal toolchain needed to build llama.cpp | `xcode-select --install` |
| Homebrew | Package manager for the tools below | see https://brew.sh |
| `cmake` | Builds llama.cpp | `brew install cmake` |
| `git` | Clones llama.cpp and the chatbot repo | `brew install git` (or comes with Xcode CLT) |
| `python3` | Runs `server_m4.py` (standard library only — no pip packages needed) | preinstalled on macOS, or `brew install python` |
| `curl` | Health-check during startup | preinstalled on macOS |
| `sudo` access | `powermetrics` requires root to read power counters | your account password |

## Model

You need a **GGUF** model file placed in a `models/` folder next to the script.
By default the script looks for:

```
models/Qwen3.5-4B-UD-IQ2_XXS.gguf
```

Download a GGUF build of your chosen model from its Hugging Face page and put it
in `./models/`. For example `https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main`
```

If your filename differs from the default, either rename it to match or edit
`MODEL_FILE` near the top of `setup_and_run.sh`.

## Usage

```bash
./setup_and_run.sh
```

Then open **http://localhost:8000** in your browser.

The first run builds llama.cpp (a few minutes); later runs skip the build.
Press **Ctrl-C** to stop — the script shuts down `llama-server` automatically.

## What the script launches

`llama-server` is started with performance-oriented flags for the M4:

| Flag | Purpose |
|---|---|
| `-ngl 99` | Offload all layers to the Metal GPU (the biggest speed lever) |
| `-fa on` | Flash attention — faster, smaller KV cache |
| `-c 8192` | Context window (raise/lower to taste; memory scales with it) |
| `-t 4` | Threads = M4 performance-core count |
| `--mlock` | Pin weights in RAM so they aren't swapped out - NOT ENABLED YET|
| `-ctk q8_0 -ctv q8_0` | 8-bit KV cache — roughly halves context memory - NOT ENABLED YET|

`server_m4.py` then runs under `sudo` on port **8000**, serving `index.html`,
proxying `/v1/*` to `llama-server` on port 8080, and exposing `/power`.

## Layout after setup

```
chatbot_m4/
├── setup_and_run.sh
├── README.md
├── index.html            # fetched from the repo if missing
├── server_m4.py          # fetched from the repo if missing
├── llama-server.log      # llama-server output
├── llama.cpp/            # cloned + built here
└── models/
    └── <your-model>.gguf
```