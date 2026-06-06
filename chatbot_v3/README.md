# llama.cpp + server.py Setup Guide

## Updatd on 6/6/6
```
Added an efficiency score
```

A guide to install and run a local LLM study setup: llama-server (inference) + server.py (proxy + power telemetry) + index.html (study UI)

## 1. Prerequisites / Dependencies
Python 3.8+, Git, CMake, C/C++ compiler

## 2. Install llama.cpp
### macOS (Metal)
```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j
cd ..
```

### Ubuntu — NVIDIA GPU (CUDA)
```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
cd ..
```

## 3. Download a Model from HuggingFace
Q4_K_M (~4.7 GB) fits comfortably on 16 GB. 14B Q4_K_M (~9 GB) is tight; 27B (~12 GB+) does NOT fit on 16 GB and will abort or swap.
```
wget https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/86a5c8ef08829a14a13778a002fc2bccbe1cf104/Qwen3.5-27B-Q3_K_S.gguf?download=true
```

## 4. Launch llama-server in one terminal
### macOS (Metal)
mlock is only for shared memory systems (i.e. M4/DGX Spark) to pin memory allocated for model

-ngl 99	Layers offloaded to GPU	99 ≈ "all". Metal (macOS) / CUDA (Linux). ⚠️ If model is too big to fit, this aborts instead of auto-reducing — lower it or remove it.

-c 4096	Context length (tokens)	Bigger = larger KV cache = more memory. Omitting uses model's full trained context (can be 32K–262K). Set explicitly.

--no-webui	Disable built-in chat UI	API still works; optional

```
./llama.cpp/build/bin/llama-server \
  -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 \
  --no-webui \
  -ngl 99 \
  -c 4096 \
  --mlock 
```

### Ubuntu (NVIDIA)
```
./llama.cpp/build/bin/llama-server \
  -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 \
  --no-webui \
  -ngl 99 \
  -c 4096 
```

## 5. Run server.py (Proxy + Power Monitor) in another terminal
```
sudo python3 server.py
```
