#!/usr/bin/env python3
"""
Simple Flask proxy for vLLM — serves the chat UI and forwards API calls.
No CORS needed since everything comes from the same origin.

Usage:
    pip install flask requests
    python3 server.py

Then open: http://150.209.23.147:8080
"""

import json
import requests
from flask import Flask, request, Response, send_from_directory
import os

app = Flask(__name__)

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Serve the chat UI ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

# ── Proxy: list models ─────────────────────────────────────────────────────────

@app.route("/v1/models", methods=["GET"])
def models():
    try:
        r = requests.get(f"{VLLM_BASE}/v1/models", timeout=5)
        return Response(r.content, status=r.status_code, content_type="application/json")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=502, content_type="application/json")

# ── Proxy: chat completions (streaming) ───────────────────────────────────────

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    body = request.get_json()
    stream = body.get("stream", False)

    try:
        r = requests.post(
            f"{VLLM_BASE}/v1/chat/completions",
            json=body,
            stream=stream,
            timeout=120
        )

        if stream:
            def generate():
                for chunk in r.iter_content(chunk_size=None):
                    yield chunk
            return Response(generate(), status=r.status_code, content_type="text/event-stream")
        else:
            return Response(r.content, status=r.status_code, content_type="application/json")

    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=502, content_type="application/json")

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"  Chat UI  →  http://0.0.0.0:{port}")
    print(f"  vLLM     →  {VLLM_BASE}")
    app.run(host="0.0.0.0", port=port, debug=False)
