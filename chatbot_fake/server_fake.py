#!/usr/bin/env python3
"""
Fake backend for quickly testing index.html — no llama-server needed.

- Serves index.html
- Implements the OpenAI-ish endpoints index.html expects:
    GET  /v1/models            -> a dummy model id
    POST /v1/chat/completions  -> ECHOES the user's last message back, streamed
                                  as SSE chunks (or one JSON blob if stream=false)
    GET  /power                -> dummy power JSON (so the UI's power widget works)
    POST /active               -> accepts the page's active/idle toggle
- Does NOT forward anything to a real model. It just plays your input back.

Run:
    python3 server_fake.py
    # then open http://localhost:9000

Optional env vars:
    PORT            listen port (default 9000)
    INDEX_FILE      which HTML file to serve at / (default index.html);
                    e.g. INDEX_FILE=index_v2.html python3 server_fake.py
    FAKE_DELAY      seconds between streamed tokens (default 0.03; 0 = instant)
    FAKE_PREFIX     text prepended to the echoed reply (default "")
"""
import json, os, re, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT        = int(os.environ.get("PORT", "9000"))
INDEX_FILE  = os.environ.get("INDEX_FILE", "index.html")
FAKE_DELAY  = float(os.environ.get("FAKE_DELAY", "0.03"))
FAKE_PREFIX = os.environ.get("FAKE_PREFIX", "")
MODEL_ID    = "fake-echo"

# shared state so /power can reflect the page's active flag (all dummy values)
state = {"current_w": 0.0, "gpu_w": 0.0, "cpu_w": 0.0, "idle_w": 0.0, "active": False}
lock = threading.Lock()


def last_user_message(messages):
    """Return the content of the most recent user-role message."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            # content may be a string or a list of parts (OpenAI vision style)
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c or ""
    return ""


def tokenize(text):
    """Split into pieces that preserve whitespace, so the echo streams verbatim."""
    return re.findall(r"\S+\s*|\s+", text) or [text]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------- GET ----------------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(os.path.join(os.path.dirname(__file__), INDEX_FILE), "rb") as f:
                    self._send(200, f.read(), "text/html")
            except FileNotFoundError:
                self._send(404, ('{"error":"%s not found"}' % INDEX_FILE).encode())
        elif self.path.rstrip("/") == "/v1/models":
            self._send(200, json.dumps({
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "fake"}],
            }))
        elif self.path == "/power":
            with lock:
                self._send(200, json.dumps({
                    "current_w": round(state["current_w"], 2),
                    "gpu_w": round(state["gpu_w"], 2),
                    "cpu_w": round(state["cpu_w"], 2),
                    "idle_w": round(state["idle_w"], 2),
                    "active": state["active"],
                }))
        else:
            self._send(404, b'{"error":"not found"}')

    # ---------------- POST ----------------
    def do_POST(self):
        if self.path == "/active":
            data = self._read_json()
            with lock:
                state["active"] = bool(data.get("active", False))
            self._send(200, b'{"ok":true}')
        elif self.path.rstrip("/") == "/v1/chat/completions":
            self._chat()
        else:
            self._send(404, b'{"error":"not found"}')

    def _read_json(self):
        ln = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(ln) if ln else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    def _chat(self):
        req = self._read_json()
        reply = FAKE_PREFIX + last_user_message(req.get("messages"))
        if not reply.strip():
            reply = "(fake server: no user message to echo)"
        model = req.get("model", MODEL_ID)
        stream = req.get("stream", False)
        want_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        completion_tokens = max(1, len(reply.split()))
        created = int(time.time())

        if not stream:
            self._send(200, json.dumps({
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0,
                          "completion_tokens": completion_tokens,
                          "total_tokens": completion_tokens},
            }))
            return

        # ---- streamed (SSE), the path index.html uses ----
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def emit(obj):
            self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
            self.wfile.flush()

        try:
            # opening role delta
            emit({"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                  "created": created, "model": model,
                  "choices": [{"index": 0, "delta": {"role": "assistant"},
                               "finish_reason": None}]})
            # content, one piece at a time
            for piece in tokenize(reply):
                emit({"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                      "created": created, "model": model,
                      "choices": [{"index": 0, "delta": {"content": piece},
                                   "finish_reason": None}]})
                if FAKE_DELAY:
                    time.sleep(FAKE_DELAY)
            # finish
            emit({"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                  "created": created, "model": model,
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            if want_usage:
                emit({"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                      "created": created, "model": model, "choices": [],
                      "usage": {"prompt_tokens": 0,
                                "completion_tokens": completion_tokens,
                                "total_tokens": completion_tokens}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # client hit Stop / navigated away — normal


if __name__ == "__main__":
    print(f"Fake echo server on http://0.0.0.0:{PORT}  (serves {INDEX_FILE}, echoes your input)")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
