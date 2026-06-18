#!/usr/bin/env python3
"""
Proxy + power monitor for the RTX 5070 Ti node.
- Serves index.html
- Proxies /v1/* to llama-server (port 8080)
- Samples GPU power (nvidia-smi) + CPU package power (sensors / AMD "PPT"),
  exposes them and their sum at /power
Run with: python3 server_rtx5070ti.py   (no root needed)

Optional env vars:
  GPU_INDEX   restrict nvidia-smi to one GPU index (e.g. "0"); default: all visible
"""
import json, os, re, subprocess, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LLAMA = "http://127.0.0.1:8080"
PORT = 8000

# Restrict GPU power to a single index if requested (handy on a shared multi-GPU
# node where nvidia-smi would otherwise list every card).
GPU_INDEX = os.environ.get("GPU_INDEX")

# Matches the AMD "PPT" (Package Power Tracking) line from `sensors`, e.g.
#   PPT:          11.19 W  (avg =   5.09 W)
PPT_RE = re.compile(r"PPT:\s*([0-9.]+)\s*W", re.IGNORECASE)

# ---- shared power state ----
state = {
    "current_w": 0.0,   # latest TOTAL power = gpu_w + cpu_w (W)
    "gpu_w": 0.0,       # GPU power via nvidia-smi (W)
    "cpu_w": 0.0,       # CPU package power via sensors / PPT (W)
    "idle_w": None,     # learned idle baseline of the total (W)
    "active": False,    # set True by the page while generating
}
lock = threading.Lock()

def read_gpu_power():
    """GPU power (W) via nvidia-smi. Sums visible GPUs (or just GPU_INDEX)."""
    cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
    if GPU_INDEX:
        cmd += ["-i", GPU_INDEX]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        vals = []
        for tok in out.stdout.split("\n"):
            tok = tok.strip()
            if not tok:
                continue
            try:
                vals.append(float(tok))
            except ValueError:
                pass  # e.g. "[N/A]"
        return sum(vals) if vals else 0.0
    except Exception:
        return 0.0

def read_cpu_power():
    """CPU package power (W) by parsing the 'PPT' line from `sensors`. 0 if absent."""
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=5)
        m = PPT_RE.search(out.stdout)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 0.0

def power_sampler():
    """
    Samples GPU + CPU power once a second (no root needed) and keeps a rolling
    idle baseline of the total. Either source degrades to 0 W if unavailable.
    """
    idle_samples = []
    while True:
        try:
            gpu = read_gpu_power()
            cpu = read_cpu_power()
            total = gpu + cpu
            with lock:
                state["gpu_w"] = gpu
                state["cpu_w"] = cpu
                state["current_w"] = total
                if not state["active"]:
                    idle_samples.append(total)
                    if len(idle_samples) > 20:
                        idle_samples.pop(0)
                    # rolling median-ish idle estimate
                    s = sorted(idle_samples)
                    state["idle_w"] = s[len(s) // 2]
            # print(f"[power] gpu={gpu:.1f} cpu={cpu:.1f} total={total:.1f} active={state['active']}")
        except Exception as e:
            print(f"Error in power sampler thread: {e}")
        time.sleep(1)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass  # quiet
    
    # Even with the fix, ThreadingHTTPServer may still log some noise. You can suppress the handler's default error logging by overriding
    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/power":
            with lock:
                payload = json.dumps({
                    "current_w": round(state["current_w"], 2),
                    "gpu_w": round(state["gpu_w"], 2),
                    "cpu_w": round(state["cpu_w"], 2),
                    "idle_w": round(state["idle_w"], 2) if state["idle_w"] else None,
                    "active": state["active"],
                }).encode()
            self._send(200, payload)
        elif self.path.startswith("/v1/"):
            self._proxy("GET")
        elif self.path in ("/", "/index.html"):
            with open("index.html", "rb") as f:
                self._send(200, f.read(), "text/html")
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        print("do_POST triggered")
        if self.path == "/active":
            # page tells us when generation starts/stops
            ln = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(ln) or b"{}")
            with lock:
                state["active"] = bool(data.get("active", False))
            self._send(200, b'{"ok":true}')
        elif self.path.startswith("/v1/"):
            self._proxy("POST")
        else:
            self._send(404, b'{"error":"not found"}')

    def request_to_string(self, req):
        # 1. Get the HTTP Method (defaults to GET, or POST if data is present)
        method = req.get_method()
        
        # 2. Get the full destination URL
        url = req.get_full_url()
        
        # 3. Collect headers (combining standard and case-insensitive headers)
        headers = {**req.headers, **req.unredirected_hdrs}
        formatted_headers = "\n".join(f"{k}: {v}" for k, v in headers.items())
        
        # 4. Decode the body data if it exists
        body_str = ""
        if req.data:
            try:
                body_str = req.data.decode('utf-8')
            except UnicodeDecodeError:
                body_str = str(req.data)  # Fallback for binary data
                
            # 5. Combine everything into an HTTP-like plain text layout
            return (
                f"--- Outgoing Request ---\n"
                f"{method} {url}\n"
                f"{formatted_headers}\n"
                f"\n"
                f"{body_str}\n"
                f"------------------------"
            )
        
    def _proxy(self, method):
        ln = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(ln) if ln else None
        req = urllib.request.Request(
            LLAMA + self.path, data=body, method=method,
            headers={"Content-Type": "application/json"}
        )
        print(self.request_to_string(req))
        try:
            with urllib.request.urlopen(req) as r:
                self.send_response(r.status)
                self.send_header("Content-Type",
                                 r.headers.get("Content-Type", "application/json"))
                self.end_headers()
                
                for line in r:                 # yields up to each \n
                    try:
                        self.wfile.write(line)
                        #print(line)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                # stream chunks (important for SSE token streaming)
                """    
                while True:
                    chunk = r.read(512)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                        print(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        # client (browser) aborted the stream — normal on Stop/Pause
                        return
                """
        except (BrokenPipeError, ConnectionResetError):
            # connection already gone; nothing to send
            return
        except Exception as e:
             # only try to respond if the socket is still alive
            try:
                self._send(502, json.dumps({"error": str(e)}).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

if __name__ == "__main__":
    threading.Thread(target=power_sampler, daemon=True).start()
    print(f"Serving on http://0.0.0.0:{PORT}  (proxying llama at {LLAMA})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("\nStopping script.")
