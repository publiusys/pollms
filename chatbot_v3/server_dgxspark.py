#!/usr/bin/env python3
"""
Proxy + power monitor for DGX Spark.
- Serves index.html
- Proxies /v1/* to llama-server (port 8080)
- Samples powermetrics, exposes /power
Run with: sudo python3 server.py
"""
import json, subprocess, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LLAMA = "http://127.0.0.1:8080"
PORT = 8000

# ---- shared power state ----
state = {
    "current_w": 0.0,   # latest total package power (W)
    "idle_w": None,     # learned idle baseline (W)
    "active": False,    # set True by the page while generating
}
lock = threading.Lock()

def dgx_power_sampler():
    """
    Parses the combined power every second in the background.
    """
    cmd = ["sensors"]
    
    while True:
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            idle_samples = []
            for line in process.stdout:
                if "sys_total" in line:
                    #print(line.strip().split(" "))
                    watts = float(line.strip().split(" ")[-2])
                    with lock:
                        state["current_w"] = watts
                        if not state["active"]:
                            idle_samples.append(watts)
                            if len(idle_samples) > 20:
                                idle_samples.pop(0)
                            # rolling median-ish idle estimate
                            s = sorted(idle_samples)
                            state["idle_w"] = s[len(s)//2]
                        print(f"[power] current={state['current_w']:.2f} idle={state['idle_w']}  active={state['active']}")
            time.sleep(1)
        except Exception as e:
            print(f"Error in power sampler thread: {e}")

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

    def _proxy(self, method):
        ln = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(ln) if ln else None
        req = urllib.request.Request(
            LLAMA + self.path, data=body, method=method,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as r:
                self.send_response(r.status)
                self.send_header("Content-Type",
                                 r.headers.get("Content-Type", "application/json"))
                self.end_headers()
                # stream chunks (important for SSE token streaming)
                while True:
                    chunk = r.read(512)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        # client (browser) aborted the stream — normal on Stop/Pause
                        return 
        except (BrokenPipeError, ConnectionResetError):
            # connection already gone; nothing to send
            return
        except urllib.error.HTTPError as e:
            # forward llama-server's real error (status + JSON body) instead of masking it
            try:
                err_body = e.read() or json.dumps({"error": str(e)}).encode()
                self._send(e.code, err_body,
                           e.headers.get("Content-Type", "application/json"))
            except (BrokenPipeError, ConnectionResetError):
                pass
        except Exception as e:
             # only try to respond if the socket is still alive
            try:
                self._send(502, json.dumps({"error": str(e)}).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

if __name__ == "__main__":
    threading.Thread(target=dgx_power_sampler, daemon=True).start()
    print(f"Serving on http://0.0.0.0:{PORT}  (proxying llama at {LLAMA})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("\nStopping script.")
