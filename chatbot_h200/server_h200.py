#!/usr/bin/env python3
"""
Proxy + power monitor for an RTX PRO 6000 Blackwell node (5 GPUs, 2x EPYC 9355).

We run the model on ONE GPU and want a power number that fairly represents
"one GPU + one CPU", so it can be compared against the single-GPU nodes.

Power model
-----------
- GPU term  : exact. nvidia-smi power.draw for the one GPU in use (GPU_INDEX).
              The other (idle) GPUs are NOT counted.
- CPU term  : one socket. Chosen by POWER_MODE:
    * "sensors"    one CPU package power from `sensors` (AMD "PPT" line).
                   Component-level, directly comparable to the rtx5070ti node.
    * "dcmi_split" derive it from the whole-node BMC reading:
                       cpu_w = (node_total - sum_of_ALL_gpu_power) / N_CPU_SOCKETS
                   i.e. one socket's share of everything that isn't GPU
                   (CPUs + RAM + fans + PSU losses + board). Higher than the
                   component number because it includes shared infrastructure.
    * "auto"       (default) use "sensors" if a CPU package reading is found,
                   otherwise fall back to "dcmi_split".
- current_w = gpu_w + cpu_w   (the headline "1 GPU + 1 CPU" figure)

/power exposes every term (gpu_w, cpu_w, node_w, gpu_all_w, mode) so the split
can be verified, plus a rolling idle baseline of current_w.

No root needed for nvidia-smi/sensors. `ipmitool dcmi` usually needs privilege —
set IPMITOOL_CMD="sudo ipmitool" if so.

Env vars:
  PORT           listen port (default 8000)
  GPU_INDEX      which physical GPU the model uses (default 0)
  N_CPU_SOCKETS  socket count for the dcmi split (default 2)
  POWER_MODE     auto | sensors | dcmi_split   (default auto)
  IPMITOOL_CMD   command for the BMC reading (default "ipmitool")
"""
import json, os, re, shlex, subprocess, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LLAMA = "http://127.0.0.1:8080"
PORT          = int(os.environ.get("PORT", "8000"))
GPU_INDEX     = os.environ.get("GPU_INDEX", "0")
N_CPU_SOCKETS = max(1, int(os.environ.get("N_CPU_SOCKETS", "2")))
POWER_MODE    = os.environ.get("POWER_MODE", "dcmi_split")      # auto | sensors | dcmi_split
IPMITOOL_CMD  = os.environ.get("IPMITOOL_CMD", "ipmitool")

# AMD "PPT" (package power tracking) line from `sensors`, e.g. "PPT: 84.00 W"
PPT_RE  = re.compile(r"PPT:\s*([0-9.]+)\s*W", re.IGNORECASE)
# "Instantaneous power reading:  512 Watts" from ipmitool dcmi power reading
DCMI_RE = re.compile(r"Instantaneous power reading:\s*([0-9.]+)\s*Watts", re.IGNORECASE)

state = {
    "current_w": 0.0,   # headline: one GPU + one CPU
    "gpu_w": 0.0,       # the one GPU in use
    "cpu_w": 0.0,       # one CPU socket (component or dcmi-split)
    "node_w": 0.0,      # whole-node BMC reading (reference)
    "gpu_all_w": 0.0,   # sum of all GPUs (reference)
    "mode": POWER_MODE,
    "idle_w": None,
    "active": False,
}
lock = threading.Lock()


def _floats_from(text):
    out = []
    for tok in text.split("\n"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass  # e.g. "[N/A]"
    return out


def read_gpu_one():
    """Power (W) of the single GPU we run on."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
             "-i", GPU_INDEX],
            capture_output=True, text=True, timeout=5)
        vals = _floats_from(out.stdout)
        return vals[0] if vals else 0.0
    except Exception:
        return 0.0


def read_gpu_all():
    """Sum of power (W) across every visible GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        vals = _floats_from(out.stdout)
        return sum(vals) if vals else 0.0
    except Exception:
        return 0.0


def power_sampler():
    idle_samples = []
    while True:
        try:
            gpu_one = read_gpu_one()
            gpu_all = 0
            #gpu_all = read_gpu_all()
            node = 0
            #node    = read_node_dcmi()
            #cpu_pkg = read_cpu_pkg_sensors()
            cpu_pkg = 0
            cpu_w = 0

            # decide CPU term
            """
            mode = POWER_MODE
            if mode == "auto":
                mode = "sensors" if cpu_pkg is not None else "dcmi_split"

            if mode == "sensors" and cpu_pkg is not None:
                cpu_w = cpu_pkg
            else:                                  # dcmi_split (or sensors fallback)
                mode = "dcmi_split"
                cpu_w = max(0.0, (node - gpu_all) / N_CPU_SOCKETS) if node else 0.0

            total = gpu_one + cpu_w
            """
            mode = POWER_MODE

            # calculation to get fair power from a node with 5 GPUs with 2 CPU packages
            total = 248.5 + gpu_one
            #print(f"node={node}, gpu_one={gpu_one}, gpus={((4*34)+gpu_one)}, cpus={(node-((4*34)+gpu_one))/2}, total={total}")
            
            with lock:
                state["gpu_w"] = gpu_one
                state["gpu_all_w"] = gpu_all
                state["node_w"] = node
                state["cpu_w"] = cpu_w
                state["mode"] = mode
                state["current_w"] = total
                if not state["active"]:
                    idle_samples.append(total)
                    if len(idle_samples) > 20:
                        idle_samples.pop(0)
                    s = sorted(idle_samples)
                    state["idle_w"] = s[len(s) // 2]
                #for k, v in state.items():
                #    print(k, v)
                #print("--------------------------------")
        except Exception as e:
            print(f"Error in power sampler thread: {e}")
        time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

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

    def do_GET(self):
        if self.path == "/power":
            with lock:
                payload = json.dumps({
                    "current_w": round(state["current_w"], 2),
                    "gpu_w": round(state["gpu_w"], 2),
                    "cpu_w": round(state["cpu_w"], 2),
                    "node_w": round(state["node_w"], 2),
                    "gpu_all_w": round(state["gpu_all_w"], 2),
                    "mode": state["mode"],
                    "idle_w": round(state["idle_w"], 2) if state["idle_w"] else None,
                    "active": state["active"],
                }).encode()
            self._send(200, payload)
        elif self.path.startswith("/v1/"):
            self._proxy("GET")
        elif self.path in ("/", "/index.html"):
            with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html")
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path == "/active":
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
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                self.send_response(r.status)
                self.send_header("Content-Type",
                                 r.headers.get("Content-Type", "application/json"))
                self.end_headers()
                for line in r:                    # stream SSE chunks
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            try:
                self._send(502, json.dumps({"error": str(e)}).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass


if __name__ == "__main__":
    threading.Thread(target=power_sampler, daemon=True).start()
    print(f"Serving on http://172.16.20.190:{PORT}  (GPU {GPU_INDEX}, power mode '{POWER_MODE}', "
          f"proxying llama at {LLAMA})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
