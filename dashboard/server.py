"""Live dashboard for agent runs.

    python dashboard/server.py            # http://127.0.0.1:5000
    python dashboard/server.py --port 8000 --run <name>

Reads runs/<name>/iterations.jsonl on every poll, so it updates live while
run_agent.py is still going. Pure stdlib (http.server) — no Flask needed.
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
HERE = Path(__file__).resolve().parent


def parse_run(name: str) -> dict:
    path = RUNS / name / "iterations.jsonl"
    out = {"name": name, "meta": {}, "iterations": [], "events": [],
           "summary": None, "exists": path.exists()}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "iteration":
            out["iterations"].append(obj)
        elif t == "event":
            out["events"].append(obj)
        elif t == "summary":
            out["summary"] = obj
        elif t == "meta":
            out["meta"] = obj
    out["status"] = ("converged" if out["summary"]
                     else ("running" if out["iterations"] else "starting"))
    return out


def list_runs() -> list[str]:
    if not RUNS.exists():
        return []
    return sorted((p.name for p in RUNS.iterdir()
                   if (p / "iterations.jsonl").exists()), reverse=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_text("utf-8"),
                              "text/html; charset=utf-8")
        if p == "/app.js":
            return self._send(200, (HERE / "app.js").read_text("utf-8"),
                              "application/javascript")
        if p == "/api/runs":
            return self._send(200, {"runs": list_runs()})
        m = re.match(r"/api/run/([\w.\-]+)$", p)
        if m:
            return self._send(200, parse_run(m.group(1)))
        self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"dashboard: http://{a.host}:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
