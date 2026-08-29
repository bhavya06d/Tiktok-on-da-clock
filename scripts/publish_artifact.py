"""Snapshot a finished run into a single self-contained HTML file for Devpost.

    python scripts/publish_artifact.py                       # latest run
    python scripts/publish_artifact.py --run demo --out site/report.html

The output inlines the dashboard CSS/JS and bakes the run log in as RUN_DATA,
so it needs no server. Chart.js still loads from cdnjs (allowed by the Artifact
CSP). Hand the path to the Artifact tool to publish it.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"


def parse_run(name: str) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("dsrv", DASH / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_run(name)


def latest_run() -> str:
    runs = sorted((ROOT / "runs").glob("*/iterations.jsonl"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("no runs found")
    return runs[-1].parent.name


def build(name: str, out: Path) -> None:
    data = parse_run(name)
    html = (DASH / "index.html").read_text("utf-8")
    app = (DASH / "app.js").read_text("utf-8")

    # drop the live-polling boot; force static render
    app = app.replace(
        'if (typeof RUN_DATA !== "undefined") render(RUN_DATA);\nelse live();',
        "render(RUN_DATA);")

    inject = (f"<script>window.RUN_DATA = {json.dumps(data)};</script>\n"
              f"<script>{app}</script>")
    html = html.replace('<script src="app.js"></script>', inject)

    # The Artifact host wraps the file in its own <!doctype/html/head/body>.
    # Strip our skeleton but keep <title>, the CDN <link>/<script>, <style>.
    for tag in ("<!doctype html>", '<html lang="en">', "<head>", "</head>",
                "<body>", "</body>", "</html>",
                '<meta charset="utf-8" />',
                '<meta name="viewport" content="width=device-width, initial-scale=1" />'):
        html = html.replace(tag, "")
    html = html.strip()
    # standalone banner
    html = html.replace(
        '<span class="badge starting" id="status-badge">…</span>',
        '<span class="badge starting" id="status-badge">…</span>'
        '<span style="margin-left:auto;font-size:12px;color:var(--muted)">'
        'snapshot · not live</span>')

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, "utf-8")
    S = data.get("summary") or {}
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    if S:
        print(f"  best {S['best_val_primary']:.4f}  Δ{S['delta_over_baseline']:+.4f}  "
              f"{S['iterations_used']} iters  {S['failed_iterations_recovered']} recovered")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "site" / "report.html")
    a = ap.parse_args()
    build(a.run or latest_run(), a.out)
