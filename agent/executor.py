"""Sandboxed execution of the agent's current solution.py.

Robustness is judged on how failures are HANDLED, not avoided: a crash or a
timeout here must never kill the agent loop. Everything comes back as an
ExecResult and the orchestrator decides what to do with it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STEP_TIMEOUT_S = int(os.environ.get("AGENT_STEP_TIMEOUT_S", 20 * 60))
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExecResult:
    metrics: dict | None      # {"gauc","ndcg5","primary"} or None on any failure
    error: str | None         # short, LLM-readable failure description
    stdout: str
    stderr: str
    returncode: int
    duration_s: float = 0.0


def run_solution(workspace: Path, data_dir: Path, split: str) -> ExecResult:
    import time
    sol = workspace / "solution.py"
    if not sol.exists():
        return ExecResult(None, "solution.py does not exist yet", "", "", -1)

    cmd = [sys.executable, str(sol), "--split", split,
           "--data-dir", str(data_dir)]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUTF8": "1",
           "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=workspace, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=STEP_TIMEOUT_S, env=env,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(None, f"TIMEOUT after {STEP_TIMEOUT_S}s",
                          e.stdout or "", e.stderr or "", -1, time.time() - t0)
    except Exception as e:  # noqa: BLE001 — the loop must survive anything
        return ExecResult(None, f"executor error: {e!r}", "", "", -1,
                          time.time() - t0)

    dur = time.time() - t0
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        return ExecResult(None, "crashed:\n" + tail, proc.stdout, proc.stderr,
                          proc.returncode, dur)

    metrics = _parse_metrics(proc.stdout)
    if metrics is None:
        return ExecResult(None, "no valid METRICS_JSON line in stdout",
                          proc.stdout, proc.stderr, proc.returncode, dur)

    sub = workspace / f"submission_{split}.csv"
    if not sub.exists():
        return ExecResult(None, f"missing {sub.name}", proc.stdout,
                          proc.stderr, proc.returncode, dur)

    return ExecResult(metrics, None, proc.stdout, proc.stderr,
                      proc.returncode, dur)


def _parse_metrics(stdout: str) -> dict | None:
    m = re.search(r"METRICS_JSON:\s*(\{.*\})", stdout)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not all(k in d and _finite(d[k]) for k in ("gauc", "ndcg5", "primary")):
        return None
    return {k: float(d[k]) for k in ("gauc", "ndcg5", "primary")}


def _finite(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v and abs(v) != float("inf")
