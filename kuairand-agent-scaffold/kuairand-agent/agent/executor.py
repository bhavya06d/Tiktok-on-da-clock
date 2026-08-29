"""Person 3 (part 1) — sandboxed execution.

Robustness (part of Technical Execution, 35%) is judged on how failures are
HANDLED, not avoided: a crash or timeout here must never kill the agent loop.
Everything returns an ExecResult; the orchestrator decides what to do with it.

Hardening backlog for Person 3:
- per-run temp copy of the workspace (so a bad iteration can't corrupt state)
- memory limit via `resource.setrlimit` in preexec_fn (Linux)
- kill the whole process group on timeout (start_new_session=True + os.killpg)
- deliberately inject failures in a test run and show recovery in the logs
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STEP_TIMEOUT_S = 15 * 60  # generous; baseline needs ~40s


@dataclass
class ExecResult:
    metrics: dict | None   # parsed METRICS_JSON, or None on any failure
    error: str | None      # short human/LLM-readable failure description
    stdout: str
    stderr: str
    returncode: int


def run_solution(workspace: Path, data_dir: Path, split: str) -> ExecResult:
    cmd = [sys.executable, "solution.py", "--split", split,
           "--data-dir", str(data_dir)]
    try:
        proc = subprocess.run(
            cmd, cwd=workspace, capture_output=True, text=True,
            timeout=STEP_TIMEOUT_S, start_new_session=True,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(None, f"TIMEOUT after {STEP_TIMEOUT_S}s",
                          (e.stdout or ""), (e.stderr or ""), -1)
    except Exception as e:  # noqa: BLE001 — the loop must survive anything
        return ExecResult(None, f"executor error: {e!r}", "", "", -1)

    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-15:]
        return ExecResult(None, "crashed:\n" + "\n".join(tail),
                          proc.stdout, proc.stderr, proc.returncode)

    metrics = _parse_metrics(proc.stdout)
    if metrics is None:
        return ExecResult(None, "no valid METRICS_JSON line in stdout",
                          proc.stdout, proc.stderr, proc.returncode)

    sub = workspace / f"submission_{split}.csv"
    if not sub.exists():
        return ExecResult(None, f"missing {sub.name}",
                          proc.stdout, proc.stderr, proc.returncode)

    return ExecResult(metrics, None, proc.stdout, proc.stderr, proc.returncode)


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
