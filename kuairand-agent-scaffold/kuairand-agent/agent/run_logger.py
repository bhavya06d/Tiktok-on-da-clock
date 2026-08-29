"""Person 3 (part 2) — per-iteration run log.

This file's output IS a required deliverable. Judges use it to score Autonomy
and Robustness. Per iteration we must record: hypothesis, code diff, resulting
metrics, and any error/recovery events. Plus a final summary with manual
intervention count, token totals, wall-clock, and iterations used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "iterations.jsonl"

    def _write(self, obj: dict) -> None:
        obj["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.path.open("a") as f:
            f.write(json.dumps(obj) + "\n")

    def iteration(self, idx: int, *, hypothesis: str, diff: str,
                  metrics: dict | None, error: str | None, accepted: bool,
                  tokens: dict, stdout_tail: str, stderr_tail: str) -> None:
        self._write({"type": "iteration", "iter": idx, "hypothesis": hypothesis,
                     "diff": diff, "metrics": metrics, "error": error,
                     "accepted": accepted, "tokens": tokens,
                     "stdout_tail": stdout_tail, "stderr_tail": stderr_tail})

    def event(self, idx: int, name: str, detail: str = "") -> None:
        """Recovery/lifecycle events: rollbacks, timeouts, convergence, etc."""
        self._write({"type": "event", "iter": idx, "event": name,
                     "detail": detail})

    def summary(self, state, *, wall_clock_s: float, total_tokens: dict) -> None:
        n_fail = sum(1 for r in state.history if r.metrics is None)
        self._write({
            "type": "summary",
            "iterations_used": len(state.history),
            "best_val_primary": state.best_primary,
            "best_iter": state.best_iter,
            "failed_iterations_recovered": n_fail,
            "manual_interventions": state.manual_interventions,  # target: 0
            "wall_clock_seconds": round(wall_clock_s, 1),
            "llm_tokens": total_tokens,
        })
