"""Per-iteration run log — a required deliverable.

Judges read iterations.jsonl to score Autonomy and Robustness. Per iteration:
hypothesis, code diff, resulting metrics, error/recovery events. Plus a final
summary: manual-intervention count, token totals, wall-clock, iterations used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "iterations.jsonl"

    def _write(self, obj: dict) -> None:
        obj["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    def meta(self, **kw) -> None:
        self._write({"type": "meta", **kw})

    def iteration(self, idx: int, *, hypothesis: str, method: str, diff: str,
                  code: str, metrics: dict | None, error: str | None,
                  accepted: bool, decision: str, tokens: dict,
                  duration_s: float, stdout_tail: str, stderr_tail: str,
                  best_primary: float) -> None:
        self._write({
            "type": "iteration", "iter": idx, "hypothesis": hypothesis,
            "method": method, "diff": diff, "code": code, "metrics": metrics,
            "error": error, "accepted": accepted, "decision": decision,
            "tokens": tokens, "duration_s": round(duration_s, 1),
            "stdout_tail": stdout_tail, "stderr_tail": stderr_tail,
            "best_primary": best_primary,
        })

    def event(self, idx: int, name: str, detail: str = "") -> None:
        self._write({"type": "event", "iter": idx, "event": name,
                     "detail": detail})

    def summary(self, state, *, wall_clock_s: float, total_tokens: dict,
                baseline_primary: float, oracle_primary: float) -> None:
        n_fail = sum(1 for r in state.history if r.metrics is None)
        best = state.best_primary if state.history else float("nan")
        headroom = ((best - baseline_primary) /
                    (oracle_primary - baseline_primary)
                    if state.history else 0.0)
        self._write({
            "type": "summary",
            "iterations_used": len(state.history),
            "best_val_primary": best,
            "best_iter": state.best_iter,
            "best_method": state.best_method,
            "delta_over_baseline": (best - baseline_primary
                                    if state.history else 0.0),
            "pct_of_oracle_headroom": round(headroom, 4),
            "failed_iterations_recovered": n_fail,
            "manual_interventions": state.manual_interventions,
            "wall_clock_seconds": round(wall_clock_s, 1),
            "llm_tokens": total_tokens,
            "baseline_primary": baseline_primary,
            "oracle_primary": oracle_primary,
        })
