"""The agent loop: propose -> apply code -> execute -> score -> decide -> reflect.

Owns convergence detection (README's own rule: ε=0.002, N=3), best-checkpoint
tracking, and rollback to the last good solution when an iteration crashes.
"""
from __future__ import annotations

import ast
import difflib
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .executor import ExecResult, run_solution
from .prompts import build_iteration_prompt, parse_agent_reply
from .run_logger import RunLogger

EPSILON = 0.002
PATIENCE = 3
MAX_ITERS = 50
WALL_CLOCK_LIMIT_S = 6 * 3600

BASELINE_PRIMARY = 0.6016
ORACLE_PRIMARY = 0.8484        # validation oracle ceiling


@dataclass
class IterationRecord:
    idx: int
    hypothesis: str
    method: str
    metrics: dict | None
    error: str | None
    accepted: bool
    variant: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class AgentState:
    best_primary: float = float("-inf")
    best_iter: int = -1
    best_method: str = ""
    history: list[IterationRecord] = field(default_factory=list)
    manual_interventions: int = 0


def converged(state: AgentState) -> bool:
    """True when val primary has not improved by > EPSILON over the last
    PATIENCE iterations (each compared against the best before that window)."""
    if len(state.history) < PATIENCE + 1:
        return False
    recent = state.history[-PATIENCE:]
    best_before = max((r.metrics["primary"] for r in state.history[:-PATIENCE]
                       if r.metrics), default=float("-inf"))
    best_recent = max((r.metrics["primary"] for r in recent if r.metrics),
                      default=float("-inf"))
    return best_recent - best_before <= EPSILON


def _extract_variant_params(code: str) -> tuple[str, dict]:
    v = re.search(r"^VARIANT\s*=\s*(.+)$", code, re.MULTILINE)
    p = re.search(r"^PARAMS\s*=\s*(\{.*?\})\s*$", code, re.MULTILINE | re.DOTALL)
    variant = ""
    params: dict = {}
    if v:
        try:
            variant = ast.literal_eval(v.group(1).strip())
        except Exception:  # noqa: BLE001
            variant = v.group(1).strip().strip("'\"")
    if p:
        try:
            params = ast.literal_eval(p.group(1))
        except Exception:  # noqa: BLE001
            params = {}
    return variant, params


def run_agent(workspace: Path, data_dir: Path, llm, logger: RunLogger,
              max_iters: int = MAX_ITERS, min_iters: int = 0) -> AgentState:
    workspace.mkdir(parents=True, exist_ok=True)
    state = AgentState()
    solution = workspace / "solution.py"
    best_copy = workspace / "solution_best.py"
    t0 = time.time()

    logger.meta(epsilon=EPSILON, patience=PATIENCE, max_iters=max_iters,
                baseline_primary=BASELINE_PRIMARY, oracle_primary=ORACLE_PRIMARY,
                llm=type(llm).__name__)

    for i in range(max_iters):
        if time.time() - t0 > WALL_CLOCK_LIMIT_S:
            logger.event(i, "wall_clock_limit_reached")
            break

        old_code = (solution.read_text(encoding="utf-8")
                    if solution.exists() else "")
        prompt = build_iteration_prompt(old_code, state)
        reply, usage = llm.complete(prompt, state)
        hypothesis, method, new_code = parse_agent_reply(reply)

        if new_code is None:
            logger.event(i, "unparseable_reply", detail=reply[:400])
            reply, usage2 = llm.complete(
                prompt + "\n\nYour last reply had no python code block. "
                "Reply again with the FULL solution.py.", state)
            usage = _merge_usage(usage, usage2)
            hypothesis, method, new_code = parse_agent_reply(reply)
            if new_code is None:
                state.history.append(IterationRecord(
                    i, hypothesis or "n/a", method, None,
                    "no code produced after retry", False))
                logger.iteration(i, hypothesis=hypothesis or "n/a",
                                 method=method, diff="", code="", metrics=None,
                                 error="no code produced", accepted=False,
                                 decision="skipped", tokens=usage,
                                 duration_s=0.0, stdout_tail="", stderr_tail="",
                                 best_primary=_bp(state))
                continue

        solution.write_text(new_code, encoding="utf-8")
        variant, params = _extract_variant_params(new_code)
        diff = "\n".join(difflib.unified_diff(
            old_code.splitlines(), new_code.splitlines(),
            "solution.py (before)", "solution.py (after)", lineterm=""))

        result: ExecResult = run_solution(workspace, data_dir, split="val")

        if result.metrics is None:
            logger.event(i, "ERROR", detail=result.error)
            rec = IterationRecord(i, hypothesis, method, None, result.error,
                                  False, variant, params)
            decision = "failed"
            if best_copy.exists():
                shutil.copy(best_copy, solution)
                logger.event(i, "RECOVERY",
                             detail=f"Rolled back to best solution from iteration {state.best_iter}")
                decision = "failed -> rolled back to best"
        else:
            primary = result.metrics["primary"]
            accepted = primary > state.best_primary + 0.0  # strict improvement
            if accepted:
                state.best_primary = primary
                state.best_iter = i
                state.best_method = method
                shutil.copy(solution, best_copy)
                decision = f"KEEP (new best {primary:.4f})"
            else:
                gap = state.best_primary - primary
                decision = (f"DISCARD (primary {primary:.4f}, "
                            f"{gap:.4f} below best {state.best_primary:.4f})")
            rec = IterationRecord(i, hypothesis, method, result.metrics, None,
                                  accepted, variant, params)

        state.history.append(rec)
        logger.iteration(
            i, hypothesis=hypothesis, method=method, diff=diff, code=new_code,
            metrics=rec.metrics, error=rec.error, accepted=rec.accepted,
            decision=decision, tokens=usage, duration_s=result.duration_s,
            stdout_tail=result.stdout[-2000:], stderr_tail=result.stderr[-2000:],
            best_primary=_bp(state))

        if len(state.history) >= max(min_iters, PATIENCE + 1) and converged(state):
            logger.event(i, "converged",
                         detail=f"no >{EPSILON} gain over last {PATIENCE} iters; "
                                f"best={state.best_primary:.4f}")
            break

    # restore best checkpoint and produce the test-split submission
    if best_copy.exists():
        shutil.copy(best_copy, solution)
        tr = run_solution(workspace, data_dir, split="test")
        logger.event(len(state.history), "final_test_submission",
                     detail=("ok" if tr.metrics else (tr.error or "failed")))

    logger.summary(state, wall_clock_s=time.time() - t0,
                   total_tokens=llm.total_tokens(),
                   baseline_primary=BASELINE_PRIMARY,
                   oracle_primary=ORACLE_PRIMARY)
    return state


def _bp(state: AgentState) -> float:
    return state.best_primary if state.best_primary != float("-inf") else 0.0


def _merge_usage(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}