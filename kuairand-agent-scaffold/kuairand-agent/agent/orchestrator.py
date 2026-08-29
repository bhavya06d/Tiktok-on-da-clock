"""Person 1 — the agent loop.

State machine: propose -> apply code -> execute -> score -> reflect -> repeat.
Owns convergence detection, best-checkpoint tracking, and rollback on failure.
"""
from __future__ import annotations

import difflib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .executor import ExecResult, run_solution
from .llm import LLMClient
from .prompts import build_iteration_prompt, parse_agent_reply
from .run_logger import RunLogger

EPSILON = 0.002          # convergence: no improvement > eps ...
PATIENCE = 3             # ... over the last N consecutive iterations
MAX_ITERS = 50           # hard cap from the rules
WALL_CLOCK_LIMIT_S = 6 * 3600


@dataclass
class IterationRecord:
    idx: int
    hypothesis: str
    metrics: dict | None          # {"gauc", "ndcg5", "primary"} or None on failure
    error: str | None
    accepted: bool                # became the new best?


@dataclass
class AgentState:
    best_primary: float = float("-inf")
    best_iter: int = -1
    history: list[IterationRecord] = field(default_factory=list)
    manual_interventions: int = 0  # increment ONLY if a human ever edits mid-run


def converged(state: AgentState) -> bool:
    """True when the best val primary hasn't improved by > EPSILON in the
    last PATIENCE iterations (compare each recent iter's best-so-far)."""
    if len(state.history) < PATIENCE + 1:
        return False
    recent = state.history[-PATIENCE:]
    best_before = max(
        (r.metrics["primary"] for r in state.history[:-PATIENCE] if r.metrics),
        default=float("-inf"),
    )
    best_recent = max((r.metrics["primary"] for r in recent if r.metrics),
                      default=float("-inf"))
    return best_recent - best_before <= EPSILON


def run_agent(workspace: Path, data_dir: Path, llm: LLMClient,
              logger: RunLogger, max_iters: int = MAX_ITERS) -> AgentState:
    state = AgentState()
    solution = workspace / "solution.py"
    best_copy = workspace / "solution_best.py"
    t0 = time.time()

    for i in range(max_iters):
        if time.time() - t0 > WALL_CLOCK_LIMIT_S:
            logger.event(i, "wall_clock_limit_reached")
            break

        old_code = solution.read_text() if solution.exists() else ""
        prompt = build_iteration_prompt(old_code, state)
        reply, usage = llm.complete(prompt)
        hypothesis, new_code = parse_agent_reply(reply)

        if new_code is None:  # LLM failed to produce code -> retry once, else skip
            logger.event(i, "unparseable_reply", detail=reply[:500])
            reply, usage2 = llm.complete(prompt + "\n\nYour last reply had no code "
                                         "block. Reply again with the FULL solution.py.")
            usage = _merge_usage(usage, usage2)
            hypothesis, new_code = parse_agent_reply(reply)
            if new_code is None:
                state.history.append(IterationRecord(i, hypothesis or "n/a",
                                                     None, "no code produced", False))
                continue

        solution.write_text(new_code)
        diff = "\n".join(difflib.unified_diff(old_code.splitlines(),
                                              new_code.splitlines(),
                                              "before", "after", lineterm=""))

        result: ExecResult = run_solution(workspace, data_dir, split="val")

        if result.metrics is None:
            # Robustness path: log the failure, roll back, and let the NEXT
            # iteration see the traceback in its history so the LLM can fix it.
            rec = IterationRecord(i, hypothesis, None, result.error, False)
            if best_copy.exists():
                shutil.copy(best_copy, solution)   # rollback to last good code
                logger.event(i, "rolled_back_to_best")
        else:
            primary = result.metrics["primary"]
            accepted = primary > state.best_primary
            if accepted:
                state.best_primary, state.best_iter = primary, i
                shutil.copy(solution, best_copy)
            rec = IterationRecord(i, hypothesis, result.metrics, None, accepted)

        state.history.append(rec)
        logger.iteration(i, hypothesis=hypothesis, diff=diff,
                         metrics=rec.metrics, error=rec.error,
                         accepted=rec.accepted, tokens=usage,
                         stdout_tail=result.stdout[-1500:],
                         stderr_tail=result.stderr[-1500:])

        if converged(state):
            logger.event(i, "converged", detail=f"best={state.best_primary:.4f}")
            break

    # Final: restore best checkpoint and produce the test-split submission file.
    if best_copy.exists():
        shutil.copy(best_copy, solution)
        run_solution(workspace, data_dir, split="test")
    logger.summary(state, wall_clock_s=time.time() - t0,
                   total_tokens=llm.total_tokens())
    return state


def _merge_usage(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
