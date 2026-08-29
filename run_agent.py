"""Entrypoint — one command, zero human intervention.

    python run_agent.py                         # offline planner (no API key needed)
    python run_agent.py --llm anthropic         # real Claude loop (needs ANTHROPIC_API_KEY)
    python run_agent.py --max-iters 8 --run-name demo

Writes runs/<name>/iterations.jsonl — read by the dashboard and scripts/analyze_runs.py.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from agent.llm import make_llm
from agent.orchestrator import run_agent
from agent.run_logger import RunLogger

ROOT = Path(__file__).resolve().parent
SEED_SOLUTION = ROOT / "solutions" / "solution.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=ROOT / "KuaiRand-Pure" / "data")
    ap.add_argument("--workspace", type=Path, default=ROOT / "workspace")
    ap.add_argument("--max-iters", type=int, default=14)
    ap.add_argument("--min-iters", type=int, default=7,
                    help="try at least this many ideas before allowing "
                         "convergence to stop the loop")
    ap.add_argument("--llm", choices=["auto", "offline", "anthropic"],
                    default="auto")
    ap.add_argument("--inject-fault", type=int, action="append", default=None,
                    help="offline planner only: make iteration N emit crashing "
                         "code, to demo the recovery path (repeatable)")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / name
    logger = RunLogger(run_dir)

    args.workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy(SEED_SOLUTION, args.workspace / "solution.py")

    llm = make_llm(args.llm, inject_faults=args.inject_fault or ())
    print(f"[run_agent] llm={type(llm).__name__}  run_dir={run_dir}")
    state = run_agent(args.workspace, args.data_dir, llm, logger,
                      max_iters=args.max_iters, min_iters=args.min_iters)
    print(f"[run_agent] done. best val primary {state.best_primary:.4f} "
          f"(iter {state.best_iter}, {state.best_method}). "
          f"log: {run_dir / 'iterations.jsonl'}")


if __name__ == "__main__":
    main()
