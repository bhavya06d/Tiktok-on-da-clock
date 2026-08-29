"""Entrypoint. One command, zero human intervention — that's the Autonomy score.

  python run_agent.py --data-dir ./data                    # real LLM run
  python run_agent.py --data-dir ./data --mock             # harness test, no API
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from agent.llm import LLMClient, MockLLM
from agent.orchestrator import run_agent
from agent.run_logger import RunLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, default=Path("workspace"))
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--mock", action="store_true",
                    help="use MockLLM (reference solution) to test the harness")
    args = ap.parse_args()

    run_dir = Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    logger = RunLogger(run_dir)
    llm = (MockLLM("ml/reference_lgbm_ranker.py") if args.mock
           else LLMClient())

    state = run_agent(args.workspace, args.data_dir, llm, logger,
                      max_iters=args.max_iters)
    print(f"Done. Best val primary: {state.best_primary:.4f} "
          f"(iter {state.best_iter}). Logs: {run_dir}/iterations.jsonl")


if __name__ == "__main__":
    main()
