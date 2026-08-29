"""solution.py — the file the agent rewrites every iteration.

Seed state: the official Factorization Machine baseline (val primary ~0.6016),
so the run trajectory starts honestly at the number we have to beat.

METHOD : fm
HYP    : reproduce the starter-kit FM baseline end-to-end to anchor the loop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solutions.runner import run_variant

VARIANT = "fm"
PARAMS = {"k": 16, "lr": 0.001, "epochs": 25}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    metrics = run_variant(VARIANT, args.split, args.data_dir, PARAMS,
                          workspace=Path(__file__).resolve().parent)
    print("METRICS_JSON: " + json.dumps(metrics))


if __name__ == "__main__":
    main()
