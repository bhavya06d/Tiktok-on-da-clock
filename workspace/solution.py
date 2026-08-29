# -*- coding: utf-8 -*-
"""solution.py - rewritten by the agent on iteration 0.

METHOD : fm
HYP    : Anchor the run: reproduce the official Factorization Machine baseline
         end-to-end (pointwise logloss, 5 categorical fields). Every
         subsequent idea is judged against this number and the oracle ceiling
         (0.8484), not against 1.0.
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

VARIANT = 'fm'
PARAMS = {'k': 16, 'lr': 0.001, 'epochs': 25}


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

