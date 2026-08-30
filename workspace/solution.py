# -*- coding: utf-8 -*-
"""solution.py - rewritten by the agent on iteration 2.

METHOD : hour_of_day
HYP    : Every idea so far only changed the loss function - nothing has
         touched what the model actually sees. Add hour-of-day (from the raw
         hourmin column) as a 6th categorical field crossed via FM's pairwise
         interaction term: day-parting (taste shifting morning/evening/night)
         is a known effect in recommendation, and the README's own headroom
         list names time features as untried. Validated in a parallel harness:
         val primary 0.6052 vs listwise's 0.6039 - best score found by either
         harness so far.
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

VARIANT = 'hour_of_day'
PARAMS = {'k': 16, 'lr': 0.0005, 'epochs': 40, 'm_neg': 4}


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

