"""The official FM baseline, unmodified. Sets the agent's starting champion.

Not owned by anyone — this is just the existing baseline.py wrapped to match
the experiment contract (see experiments/README.md).
"""
from baseline import run_fm

PRIORITY = 0
DESCRIPTION = 'Official FM baseline (k=16, pointwise logloss) — unmodified, starting champion.'
AUTHOR = 'baseline'  # the fixed reference point, not a proposed idea from anyone


def run(splits):
    return run_fm(splits, verbose=False)
