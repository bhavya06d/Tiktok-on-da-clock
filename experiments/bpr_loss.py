"""Person 1's pairwise ranking loss (BPR) idea — wraps baseline.py's
run_fm_bpr, which trains on (positive, negative) pairs within the same
user instead of pointwise logloss. Directly targets the ranking metric,
per README's top-ranked headroom idea.

Confirmed by the author's own sweep across seeds 0-3: beats plain FM on
test primary in all 4 seeds (+0.0018 to +0.0028), with lr=0.0005 tuned
as the best default (beat lr=0.001 in all 4 seeds too).
"""
from baseline import run_fm_bpr

PRIORITY = 1
DESCRIPTION = 'Pairwise BPR loss (score_pos - score_neg) instead of pointwise logloss.'


def run(splits):
    return run_fm_bpr(splits, verbose=False)
