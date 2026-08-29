"""Template for a new experiment.

Copy this file to experiments/<your_idea>.py and fill in `run`. See
experiments/README.md for the full contract. Files starting with `_`
(like this one) are skipped by the agent's auto-discovery, so this file
itself is never run.
"""
# from evaluate import evaluate
# from data import encode

PRIORITY = 100                     # lower = agent.py runs it earlier (baseline=0)
DESCRIPTION = 'TODO: one-line description of the idea being tested'


def run(splits):
    """splits = data.load() output: {'train': [...], 'valid': [...], 'test': [...]}
    each a list of (date, user_id, video_id, author_id, tab, duration_ms, label) rows.

    Must return {'valid': {...}, 'test': {...}} where each side is exactly what
    evaluate.evaluate() returns: {'GAUC':.., 'nDCG@5':.., 'primary':.., 'users':.., 'rows':..}.

    Never modify evaluate.py — import and call it unmodified.
    """
    raise NotImplementedError("copy this file, rename it, and implement run()")
