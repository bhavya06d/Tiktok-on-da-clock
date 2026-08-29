# Experiment Contract

Drop a new file in this folder and `agent.py` automatically picks it up and
tests it — no need to edit `agent.py`, `baseline.py`, or coordinate with
anyone else. This is deliberate: everyone's changes land in their own new
file, so there is nothing to merge-conflict over.

## Rules

1. Copy `_template.py` to `<your_idea>.py` (e.g. `bpr_loss.py`, `user_history.py`, `multitask.py`).
2. Implement `run(splits) -> {'valid': {...}, 'test': {...}}` — the same shape
   `evaluate.evaluate()` returns. Feel free to copy the `FM` class out of
   `baseline.py` into your own file and modify it (new loss, new features,
   extra heads, whatever you need) — keep your training code in your own
   file rather than editing `baseline.py` directly, that's what keeps
   merges clean.
3. Optional module attributes: `PRIORITY` (int, default 100, lower runs
   earlier) and `DESCRIPTION` (str, shown in the agent's log).
4. Never modify `evaluate.py` — it's the fixed scoring contract.
5. Files starting with `_` (like `_template.py`) are skipped by discovery.

## Assigned ideas (see ../PLAN.md)

| File | Owner | Idea |
|---|---|---|
| `baseline_fm.py` | (done) | Official FM, unmodified — starting champion |
| `bpr_loss.py` | Person 1 | Pairwise ranking loss (BPR) |
| `user_history.py` | Person 2 | User interaction history / sequence features |
| `multitask.py` | Person 3 | Multi-task auxiliary labels |

## How the agent uses this

`agent.py` runs every experiment file here in priority order, tracks the
best validation `primary` score seen so far (the "champion"), and keeps or
discards each new one using the eps/N convergence rule from the top-level
`README.md` (eps=0.002, N=3 consecutive non-improving attempts stops the
run). Every attempt is logged to `agent_log.jsonl` and summarized in
`agent_summary.json`.
