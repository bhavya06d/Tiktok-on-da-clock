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
3. Set `AUTHOR = 'human'` or `AUTHOR = 'agent'` — see "Why AUTHOR matters" below.
   This is not optional, be honest about it.
4. Optional module attributes: `PRIORITY` (int, default 100, lower runs
   earlier) and `DESCRIPTION` (str, shown in the agent's log — treat it as
   the hypothesis being tested, judges read this).
5. Never modify `evaluate.py` — it's the fixed scoring contract.
6. Files starting with `_` (like `_template.py`) are skipped by discovery.

## Why AUTHOR matters (read this)

The hackathon's actual requirement is that an AI proposes and writes the
iteration code, not a human — see the official problem statement:
*"Writing the code for each stage is part of the agent's job, not something
provided in advance."* Autonomy (how little human intervention a run needed)
is directly graded.

Hand-implementing an idea yourself (like Person 1's `bpr_loss.py` — real,
valuable, tested work) is still worth doing: it's background validation and
a safety-net score. But it is a **manual intervention**, not the thing being
scored as "the agent." `agent.py` tracks two separate champions so this
never gets blurred:

- **Overall champion** — best score from any author, i.e. what's actually
  achievable right now.
- **Autonomous champion** — best score among `AUTHOR = 'agent'` entries
  only, i.e. what the agent found *on its own*. This is the number that
  answers the hackathon's real question, and it's the one that should be
  the centerpiece of the submission log.

See ../PLAN.md for the fuller writeup of this distinction.

## Ideas delivered so far (see ../PLAN.md)

| File | Owner | Idea | Result |
|---|---|---|---|
| `baseline_fm.py` | (done) | Official FM, unmodified — starting champion | valid 0.6015 (anchor) |
| `bpr_loss.py` | Person 1 | Pairwise ranking loss (BPR) | valid 0.6037 — best score achievable |
| `bpr_hard_negative.py`, `bpr_hard_negative_warmstart.py` | agent | Hard-negative mining on BPR (two attempts) | both underperform champion |
| `listwise_softmax.py` | agent | Listwise softmax loss (BPR generalized to M negatives) | valid 0.6039 — best fully-autonomous score |
| `hour_of_day.py` | agent | Hour-of-day feature on top of listwise | valid 0.6052 — best raw score in the repo |
| `user_history.py` | Person 2 | DIN-style candidate-aware attention over user history | valid 0.6010 — underperforms, real negative result (leakage-checked) |
| `seq_dur_drift.py` | Person 2 (agent-tagged) | Short-term duration-preference drift feature | valid 0.6017 — near-flat |

`multitask.py` (Person 3's assigned idea) hasn't landed here yet, though a
multitask variant exists in the parallel `run_agent.py`/`solutions/runner.py`
system (also underperforms there, val 0.5880) - see AGENT.md.

## How the agent uses this

`agent.py` runs every experiment file here in priority order, tracks the
best validation `primary` score seen so far (the "champion"), and keeps or
discards each new one using the eps/N convergence rule from the top-level
`README.md` (eps=0.002, N=3 consecutive non-improving attempts stops the
run). Every attempt is logged to `agent_log.jsonl` and summarized in
`agent_summary.json`.
