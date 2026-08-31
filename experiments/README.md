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
6. Files starting with `_` (like `_template.py`, `_runner.py`) are skipped by discovery.
7. Your `run(splits)` executes in its **own subprocess** (`_runner.py`) with a
   hard 30-minute timeout. Keep a single train+eval well under that. If it
   crashes or hangs, the orchestrator logs the traceback and moves on — it
   never takes down the loop, and the champion is untouched.

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
| `baseline_fm.py` | (done) | Official FM, unmodified — starting champion | valid 0.6015 |
| `bpr_loss.py` | Person 1 | Pairwise ranking loss (BPR) | valid 0.6037 — tried within the scored window, not selected |
| `bpr_hard_negative.py` | agent | Hard-negative mining on top of BPR | valid 0.5803 — discarded |
| `bpr_hard_negative_warmstart.py` | agent | Warm-start before hard-negative mining | valid 0.6008 — discarded |
| `listwise_softmax.py` | agent | Listwise softmax loss | valid 0.6039 — **official checkpoint (submitted)** |
| `hour_of_day.py` | agent | Hour-of-day as a 6th field on top of listwise softmax | valid 0.6052 — best raw score in the run, found after convergence, not the scored checkpoint |
| `user_history.py` | Person 2 | DIN-style candidate-aware attention over each user's history | valid 0.6010 — found after convergence, discarded |
| `multitask.py` | agent | Multi-task FM: shared embeddings, aux head on is_click | valid 0.6018 — found after convergence, discarded |
| `seq_dur_drift.py` | agent | Short-term duration-preference drift as an extra FM field | valid 0.6017 — found after convergence, discarded, near-flat |

**Two separate mechanisms, not one.** The convergence rule (ε=0.002, N=3)
governs *when to stop*: `bpr_hard_negative`, `bpr_hard_negative_warmstart`,
and `listwise_softmax` are 3 consecutive attempts that each failed to beat
`bpr_loss` by more than ε, so the run stopped searching there. Separately,
*which checkpoint is submitted* is simply the highest validation score
found up through that stopping point, no ε filter — among attempts 0-4,
that's `listwise_softmax` (0.6039), not `bpr_loss` (0.6037), even though
neither individually cleared ε against the other. `agent.py` keeps running
remaining experiment files after convergence so nothing gets silently
dropped from the log (which is how `hour_of_day`'s higher raw score,
0.6052, got found and reported) — but nothing found that way is eligible
to be the scored checkpoint, since it falls outside "at that point." See
`agent.py`'s `official_checkpoint_*` fields and comments for the exact
logic, and `results/RESOURCES.md` for the full numbers.

A second, LLM-in-the-loop agent implementation also exists in `agent/` +
`solutions/runner.py` (real Anthropic API calls, live token accounting,
demonstrated crash recovery) - see `AGENT.md`. It's a different, arguably
more literal take on "the agent writes its own code" than the file-discovery
approach here. **Reconciled, not competing:** the two systems' broken
dependencies/imports were fixed, and the same discoveries
(`bpr_numpy`/`listwise_numpy`/`hour_of_day`) are ported into both, verified
to reproduce identical scores through either harness. `agent.py` here is the
scored path for this submission (see top-level `README.md`); the other
remains documented as a second, independently working system.

## How the agent uses this

`agent.py` runs every experiment file here in priority order, tracks the
best validation `primary` score seen so far (the "champion"), and keeps or
discards each new one using the eps/N convergence rule from the top-level
`README.md` (eps=0.002, N=3 consecutive non-improving attempts stops the
run). Every attempt is logged to `agent_log.jsonl` and summarized in
`agent_summary.json`.
