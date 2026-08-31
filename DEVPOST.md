# Devpost — Autonomous ML Research Agent for KuaiRand-Pure

## Inspiration

Most "beat the baseline" entries hand-tune a model and report a number. The
KuaiRand-Pure starter kit itself says that's hard here: the FM baseline already
consumes ~30% of the usable range between random and a perfect oracle, and every
static-feature / capacity tweak the organizers tried failed. So we built the
thing that's actually worth watching — an autonomous research loop that proposes
an idea, writes the code for it, runs the official scorer, decides
keep-or-discard against the running champion, and moves on, logging every step.

## What it does

`python agent.py` runs unattended. It auto-discovers every `experiments/*.py`
(each a self-contained hypothesis exposing `run(splits)`), and for each one:

1. **runs** it — trains on the train split only, scores on validation with the
   **unmodified** official `evaluate.py`;
2. **decides** — keep as the new champion only if it beats the current
   champion's validation primary by more than ε = 0.002; otherwise discard;
3. **converges** — after N = 3 consecutive non-improving attempts it locks in
   the scored checkpoint (`converged_at` in the summary);
4. **logs** — every attempt's hypothesis, **full source code**, valid + test
   metrics, decision, and any error/traceback goes to `agent_log.jsonl`;
   `agent_summary.json` carries the champions, resource totals and convergence
   point. `dashboard.py` renders both into a static HTML report.

Each idea is written by Claude in-session (tagged `AUTHOR = 'agent'`) or by a
teammate (`AUTHOR = 'human'`, counted as a manual intervention). `agent.py`
tracks **two champions in parallel** — overall best, and best among
agent-authored ideas only — so the autonomy claim is never inflated.

## How we frame results

Against the **oracle ceiling (0.8484 val / 0.8645 test)**, not 1.0 — the FM
baseline already sits at ~31% of that headroom, so "0.60 vs a perfect 1.0" is
the wrong mental model.

## What the agent found

| | val primary | test primary | vs FM baseline |
|---|---|---|---|
| FM baseline (official) | 0.6016 | 0.5946 | — |
| listwise softmax loss (agent) | 0.6039 | 0.5973 | +0.0027 |
| **+ hour-of-day feature (agent) — champion** | **0.6052** | **0.5986** | **+0.0040** |

Two findings we'd highlight:

1. **Idea vs. implementation.** A ranking loss *does* beat pointwise logloss
   here — but our first attempt (a warm-started torch BPR with K=24
   list-sampling) scored 0.565, which reads exactly like "the idea failed." The
   same idea in ~40 lines of plain numpy (listwise softmax, M=4 random
   negatives) was a real win at 0.6039. The agent's job includes noticing that
   distinction before discarding a hypothesis.
2. **The first feature-side win.** Every idea up to `hour_of_day` only changed
   the loss function. `hour_of_day` adds one categorical field (hour, from the
   raw `hourmin` column — day-parting is a known recsys effect and time
   features were the README's own untried headroom item). A same-day
   day-of-week follow-up added nothing — 14 days of train data means each
   weekday appears twice, too sparse to learn.

LambdaRank and DIN-style history / multi-task features genuinely underperform on
this dataset, independent of implementation — the agent tried all three and
discarded all three. Knowing when to stop is as much the result as knowing what
to try.

## How we built it

Pure numpy for the scored agent path (`agent.py`, `baseline.py`,
`experiments/`) — no GPU, ~10 min per full run on a laptop CPU. `evaluate.py`
is imported unmodified. Each experiment runs in its own subprocess with a
30-min hard timeout, so a crash or hang in one idea can't take down the loop
or corrupt the champion state (verified with a real injected crash). Committed
receipts of a full run live in [`results/`](results/).

We also built a second, independent implementation
(`agent/` + `run_agent.py` + `solutions/runner.py`, documented in
[`AGENT.md`](AGENT.md)) with real Anthropic API wiring, live token
accounting, and a broader variant set (LightGBM LambdaRank, a torch
multi-task net, rank-ensembling). Not the scored path for this submission,
but a genuine second working system, later reconciled with the first —
`bpr_numpy`/`listwise_numpy`/`hour_of_day` are shared between both.

**Tools, APIs, libraries, datasets:**
- **Development tool:** Claude Code (Anthropic), used both to write the
  codebase and, in-session, to propose and implement the agent-authored
  experiments (`AUTHOR = 'agent'` in `experiments/*.py`) — the thing being
  scored as autonomous behavior, not just a coding assistant.
- **APIs:** the Anthropic API (`agent/llm.py`'s `LLMClient`) is wired into the
  second system for real, unattended LLM calls; not switched on for this
  submission's scored run (see Known limitations).
- **Libraries:** `numpy` only for the scored path; `pandas`, `scikit-learn`,
  `lightgbm`, `torch` for the second system (`requirements.txt`).
- **Dataset:** KuaiRand-Pure (Kuaishou / KuaiRand, via kuairand.com) — the
  required benchmark, no external data used anywhere.

Team: P1 pairwise BPR loss · P2 user-history / sequence features · P3 multi-task
learning · P4 the agent loop & reliability · P5 experiment tracking + this
write-up.

## Resource usage (Feasibility & Practicality)

Full detail in [`results/RESOURCES.md`](results/RESOURCES.md); summary:

| Metric | Value |
|---|---|
| GPU-hours | 0 (numpy/CPU only) |
| Agent wall-clock | ~478s (~8 min), of the 21,600s (6h) cap |
| Iterations used | 9, of the 50 cap |
| Convergence checkpoint | attempt 5 (champion `bpr_loss`, ε=0.002/N=3) |
| `agent.py` runtime LLM tokens | 0 (deterministic orchestrator, no LLM calls at runtime) |
| Claude Code authoring-session tokens | not separately metered — flat-rate Claude Pro subscription, not the pay-per-token API; reported as quota usage instead (~66% of a 5-hour session / ~58% of the weekly limit at time of writing) |

## Try it

```bash
pip install numpy
python agent.py --reveal-test-live
python dashboard.py --out dashboard.html
python make_final_submission.py --split test
python submit.py --check --split test submission.csv
```

## Known limitations

- Score gain is +0.004 test primary — real and past the ε threshold, but ~1.6%
  of the oracle headroom.
- Total LLM token usage for the Claude Code sessions that authored the
  agent-tagged experiments isn't separately metered (flat-rate subscription,
  not the pay-per-token API) — reported as quota-percentage instead; see
  `results/RESOURCES.md`.
