# Team Plan — Autonomous ML Research Agent (KuaiRand)

TikTok TechJam 2026, Track 2. Starter kit setup is done — FM baseline confirmed
working locally (test primary 0.5953, matches README's 0.5946).

## The Angle That Stands Out

Most teams will hand-tune a model and report "we got a better score." That's not
what's judged. Our differentiator: build a visible, autonomous research loop — an
agent that proposes a hypothesis, writes/edits code, runs `evaluate.py`, reads the
result, decides keep-or-discard, and moves to the next idea — with every step
logged so judges can watch it think, not just see a final number.

We also frame results against the **oracle ceiling (0.8645)**, not against 1.0 —
per the README, the baseline already eats 30.7% of the usable range, so real
headroom is 0.27, not 0.41. Reporting it this way shows we understood the problem,
not just the number.

## The Agent Loop

```
1. PROPOSE  -> pick next idea from a queue (pairwise loss, sequence features, multi-task, ...)
2. IMPLEMENT -> edit data.py/baseline.py to realize it (LLM-generated code change)
3. EVALUATE -> run baseline.py, get valid GAUC/nDCG/primary
4. DECIDE   -> beats current best by > eps (0.002, per README's own convergence rule)? keep : discard, log why
5. REPEAT   -> until N=3 iterations show no improvement (README's own stopping rule) or idea queue empty
```

Using the README's own eps/N convergence rule (eps=0.002, N=3) for the agent's
stopping logic shows the agent respects the problem's stated rigor instead of
inventing an arbitrary threshold.

## Idea Queue (from README's "headroom" section, ranked by likely payoff)

1. **Pairwise/listwise loss (BPR or per-user softmax)** instead of pointwise
   logloss — the eval metrics are ranking metrics, so this should be the most
   direct win. Not yet tried by organizers.
2. **User history / sequence modeling** — every user has up to hundreds of
   interactions in train, currently completely unused. DIN/SIM-style interest
   modeling is a blank space.
3. **Multi-task learning** — `is_click`, `is_like`, `is_follow`, `is_comment`,
   `is_forward`, `play_time_ms` are all sitting in the logs unused as auxiliary
   signals for the `long_view` main task.
4. **Watch-time censored regression** — model actual watch duration with a
   one-sided loss (a video that plays to completion has a censored/unknown
   "would-have-watched-longer" duration). Research-heavy, see CWM for reference.
5. Already ruled out by organizers, don't retry: adding more static features,
   increasing embedding dimension (k=8/16/32) — neither moved the score.
6. Reminder: pure user-side features contribute nothing on their own (ranking
   is within-user, so a per-user constant doesn't change order) — only useful
   crossed with item-side features.

## Guardrail

Don't touch `evaluate.py` — it's the fixed scoring contract. Using it unmodified
is what makes reported numbers trustworthy to judges who know this dataset.

## Roles (5 people)

| # | Role | Owns |
|---|---|---|
| 1 | **Agent Orchestrator** | The loop itself: hypothesis queue, LLM calls to propose/write code edits, invoking eval, keep/discard/convergence logic, iteration log. |
| 2 | **ML Builder A — Loss/Ranking** | Reference implementation of pairwise (BPR) loss in `baseline.py`, exposed as a parametrized experiment the agent can call/toggle. |
| 3 | **ML Builder B — Sequence/History** | Reference implementation using each user's train-set interaction history (simple version: recent-N item embeddings averaged in; stretch: attention/DIN-style). |
| 4 | **Infra/Eval/Logging** | Experiment tracker (every iteration -> hypothesis, diff, scores, decision, timing), score-over-iterations chart, oracle-ceiling framing baked into reporting. |
| 5 | **Integration + Demo/Pitch** | Runs the whole pipeline end-to-end early and often, catches breakage between pieces, owns submission write-up and live demo script, floats to unblock. |

## Timeline

1. **Now -> +25%:** Everyone works in parallel against the verified baseline
   (0.5946, already confirmed — don't re-derive it). Builders 2 & 3 work directly
   in `baseline.py` behind flags/functions the orchestrator can call.
2. **+25% -> +60%:** Orchestrator integrates Builders 2 & 3's work as callable
   experiments; Infra's logging comes online; run the loop for real, even with
   only 2 ideas wired up.
3. **+60% -> +85%:** Let the agent run longer iterations unattended — this is
   demo footage (screen-record it working). Fix whatever the real run exposes.
4. **+85% -> done:** Freeze code, polish the dashboard/log output, rehearse the
   pitch: lead with the agent's reasoning trace, not just the final score.
