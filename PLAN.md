# Team Plan — Autonomous ML Research Agent (KuaiRand)

TikTok TechJam 2026, Track 2. Starter kit setup is done — FM baseline confirmed
working locally (test primary 0.5953, matches README's 0.5946).

## READ THIS FIRST — correction after reading the official problem statement

Everything below this section was written before we had the actual official
problem doc (bit.ly/TikTokTechJam2026Info). One thing in the original plan
was wrong and matters a lot:

**The agent itself must propose and write the iteration code. A human
hand-writing an idea (however good) does not count as "the agent" — it
counts as manual intervention, which is directly graded (Autonomy, under
Impact & Relevance).** Direct quote from the official doc: *"Writing the
code for each stage is part of the agent's job, not something provided in
advance."*

This does **not** mean Persons 1-3's hand-implemented ideas (BPR loss,
history, multitask) are wasted — they're still real, valuable, tested code.
It means: keep building them, but don't present them as "the agent's"
output. `experiments/README.md` now has an `AUTHOR = 'human' | 'agent'`
convention and `agent.py` tracks two separate champions (overall best vs.
autonomous-only best) so this never gets blurred in the log. Read
`experiments/README.md`'s "Why AUTHOR matters" section before adding a new
experiment file.

Other confirmed facts from the official doc worth knowing:
- **Metric/baseline/convergence rule are all exactly what we already had** —
  GAUC/nDCG@5, primary=mean, baseline 0.5946, eps=0.002/N=3. No changes there.
- **Hard caps:** 50 iterations OR 6h wall-clock per run, whichever first.
- **No external training data** — KuaiRand only, no augmenting with other
  datasets. Open-source libraries, papers, and pretrained weights are fine.
- **Deliverables we hadn't planned for:** each log entry needs the actual
  code (not just hypothesis + score) — `agent.py` now captures this
  automatically. Final submission also needs total LLM token usage, total
  wall-clock, and iteration count reported (Feasibility & Practicality
  scoring) — nobody owns instrumenting this yet.
- **No paid API needed.** The doc explicitly allows using Claude Code (or
  Trae, ByteDance's tool, free 7-day trial) as the coding agent — we're
  already doing this live in-session, no extra signup or spend.

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

## 5 People, 5 Features

Everyone has their own Claude to build with, so the split is one concrete,
independent feature per person — not specialized "roles" that block on each
other. Build in parallel against the verified baseline (0.5946, already
confirmed working — don't re-derive it), then integrate.

| # | Feature | Task | Deliverable |
|---|---|---|---|
| 1 | **Pairwise Ranking Loss (BPR)** | Change training from "predict long_view yes/no per video" to "video A ranks above video B" for pairs within the same user. README's top-ranked idea — smallest change, most likely direct win. | Working variant, before/after score. |
| 2 | **User History / Sequence Features** | Pull in each user's past interactions from train — currently completely unused. Start simple: average embeddings of last N watched videos as an extra feature. Stretch: attention over history (DIN-style). | Working variant, before/after score. |
| 3 | **Multi-Task Learning** | Logs have `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms` — all unused. Train the model to predict several of these alongside `long_view` (shared embeddings, multiple output heads). | Working variant, before/after score. |
| 4 | **The Agent Loop** | Build the orchestrator: picks an idea, edits code (or toggles Person 1-3's implementations), runs `baseline.py`, reads the score, decides keep/discard using the README's own convergence rule (eps=0.002, N=3), logs everything. This is what makes it an *agent*, not three people hand-tuning a model. | A loop that runs unattended for multiple iterations and produces a log. |
| 5 | **Experiment Tracking + Demo** | Build the thing that makes the other four legible: a log/dashboard of every iteration (hypothesis -> code diff -> score -> decision), a score-over-time chart, the oracle-ceiling framing (0.5946 -> 0.8645 max) baked into reporting. Also stitches everyone's pieces into one working repo before the deadline and runs the live demo. | The thing judges actually look at. |

**Optional 6th idea (stretch, no dedicated owner):** watch-time censored
regression — model actual watch duration with a one-sided loss (README flags
this as the "some research depth" option, see CWM for reference). Only tackle
if someone finishes early.

**Already ruled out by organizers, don't retry:** adding more static features,
increasing embedding dimension (k=8/16/32) — neither moved the score.

**Reminder:** pure user-side features contribute nothing on their own (ranking
is within-user, so a per-user constant doesn't change order) — only useful
crossed with item-side features. Relevant for Person 3's multi-task heads.

## Timeline

1. **Now -> +25%:** Everyone builds their feature in parallel against the
   verified baseline. Person 4 builds the loop skeleton against a stub/mock
   experiment while waiting for 1-3 to have something callable.
2. **+25% -> +60%:** Person 4 wires in Persons 1-3's implementations as
   callable experiments; Person 5's logging comes online; run the loop for
   real, even with only 1-2 features wired up.
3. **+60% -> +85%:** Let the agent run longer iterations unattended — this is
   demo footage (screen-record it working). Fix whatever the real run exposes.
4. **+85% -> done:** Freeze code, polish the dashboard/log output, rehearse the
   pitch: lead with the agent's reasoning trace, not just the final score.
