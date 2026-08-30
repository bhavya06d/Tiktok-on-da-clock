# Devpost submission — KuaiRand Autonomous ML Research Agent

## Inspiration

Most entries to a "beat the baseline" task hand-tune a model and report a
number. The KuaiRand-Pure starter kit itself warns that this baseline is hard
to move: it already eats ~30% of the usable range between random and a perfect
oracle, and every feature/capacity tweak the organizers tried failed. So we
built the thing that's actually interesting to watch — an **autonomous research
loop** that proposes a hypothesis, writes the code, runs the official scorer,
decides keep-or-discard, and moves on, logging every step.

## What it does

`python run_agent.py` runs unattended. Each iteration the agent:

1. **proposes** the next idea from a queue drawn from the README's own
   "headroom" list — ranking loss, user-history features, multi-task learning,
   watch-time censored regression, pairwise/listwise BPR, rank-ensembling —
   each with a written hypothesis citing the method it draws on
   (LambdaRank, MMoE/ESMM, CWM);
2. **rewrites** `solution.py` to implement it;
3. **runs** it in a sandboxed subprocess (timeout + traceback capture);
4. **scores** it with the unmodified `evaluate.py`;
5. **decides** keep or discard against the current best;
6. **stops** when the dataset's own convergence rule fires — ε = 0.002, N = 3.

A live dashboard shows the score-over-time trajectory against the oracle
ceiling, every hypothesis → diff → score → decision, and the recovery events.

## How we frame results

Against the **oracle ceiling (0.8484 val / 0.8645 test)**, not 1.0. The FM
baseline already sits at ~31% of that headroom, so "0.60 vs a perfect 1.0" is
the wrong mental model.

## What we found

The organizers' hypothesis that a ranking loss would be the biggest win **did
hold** — a listwise softmax loss on the same FM architecture beat the
pointwise baseline (val 0.6039 vs 0.6015, test 0.5973 vs 0.5946), and the
agent correctly kept it. That result took two tries: a more elaborate torch
implementation (warm-started, K=24 list sampling) scored only 0.565 with
default params, which could easily read as "the idea failed" — but the same
idea implemented simply, in ~40 lines of plain numpy, is the run's best
result. LambdaRank and DIN-style history features did genuinely underperform
here, independent of implementation. Rank-ensembling (FM ⊕ LambdaRank) is a
real but sub-ε gain, correctly treated as convergence rather than a win. That
discipline — trying real ideas, keeping only what clears the bar, catching
"idea vs. implementation" before writing off a hypothesis, stopping on the
stated rule, zero human intervention — is the submission.

## How we built it

Python, `numpy / pandas / scikit-learn / lightgbm / torch`. The agent
(`agent/`) is model-agnostic; the brain is swappable between a deterministic
offline planner (runs with no API key, for the live demo) and a real Claude
client with token accounting. Leakage was a real bug we hit and fixed —
target encodings are now leave-one-out on the train split.

## Try it

```
pip install -r requirements.txt     # see requirements.txt for a macOS libomp note
python run_agent.py                 # the loop
python scripts/analyze_runs.py      # the results table
```
