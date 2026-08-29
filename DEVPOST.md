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

The organizers' hypothesis that a ranking loss would be the biggest win did not
hold in our experiments (LambdaRank, listwise/BPR on the FM architecture, and
DIN-style history features all landed **below** the well-tuned pointwise FM).
The agent correctly discarded them. The one gain it kept was rank-ensembling
(FM seed-ensemble + LambdaRank blend), a marginal but real improvement — and by
the dataset's own ε = 0.002 rule the agent then declared convergence rather than
chasing noise. That discipline — trying six real ideas, keeping only what beats
the bar, stopping on the stated rule, zero human intervention — is the
submission.

## How we built it

Python, `numpy / pandas / scikit-learn / lightgbm / torch`. The agent
(`agent/`) is model-agnostic; the brain is swappable between a deterministic
offline planner (runs with no API key, for the live demo) and a real Claude
client with token accounting. Leakage was a real bug we hit and fixed —
target encodings are now leave-one-out on the train split.

## Try it

```
python run_agent.py                 # the loop
python dashboard/server.py          # watch it
python scripts/analyze_runs.py      # the results table
```
