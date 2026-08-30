# KuaiRand Autonomous ML Research Agent

An LLM-style agent that improves the KuaiRand-Pure within-user ranking pipeline
on its own: **propose a hypothesis → write `solution.py` → run it → read the
score → keep or discard → reflect → repeat**, until the dataset's own
convergence rule fires (ε = 0.002, N = 3) or the iteration cap is hit. Every
step is logged so a judge can watch it reason, not just see a final number.

Progress is measured against the **oracle ceiling (val 0.8484 / test 0.8645)**,
not against 1.0 — the FM baseline already consumes ~30% of the usable range, so
the real headroom is ~0.25, not ~0.40.

```
python run_agent.py                 # offline planner — no API key needed
python run_agent.py --llm anthropic # real Claude loop (needs ANTHROPIC_API_KEY)
python run_agent.py --inject-fault 2 # demo the crash-recovery path
python scripts/analyze_runs.py      # results table + autonomy summary
python runs_to_dashboard_log.py runs/demo/iterations.jsonl && \
python dashboard.py --log runs_agent_log.jsonl --summary runs_agent_summary.json \
                    --out runs_dashboard.html   # same dashboard Person 5 built, this system's data
python scripts/publish_artifact.py  # snapshot a run to one shareable HTML
```

Published snapshot of the demo run:
<https://claude.ai/code/artifact/40273dad-5298-4cb4-bfa9-684068872da4>

**Setup:** `pip install -r requirements.txt` (numpy/pandas/scikit-learn/lightgbm/torch;
`anthropic` only needed for `--llm anthropic`). On macOS, lightgbm also needs
`brew install libomp` (not a pip package) - see requirements.txt.

## How it works

| Piece | File | Responsibility |
|---|---|---|
| Orchestrator | `agent/orchestrator.py` | the loop; ε/N convergence; best-checkpoint tracking; rollback to last-good `solution.py` on any crash |
| Executor | `agent/executor.py` | runs `solution.py` in a subprocess with a timeout; captures the traceback and feeds it back to the next iteration |
| Run logger | `agent/run_logger.py` | `runs/<name>/iterations.jsonl` — hypothesis, code diff, metrics, decision, recovery events, final summary |
| Brain | `agent/llm.py` | `OfflinePlanner` (deterministic idea queue, runs with zero external deps) **and** `AnthropicLLM` (real Claude, full token accounting); auto-selected on `ANTHROPIC_API_KEY` |
| Prompts | `agent/prompts.py` | task brief + idea bank citing real methods (LambdaRank, MMoE/ESMM, CWM duration-bias) + compressed history |
| Variants | `solutions/runner.py` | every idea, one function; all honour the `solution.py` contract and call the **unmodified** `evaluate.py` |
| Features | `ml/features.py` | shared, leakage-guarded feature engineering (leave-one-out target encoding, temporal, side-features, behavioural history) |
| Dashboard | `dashboard.py` (Person 5) | renders `agent_log.jsonl`/`agent_summary.json` to a static HTML report. This system's own log format (`runs/<name>/iterations.jsonl`) is different - bridge it first with `runs_to_dashboard_log.py` (below), then point `dashboard.py` at the converted files |

### The contract every `solution.py` honours
- accepts `--split val|test` and `--data-dir <path>`
- trains only on the train split (+ KuaiRand side-feature files)
- prints exactly `METRICS_JSON: {"gauc": .., "ndcg5": .., "primary": ..}`
- writes `submission_<split>.csv` (`row_id,user_id,video_id,score`) in official order
- `evaluate.py` is imported and never edited — that's what keeps the numbers trustworthy

## Ideas the agent explores (the README's own "headroom" list)

| # | Method | Variant | Draws on |
|---|---|---|---|
| 0 | Pointwise FM baseline (anchor) | `fm` | starter kit |
| 1 | Ranking loss — LightGBM **LambdaRank**, groups = per-user impressions | `lambdarank` | idea #1 |
| 2 | **User-history / sequence** features (DIN/SIM-lite): expanding & rolling long_view rate, watch-ratio, author affinity | `history` | idea #2 |
| 3 | **Multi-task** shared-embedding net, aux heads on click / like / watch (MMoE/ESMM) | `multitask` | idea #3 |
| 4 | **Watch-time censored regression** — one-sided loss on completed plays | `censored` | idea #4 (CWM) |
| 5 | Pairwise/listwise **BPR / ListNet** on the FM architecture, torch, warm-started | `bpr` | idea #1 |
| 5b | Same idea, simpler: numpy-only pairwise BPR on the FM architecture | `bpr_numpy` | idea #1 |
| 5c | Same idea again: numpy-only **listwise softmax** (1 positive vs M random negatives) | `listwise_numpy` | idea #1 |
| 6 | **Ensemble** on per-user ranks: FM seed-ensemble + LambdaRank blend | `combined` | idea #6 |
| 7 | **Hour-of-day** feature (day-parting) on the listwise architecture | `hour_of_day` | time features (README headroom) |

Three variants for the same idea (5/5b/5c) is deliberate, not redundancy: see
Results below - the simpler numpy version is the one that actually works.

Run any one directly:

```
python -m solutions.runner --variant lambdarank --split val
```

## Results

Demo run (`runs/demo/`, offline planner, `--inject-fault 3`), regenerate with
`python run_agent.py --max-iters 14 --min-iters 7 --inject-fault 3 --run-name demo && python scripts/analyze_runs.py runs/demo/iterations.jsonl`:

| iter | method | val primary | GAUC | nDCG@5 | Δ baseline | decision |
|---:|---|---:|---:|---:|---:|---|
| 0 | fm (baseline anchor) | 0.6015 | 0.6671 | 0.5358 | −0.0001 | **KEEP** |
| 1 | listwise_numpy | 0.6039 | 0.6707 | 0.5372 | +0.0023 | **KEEP** |
| 2 | hour_of_day | 0.6052 | 0.6719 | 0.5385 | +0.0036 | **KEEP (best)** |
| 3 | fault-injection | — crash — | | | | rolled back to best |
| 4 | lambdarank | 0.5903 | 0.6519 | 0.5288 | −0.0113 | discard |
| 5 | history (DIN-lite) | 0.5811 | 0.6397 | 0.5225 | −0.0205 | discard |
| 6 | multitask (MMoE) | 0.5880 | 0.6486 | 0.5274 | −0.0136 | discard → **converged** |

**Best: iter 2 `hour_of_day`, val primary 0.6052 / test primary 0.5986**
(test measured directly, not inferred - see note below) - the best score in
either harness, beating even the hand-tuned `bpr_numpy` (0.6037 val). 7
iterations, **1 crash recovered, 0 manual interventions**, 6.4 min
wall-clock, ε=0.002/N=3 convergence fired on its own.

**Findings.**
1. *(Corrected from an earlier version of this doc.)* A ranking loss *does*
   beat pointwise logloss here - the original claim that it didn't was an
   artifact of one implementation, not the idea itself. The `bpr` variant
   (torch, warm-started, K=24 list-sampling) scored only 0.565 with default
   params - worse than baseline. The same idea in ~40 lines of plain numpy
   (`listwise_numpy`, no warm-start, M=4 random negatives) scores 0.6039, a
   real win. Lesson: when a well-motivated idea underperforms, check whether
   the idea is wrong or the implementation is, before discarding it.
2. Every idea up to this point only changed the loss function - nothing had
   touched what the model actually sees. `hour_of_day` is the first to: one
   new categorical field (hour-of-day, from the raw `hourmin` column),
   everything else unchanged, real +0.0013 gain on top of `listwise_numpy`.
   A same-day follow-up (day-of-week, not committed as a separate file - see
   its docstring) added nothing further (0.6051, noise-level) - with only 14
   days of train data, each weekday is seen twice, too sparse to learn from.
   Rank-ensembling `hour_of_day` with `lambdarank` was also tried and hurt
   slightly (0.6052 -> 0.6048 as blend weight shifted toward lambdarank) -
   consistent with lambdarank being too far behind (0.5903) to add value via
   rank-averaging with a much stronger model.
3. LambdaRank and DIN-style history/multitask features genuinely underperform
   here, independent of implementation - not every well-cited method
   transfers to every dataset, and knowing when to stop is as much the
   result as knowing what to try.

*Test-primary note:* solution.py's contract only ever prints val metrics for
`--split test` (the official test split has no labels available to the
agent, by design - see the contract above). 0.5986 for `hour_of_day` (and
0.5973 for `listwise_numpy`) were confirmed separately by direct evaluation
with a labeled local test split (`agent.py`'s parallel harness, same
code/config, seed=0) - not numbers the agent itself could have seen or used
to decide anything.

## Robustness / autonomy evidence

- Crashes and timeouts are caught by the executor, logged as `event`s, and the
  traceback is handed to the next iteration's prompt so the agent can react.
  On failure the orchestrator rolls `solution.py` back to the last good version.
- `runs/demo/iterations.jsonl` includes at least one real recovered failure
  (see the `rolled_back_to_best` events).
- The final summary line reports: iterations used, failures recovered,
  **manual interventions (0)**, wall-clock, and LLM tokens.
