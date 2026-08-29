# KuaiRand Autonomous ML Research Agent

An LLM-driven agent that autonomously iterates on a KuaiRand-Pure ranking pipeline:
propose → write code → run → evaluate → reflect → repeat, until convergence
(ε = 0.002, N = 3, cap 50 iterations / 6h). Scored on validation-best checkpoint,
evaluated once on the hidden test set.

Official baseline (FM, k=16): val primary 0.6016 / hidden-test 0.5946.
Sanity rungs: random 0.4753, item popularity 0.5715. Ceiling ~0.8645.

## Architecture

```
run_agent.py
   │
   ▼
agent/orchestrator.py      the loop: state, convergence, best-checkpoint tracking   (Person 1)
   ├── agent/llm.py        LLM client + token accounting                            (Person 2)
   ├── agent/prompts.py    system prompt, idea bank, history compression            (Person 2)
   ├── agent/executor.py   sandboxed subprocess run, timeout, error capture         (Person 3)
   └── agent/run_logger.py per-iteration JSONL log (hypothesis/diff/metrics/errors) (Person 3)

workspace/solution.py      THE file the agent rewrites each iteration              (Person 4 seeds it)
ml/                        human-written reference implementations for the idea bank (Person 4)
scripts/analyze_runs.py    plots, results table, intervention count                 (Person 5)
```

Contract: `workspace/solution.py` must accept `--split val|test` and print exactly one
line `METRICS_JSON: {"gauc": ..., "ndcg5": ..., "primary": ...}` to stdout, and write
`workspace/submission_<split>.csv` in the starter-kit schema. Everything else is free.

## Team split

| Person | Owns | Definition of done |
|---|---|---|
| 1 | Orchestrator | Full loop runs end-to-end with a mock LLM; convergence + rollback work |
| 2 | LLM & prompts | Agent produces valid code ≥90% of iterations; tokens counted & logged |
| 3 | Executor & logging | Crashes/timeouts never kill the run; JSONL log matches deliverable spec |
| 4 | ML domain | Baseline reproduced (±0.001 of 0.6016 val); LightGBM ranker beats it; feature lib ready |
| 5 | Analysis & submission | Results table, run-log summary, README, Devpost text, 3-min video |

## Week plan

1. **Days 1–2**: P4 reproduces baseline + eval self-check (random/popularity rungs).
   P1+P3 get the loop running with a *mock* LLM that just returns the baseline code.
2. **Days 3–4**: P2 wires the real LLM + idea bank. First real autonomous run.
   P4 confirms LightGBM LambdaRank reference beats baseline (this de-risks everything).
3. **Days 5–6**: Full runs. P3 hardens recovery (inject failures on purpose, show recovery
   in logs). P5 starts report + video from real run logs.
4. **Final day**: Best full run = the submission. Freeze, write up, record video.

## Run

```
export ANTHROPIC_API_KEY=...   # or set OPENAI_API_KEY and swap the client in agent/llm.py
python run_agent.py --data-dir ./data --max-iters 50
```

## Rules we must not break

- Train on KuaiRand data only. Side-feature files that ship with KuaiRand are allowed;
  any external dataset is not.
- Develop on train + validation only; hidden test is scored once on the final submission.
- Log every iteration: hypothesis, code diff, metrics, error/recovery events.
- Report: manual-intervention count (target 0), total LLM tokens, wall-clock, iterations used.
