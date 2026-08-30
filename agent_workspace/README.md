# agent_workspace — Autonomous ML Research Agent (Person 4)

The propose → modify → train → evaluate → decide loop, plus failure recovery
and the official convergence rule, for the KuaiRand-Pure within-user ranking
task. **Everything here is self-contained** — it reads the dataset and (read
only) the repo's `evaluate.py`, and writes nothing outside this folder.

```
agent_workspace/
├── run_agent.py        the orchestrator + reliability loop  (Person 4)
├── solution.py         the ML pipeline + 4 toggleable strategies
├── solution_backup.py  last KEPT version (auto-managed; rollback target)
├── logs/
│   ├── iterations.jsonl structured record per iteration  (judges read this)
│   ├── agent.log        human-readable running commentary (tail -f this)
│   └── summary.json     final champion + autonomy stats
└── .cache/             parsed dataset cache (auto-built on first run)
```

## Prerequisites

- Python 3.9+ and **numpy only** (no torch / pandas / scikit-learn).
  ```bash
  pip install numpy
  ```
- The dataset at `../KuaiRand-Pure/data/` (repo root). If it's elsewhere:
  `export KUAIRAND_DATA=/abs/path/to/KuaiRand-Pure/data`.

## Run it — one command

```bash
cd agent_workspace
python run_agent.py
```

That's it. The loop runs unattended: baseline → Person 1 (BPR) → Person 2
(history features) → Person 3 (multi-task) → hyper-parameter perturbations of
whichever is winning, stopping automatically on the official rule
(ε = 0.002, N = 3 consecutive non-improving iterations) or the 50-iteration /
6-hour ceilings. Expect ~15–25 min on a laptop CPU; the first iteration pays a
one-time ~30 s dataset-parse cost, then it's cached.

Optional flags:

```bash
python run_agent.py --max-iters 8          # shorter run
python run_agent.py --eps 0.001 --patience 5
python run_agent.py --step-timeout 900     # tighter per-iteration hang guard
python run_agent.py --demo-fault 2         # inject a crash at iteration 2 to
                                           # SHOW the recovery path in the logs
                                           # (the real queue item runs at 3;
                                           #  nothing is skipped)
```

## Watch it work

```bash
# live commentary (Git Bash / WSL / macOS / Linux)
tail -f logs/agent.log

# live commentary (PowerShell)
Get-Content logs/agent.log -Wait

# pretty-print the structured log so far
python -c "import json;[print(json.dumps(json.loads(l),indent=2)) for l in open('logs/iterations.jsonl')]"

# final result
cat logs/summary.json
```

Each `logs/iterations.jsonl` line contains: `iteration`, `timestamp`,
`strategy`, `hyperparams`, `hypothesis`, `config_diff` (the exact toggle
applied), `primary` / `gauc` / `ndcg5`, `duration_sec`, `decision`
(`KEEP` / `DISCARD` / `ERROR` / `TIMEOUT`), `peak_primary`,
`consecutive_no_improve`, and `error` + `stderr_tail` on any failure.

## Test one strategy by hand (no loop)

```bash
python solution.py --strategy bpr        # prints: PRIMARY_SCORE: 0.XXXX
python solution.py --strategy multitask
```

## The four strategies (`solution.py`)

| `STRATEGY` | Owner | Idea |
|---|---|---|
| `baseline` | — | Official pointwise-logloss FM (k=16). The anchor. |
| `bpr` | Person 1 (Shaun) | Pairwise BPR loss over within-user positive/negative impression pairs — training objective matched to the ranking metric. |
| `history` | Person 2 (Bhavika) | Train-only user-history fields that vary within a user's impressions: has-long-viewed-this-author-before + author long-view-rate bucket. |
| `multitask` | Person 3 | Shared FM embedding table + auxiliary `is_click` / `is_like` / `is_forward` heads regularising the `long_view` head. |

`run_agent.py` toggles between them by rewriting two lines in `solution.py`
(`STRATEGY = ...` and `HYPERPARAMS = ...`) — that rewrite is the "code diff"
the agent applies and logs.

## Reliability guarantees (Person 4)

| Failure mode | Handling |
|---|---|
| Syntax / import / non-zero exit | subprocess `returncode != 0` → roll `solution.py` back to `solution_backup.py` |
| Runtime exception / OOM | subprocess dies; full stderr traceback captured to `iterations.jsonl` before rollback |
| Infinite loop / hang | hard 30-min per-iteration subprocess timeout → forced kill → rollback |
| No `PRIMARY_SCORE` emitted | classified as an error → rollback |
| Score regression / stall | new primary must beat the running peak by > ε or it's `DISCARD` + instant rollback |
| Missing official `evaluate.py` | `solution.py` falls back to a bundled byte-identical scorer mirror |

`solution_backup.py` always holds the last KEPT (best-scoring) version, so the
loop can never corrupt its own state.

## Isolation

This folder never imports, reads-for-write, or modifies any teammate file. The
only outside dependency is a **read-only** `from evaluate import evaluate`
(the fixed official scorer, which nobody edits) — with a bundled fallback if
even that is unavailable.
