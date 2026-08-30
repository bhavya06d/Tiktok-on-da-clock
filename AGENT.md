# Agent systems — index

The canonical autonomous agent, its results, and how to reproduce them are in
the **top-level [README.md](README.md)** and the committed receipts in
**[results/](results/)** (`agent_log.jsonl`, `agent_summary.json`,
`dashboard.html`). This file just maps the two agent code paths in the repo.

## 1. `agent.py` + `experiments/` — the scored system

The one judges should run. `agent.py` auto-discovers every `experiments/*.py`,
runs each in an isolated subprocess (30-min timeout, full traceback capture),
keeps/discards on validation primary vs the running champion, applies the
ε = 0.002 / N = 3 convergence rule, and logs every attempt (hypothesis + full
source + metrics + decision) to `agent_log.jsonl`. It makes **no LLM calls** —
the `AUTHOR = 'agent'` experiment files were written by Claude in separate
Claude Code sessions (that's where the tokens are; see
[results/RESOURCES.md](results/RESOURCES.md)).

```bash
python agent.py --reveal-test-live
```

**Canonical run:** converged at attempt 5. Scored champion `bpr_loss`
(val 0.6037 / test 0.5985, +0.0039 vs the 0.5946 FM baseline); autonomous
champion `listwise_softmax` (test 0.5973); best raw model `hour_of_day`
(test 0.5986, the one in `submission.csv`). Full table in the README.

## 2. `agent/` + `run_agent.py` + `solutions/` — earlier self-contained variant

A separate, more end-to-end orchestrator: it rewrites a single `solution.py`
each iteration (subprocess execution, timeout, rollback to the last good
version on any crash) and can be driven by a real Claude client
(`agent/llm.py`) or a deterministic offline planner. Kept for that
subprocess-isolation + rollback design and because its demo run
`runs/demo/iterations.jsonl` includes a recovered injected failure. Not the
scored path.

```bash
python run_agent.py                 # offline planner (0 tokens)
python run_agent.py --llm anthropic # real Claude loop (needs ANTHROPIC_API_KEY)
```
