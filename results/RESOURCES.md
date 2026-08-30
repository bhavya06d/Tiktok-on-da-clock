# Resource Consumption — Feasibility & Practicality

Numbers for the submission's Feasibility & Practicality section. Everything
except LLM tokens is captured automatically in `agent_summary.json`; the token
count is filled in by hand (the agent has no visibility into the Claude Code
sessions that authored its experiments — see below).

## Canonical run (`results/agent_log.jsonl` + `results/agent_summary.json`)

| Metric | Value |
|---|---|
| Hardware | Laptop CPU, single machine, no GPU |
| **GPU hours** | **0** (numpy only) |
| **Agent wall-clock** | **~800 s (~13 min)** — sum of the 9 experiment run times |
| 6-hour wall-clock cap | not approached (800 s of 21 600 s) |
| **Iterations used** | **9** of the 50 cap |
| Convergence checkpoint | attempt **5**, champion `bpr_loss` (ε = 0.002, N = 3) |
| Failures recovered | 0 in this clean run (an earlier run demonstrated crash + timeout recovery) |
| Per-experiment timeout | 1800 s hard ceiling (subprocess-enforced) |

### Result

| | val primary | test primary | vs FM baseline (0.5946) |
|---|---|---|---|
| Scored champion (ε/N rule) — `bpr_loss` | 0.6037 | **0.5985** | **+0.0039** |
| Autonomous champion (`AUTHOR='agent'`) — `listwise_softmax` | 0.6039 | 0.5973 | +0.0027 |
| Best raw model (submitted) — `hour_of_day` | 0.6052 | 0.5986 | +0.0040 |

`hour_of_day` has the highest raw score but was *discarded* by the convergence
rule (only +0.0015 over `bpr_loss`, below ε = 0.002). `submission.csv` uses
`hour_of_day` as the best model actually achievable; the ε/N-scored checkpoint
is `bpr_loss`. They differ by 0.0001 on test.

## LLM tokens

**`agent.py` spends 0 tokens at runtime** — it is a deterministic orchestrator
(discover → run in subprocess → score → keep/discard → converge → log). No LLM
calls. Confirmed: `grep -i 'anthropic\|openai\|messages.create' agent.py` → nothing.

All token cost is in the **Claude Code sessions** that proposed and wrote the
`AUTHOR = 'agent'` experiments (`listwise_softmax`, `hour_of_day`,
`bpr_hard_negative`, `bpr_hard_negative_warmstart`, `seq_dur_drift`).

> **TODO before submission — fill this in.** In Claude Code run `/cost`, and/or
> open Anthropic Console → *Usage* for the project's working dates. Sum input +
> output across the authoring sessions. Then either re-run
> `python agent.py --reveal-test-live --llm-tokens-in <X> --llm-tokens-out <Y>`
> (fast — parsed data is cached), or edit `agent_summary.json`'s
> `llm_tokens_input` / `llm_tokens_output` / `total_llm_tokens` and this table:

| Metric | Value |
|---|---|
| `agent.py` runtime tokens | **0** (deterministic, no LLM) |
| Claude Code authoring sessions | `<N>` |
| Model | `<claude-... — check /model in Claude Code>` |
| Input tokens (authoring) | `<~X>` |
| Output tokens (authoring) | `<~Y>` |
| **Total LLM tokens** | `<~X + Y>` |
| Estimated cost | `<$Z>` |
