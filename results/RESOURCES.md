# Resource Consumption — Feasibility & Practicality

Required for the submission's Feasibility & Practicality section. Wall-clock,
iteration count and GPU hours are captured automatically in
`results/agent_summary.json`; LLM token usage is filled in by hand below because
the agent has no visibility into its own Claude Code sessions.

## Compute

| Metric | Value | Source |
|---|---|---|
| Hardware | Laptop CPU, single machine | — |
| GPU hours | **0** | numpy-only, no GPU (`agent_summary.json: gpu_hours`) |
| Agent wall-clock (one full run) | see `agent_summary.json: total_wall_clock_seconds` | `agent.py` sums per-experiment run time |
| Iterations used | see `agent_summary.json: attempts` (of 50 cap) | `agent.py` |
| 6-hour wall-clock cap | not approached | `agent_summary.json: wall_clock_cap_seconds = 21600` |
| Convergence checkpoint | `agent_summary.json: converged_at` | ε = 0.002, N = 3 |

## LLM tokens

The agent-authored experiments (`AUTHOR = 'agent'` in `experiments/`:
`listwise_softmax`, `hour_of_day`, `bpr_hard_negative`,
`bpr_hard_negative_warmstart`, `seq_dur_drift`) were proposed and written by
Claude across interactive Claude Code sessions. `agent.py` itself runs offline
and spends **0 tokens** — all token cost is in those authoring sessions.

> **TODO before submission — one of the team fills this in.** Open the Claude
> Code usage panel (or the Anthropic Console usage dashboard) for this project,
> sum input + output tokens across the sessions that produced the
> `AUTHOR = 'agent'` experiment files, and replace the placeholders:

| Metric | Value |
|---|---|
| Claude Code sessions (agent authoring) | `<N>` |
| Total input tokens | `<~X>` |
| Total output tokens | `<~Y>` |
| **Total tokens** | `<~X + Y>` |
| Model | `claude-*` (Claude Code default) |
| Estimated cost | `<$Z>` |

After filling this in, also set `total_llm_tokens` in
`results/agent_summary.json` to the same total so the two agree.
