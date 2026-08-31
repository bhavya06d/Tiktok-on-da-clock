# Resource Consumption: Feasibility & Practicality

Numbers for the submission's Feasibility & Practicality section. Everything
except LLM tokens is captured automatically in `agent_summary.json`; the token
count is filled in by hand (the agent has no visibility into the Claude Code
sessions that authored its experiments: see below).

## Canonical run (`results/agent_log.jsonl` + `results/agent_summary.json`)

| Metric | Value |
|---|---|
| Hardware | Laptop CPU, single machine, no GPU |
| **GPU hours** | **0** (numpy only) |
| **Agent wall-clock** | **482.4 s (~8 min)**: sum of the 9 experiment run times |
| 6-hour wall-clock cap | not approached (482 s of 21 600 s) |
| **Iterations used** | **9** of the 50 cap |
| Convergence checkpoint | attempt **5**, official checkpoint `listwise_softmax` (ε = 0.002, N = 3) |
| Failures recovered | 0 in this clean run (an earlier run demonstrated crash + timeout recovery) |
| Per-experiment timeout | 1800 s hard ceiling (subprocess-enforced) |

### Result

The rule separates *when to stop* (ε/N, the patience rule below) from
*which checkpoint is submitted* ("the validation-best checkpoint at that
point": plain best-so-far, no ε filter). Bounded to attempts up through
the convergence trigger (attempt 5 of 9):

| | val primary | test primary | vs FM baseline (0.5946) |
|---|---|---|---|
| **Official checkpoint: `listwise_softmax`** (submitted) | **0.6039** | **0.5973** | **+0.0027** |
| Also tried within that window, not selected: `bpr_loss` (human) | 0.6037 | 0.5985 | +0.0039 |

Found after convergence, real but not the scored checkpoint under the
rule's "at that point" wording (the loop keeps running remaining
experiment files so nothing gets silently dropped from the log, but
nothing found this way is eligible to be submitted):

| | val primary | test primary | vs FM baseline (0.5946) |
|---|---|---|---|
| `hour_of_day`: highest raw score anywhere in the run | 0.6052 | 0.5986 | +0.0040 |
| `user_history` (human, 2nd human idea, also post-convergence) | 0.6010 | 0.5953 | +0.0007 |

`submission.csv` is trained from `listwise_softmax`'s exact configuration
(`make_final_submission.py`), matching the official checkpoint above, not
`hour_of_day`.

## LLM tokens

**`agent.py` spends 0 tokens at runtime**: it is a deterministic orchestrator
(discover → run in subprocess → score → keep/discard → converge → log). No LLM
calls. Confirmed: `grep -i 'anthropic\|openai\|messages.create' agent.py` → nothing.

All token cost is in the **Claude Code sessions** that proposed and wrote the
`AUTHOR = 'agent'` experiments (`listwise_softmax`, `hour_of_day`,
`bpr_hard_negative`, `bpr_hard_negative_warmstart`, `seq_dur_drift`).

**Why there's no exact token count:** the authoring account is on **Claude
Pro**, a flat-rate subscription, not the pay-per-token API: Anthropic's own
account panel for this plan type shows usage as a percentage of a rolling
quota, not an input/output token count (that granularity is only exposed to
metered API accounts). `agent.py --llm-tokens-in/--llm-tokens-out` exists for
teams on the API who can pull an exact number from Anthropic Console → Usage;
this run used neither.

| Metric | Value |
|---|---|
| `agent.py` runtime tokens | **0** (deterministic orchestrator, no LLM calls at runtime) |
| Authoring account type | Claude Pro (flat-rate subscription) |
| Exact input/output tokens | not exposed on this plan type |
| Session quota used (5h rolling) | **~66%**, at time of writing |
| Weekly quota used (7-day rolling) | **~58%**, at time of writing |
| Estimated cost | N/A: flat monthly subscription, not metered per call |
