# Devpost: Autonomous ML Research Agent for KuaiRand-Pure

## What we built

We built an agent, not a recommender model: `agent.py` auto-discovers every
hypothesis under `experiments/`, runs it, scores it against the unmodified
official `evaluate.py`, and tracks the best validation score seen so far,
stopping on its own once 3 consecutive attempts fail to add more than 0.002.
The rule separates when to stop (the ε/N patience rule) from which
checkpoint is submitted (the validation-best result up to that point, no ε
filter). Bounded that way, our official checkpoint is `listwise_softmax`
(listwise ranking loss), validation primary 0.6039 against the official
baseline's 0.6016, a real gain of +0.0023, agent-authored with zero
human-written hypothesis behind it. That sits against an oracle ceiling of
0.8645, not 1.0. The baseline already spends about 31% of the usable
range, so the real headroom left is about 25 points, not a full point of
daylight to a perfect score.

## Technical execution: three real iterations

The run starts by anchoring on the baseline (0.6015 val, matching the
organizers' published 0.6016). A teammate's pairwise BPR loss came next,
0.6037, a real improvement we kept trying to beat rather than settle for.
The agent's own first attempt, hard-negative mining on the pairwise loss,
scored 0.5803, a clear miss, correctly discarded. Its next attempt,
listwise softmax over the same 5-field architecture instead of one
negative at a time, reasoned that GAUC and nDCG are ranking metrics and
the objective should match them directly. That scored 0.6039, edging out
the teammate's result, and became the official checkpoint once three
consecutive attempts failed to add another 0.002 and convergence
triggered. We kept the loop running past that point rather than stopping
dead, and it found an even higher raw number afterward, hour-of-day as a
sixth field, 0.6052, the first idea to change what the model sees rather
than how it trains. That result is real and reported, but falls outside
the convergence window under the rule's "at that point" wording, so it is
not the scored checkpoint. We separately verified the recovery path by
deliberately injecting a crash into a test run: the orchestrator logged
the traceback, left the last good champion untouched, and continued to
the next hypothesis without a restart.

## Innovation: full-stack, not hyperparameter noise

Every hypothesis is written before the code, not after. Across the run the
agent targeted the loss function (pairwise and listwise ranking losses
against the pointwise baseline), the feature set (hour-of-day, short-term
duration-preference drift), and, through a teammate's contribution,
user-history attention and multi-task auxiliary heads, each one a genuinely
different lever, not the same knob turned twice. The log also shows the
agent catching a distinction most teams would miss: a more elaborate,
warm-started implementation of a ranking loss can score badly on its own
merits while the same underlying idea, implemented simply, is a real win.
Knowing that difference, rather than writing off the idea outright, is
part of what got the run to its best number.

## Autonomy

Within the scored window, 1 idea was human-authored (`bpr_loss`, tried and
not selected) against 4 agent-authored attempts, so the scored checkpoint
needed zero human-written hypothesis behind it. Across the full 9-attempt
run, `author_counts` is 1 baseline, 2 human-authored, 6 agent-authored,
giving 2 manual interventions total. The second human idea (`user_history`)
came after convergence, reported honestly but not counted against the
scored result. Convergence itself, 3 consecutive attempts under the 0.002
threshold, was detected and stopped by the loop, not called by a person.

## Feasibility

The scored path is pure numpy, no GPU, and the full 9-attempt run completes
in 482 seconds, about 8 minutes, against a 6-hour cap and a 50-iteration
cap. `agent.py` itself makes 0 LLM calls at runtime; it is a deterministic
orchestrator. The token cost that does exist sits in the Claude Code
sessions that proposed and wrote the agent-authored experiments. That
account is on a flat-rate Claude Pro subscription rather than the
pay-per-token API, so an exact input and output count is not exposed,
only rolling quota usage, reported here as roughly 66% of a 5-hour session
and 58% of the weekly limit at time of writing rather than left blank.

A second, independent implementation also exists (`agent/` + `run_agent.py`,
real Anthropic API wiring and live token accounting), not the scored path
for this submission, but a second working system.

## One honest limitation

The scored gain is real but modest, about 1% of the attainable headroom
between baseline and oracle on test. Three structurally different ideas,
ranking loss, a time feature, and (via a teammate) behavioral history and
multi-task learning, were each implemented correctly and tested honestly,
and only the loss-function change actually became the scored checkpoint.
The time-feature result that scored higher still (hour-of-day, reported
above) arrived after the run had already converged, so it stands as a real
finding but not the submitted one. The agent stops correctly when an idea
plateaus. It does not yet redirect its own search toward a meaningfully
different family of ideas once one area stops paying off, and it does not
yet reopen a converged run when a later attempt turns out to have found
something better.

## Checklist

- **Development tools:** Claude Code (terminal / VS Code extension), macOS, git.
- **APIs:** `agent.py` makes 0 LLM calls at runtime (deterministic). Agent-authored
  experiments were proposed and written in Claude Code sessions, not via API.
  The secondary system separately wires in the Anthropic API for live, unattended
  runs.
- **Libraries:** numpy only for the scored path (`agent.py`, `baseline.py`,
  `experiments/`).
- **Datasets:** KuaiRand-Pure only, organizers' fixed train / validation / test
  splits. No external training data.
