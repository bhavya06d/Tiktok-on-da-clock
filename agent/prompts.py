"""Prompt construction, idea bank, reply parsing — the agent's intelligence.

The Innovation criterion is judged on WHAT the agent chose to try and WHY, so
the idea bank cites real methods and every reply must carry a written hypothesis
plus a full replacement solution.py. History is compressed to keep token cost
(Feasibility) low.
"""
from __future__ import annotations

import re

TASK_BRIEF = """You are an autonomous ML research agent improving a within-user
ranking pipeline on KuaiRand-Pure (short-video recommendation).

TASK (pinned by organizers, do not reinterpret):
- Positive label: the native `long_view` column (0/1, logged on every impression).
- Rank within each user's logged impressions. NOT retrieval.
- Metrics: GAUC and nDCG@5; primary = mean of the two. Only relative order matters.
- Splits: train 20220408-20220421, val 20220422-20220428, test 20220429-20220508.
  Train on train only. Never touch test labels.
- Official FM baseline: val primary 0.6016 / test 0.5946. Random 0.4753,
  popularity 0.5715, oracle ceiling 0.8645. Beat 0.6016 decisively.
- CPU-only; a full train+eval should stay under ~10 minutes.

HARD CONTRACT for solution.py:
- Accepts `--split val|test` and `--data-dir <path>`.
- Trains only on the train split (+ KuaiRand side-feature files, which are allowed).
- Prints exactly one line: METRICS_JSON: {"gauc": <f>, "ndcg5": <f>, "primary": <f>}
- Writes submission_<split>.csv with header row_id,user_id,video_id,score
  (row_id = 0-based index over the eval split in official order).
- Uses only numpy, pandas, scikit-learn, lightgbm, torch.
- The official scorer is evaluate.py — never modify it; import and call it.
"""

IDEA_BANK = """IDEA BANK — established methods (cite the one you use; prefer big
structural moves over hyperparameter noise). The starter kit already ruled out
"more static features" and "bigger embeddings" — do not retry those.

1. RANKING LOSS (LightGBM LambdaRank): objective=lambdarank, group = per-user
   impression counts, label = long_view. Directly optimises NDCG; trains in
   seconds on CPU. The eval metrics are ranking metrics, so aligning the loss
   is the organizers' #1 untried idea.
2. FEATURE ENGINEERING (train-split only, leakage-guarded): smoothed per-user /
   per-video / per-author long_view rates, watch-ratio target encoding,
   temporal (hour, day-of-week), KuaiRand user/video side files.
3. USER HISTORY / SEQUENCE (DIN/SIM-style): each user has hundreds of prior
   train interactions, currently unused. Expanding & rolling long_view rate,
   mean watch ratio, author affinity from history.
4. MULTI-TASK LEARNING (MMoE / ESMM-style): shared embeddings with auxiliary
   heads on is_click / is_like / play_time to regularise the long_view head.
5. WATCH-TIME CENSORED REGRESSION (CWM, Zhao et al. KDD 2024): a video played
   to completion has a right-censored "would-have-watched" time; use a one-sided
   loss instead of squared error.
6. STACKING / ENSEMBLE: feed a diverse model's score (e.g. the multi-task net)
   as a feature into the LambdaRank model, or average per-user ranks.
7. If the last run FAILED, your first duty is to fix that error.
"""

REPLY_FORMAT = """Reply in EXACTLY this format and nothing else:
HYPOTHESIS: <2-4 sentences: what you will change, which method it draws on, and
why you expect it to raise GAUC/nDCG@5>
METHOD: <one short tag, e.g. lambdarank | history | multitask | censored | combined>
```python
<the COMPLETE new solution.py — full file, not a diff>
```"""


def compress_history(state) -> str:
    lines = []
    for r in state.history[-12:]:
        if r.metrics:
            lines.append(
                f"iter {r.idx} [{r.method}]: primary={r.metrics['primary']:.4f} "
                f"(gauc={r.metrics['gauc']:.4f} ndcg5={r.metrics['ndcg5']:.4f})"
                f"{' <- BEST' if r.accepted else ''} | {r.hypothesis[:140]}")
        else:
            lines.append(f"iter {r.idx} [{r.method}]: FAILED "
                         f"({(r.error or '')[:220]})")
    best = f"{state.best_primary:.4f}" if state.history else "n/a"
    return f"Best val primary so far: {best}\n" + "\n".join(lines)


def build_iteration_prompt(current_code: str, state) -> str:
    return (f"{TASK_BRIEF}\n\n{IDEA_BANK}\n\nRUN HISTORY:\n"
            f"{compress_history(state)}\n\nCURRENT solution.py:\n"
            f"```python\n{current_code}\n```\n\n{REPLY_FORMAT}")


def parse_agent_reply(reply: str) -> tuple[str, str, str | None]:
    hyp = re.search(r"HYPOTHESIS:\s*(.+?)(?=\nMETHOD:|\n```|\Z)", reply, re.DOTALL)
    hypothesis = hyp.group(1).strip() if hyp else "(no hypothesis)"
    meth = re.search(r"METHOD:\s*([^\n]+)", reply)
    method = meth.group(1).strip() if meth else "unknown"
    code = re.search(r"```python\n(.*?)```", reply, re.DOTALL)
    return hypothesis, method, (code.group(1) if code else None)
