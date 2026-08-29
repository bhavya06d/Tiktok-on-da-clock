"""Person 2 (part 2) — prompt construction, idea bank, reply parsing.

This file IS the agent's intelligence. The Innovation criterion (20%) is judged
on what the agent chose to try and WHY, so the idea bank cites real methods and
the reply format forces a written hypothesis every iteration.

Keep prompts compact (Feasibility = token cost): send compressed history,
never raw data.
"""
from __future__ import annotations

import re

TASK_BRIEF = """You are an autonomous ML research agent improving a ranking
pipeline on KuaiRand-Pure (short-video recommendation).

TASK (pinned by organizers):
- Positive relevance label: the native `long_view` column (logged on EVERY
  impression, so no sample-selection bias on the scored label).
- Rank within each user's logged impressions (~5 per user in eval). NOT retrieval.
- Metrics: GAUC and nDCG@5; primary = mean of the two. Only relative score order
  matters; NaN/Inf are rejected.
- Eval conventions: users with zero positives count as nDCG=0 and are included;
  GAUC only counts users with 0 < positives < impressions, weighted by positive
  count; gain = 2^rel - 1.
- Official baseline (FM k=16, 5 categorical fields): val primary 0.6016.
  Random 0.4753, popularity 0.5715, ceiling ~0.8645. Beat 0.6016 decisively.
- Train split: dates 20220408-20220421; val: 20220422-20220428. Never touch test.
- CPU-only friendly; each full train+eval should stay under ~5 minutes.

HARD CONTRACT for solution.py:
- Accepts --split val|test.
- Trains ONLY on the train split (plus side-feature files shipped with KuaiRand).
- Prints exactly one line to stdout:
  METRICS_JSON: {"gauc": <f>, "ndcg5": <f>, "primary": <f>}
  (for --split test, evaluate.py has no labels available to you; print the
  last-known val metrics instead and still write the submission file)
- Writes submission_<split>.csv with header row_id,user_id,video_id,score,
  one row per evaluation-split row, row_id = 0-based index from data.load().
- Uses only: numpy, pandas, scikit-learn, lightgbm, torch (all installed).
"""

IDEA_BANK = """IDEA BANK — established methods worth trying (cite the one you
use in your hypothesis; prefer big structural moves over hyperparameter noise):
1. LightGBM with objective=lambdarank, groups = user impressions, label =
   long_view, eval_at=[5]. Directly optimizes NDCG; seconds to train on CPU.
2. Feature engineering beyond raw IDs: time-aware historical stats computed on
   the TRAIN split only (per-user long_view rate, per-video long_view rate,
   interaction counts, user activity), temporal features (hour, day-of-week),
   and the KuaiRand user/video side-feature files (allowed — part of the dataset).
   Guard against target leakage: never use future or eval-split rows for stats.
3. Multi-task learning (ESMM/MMoE-style): auxiliary heads on click / like /
   play_time to regularize the long_view head; shared embeddings.
4. Duration-bias correction (Zhao et al., KDD 2024, CWM): completed plays are
   censored by video length; normalize watch signals within duration buckets.
5. DeepFM / DCN-v2 with embedding dim 16-32 for learned feature crossing.
6. Ensembling: average the per-user RANKS (not raw scores) of your two best
   diverse models, e.g. GBDT + deep model.
7. If a run FAILED last iteration, your first duty is to fix the error.
"""

REPLY_FORMAT = """Reply in EXACTLY this format:
HYPOTHESIS: <2-4 sentences: what you will change, which method it draws on,
and why you expect it to raise GAUC/nDCG@5>
```python
<the COMPLETE new solution.py — full file, not a diff>
```"""


def compress_history(state) -> str:
    """Cheap token-wise summary of past iterations for the prompt."""
    lines = []
    for r in state.history[-12:]:  # cap context growth
        if r.metrics:
            lines.append(f"iter {r.idx}: primary={r.metrics['primary']:.4f} "
                         f"(gauc={r.metrics['gauc']:.4f}, "
                         f"ndcg5={r.metrics['ndcg5']:.4f})"
                         f"{' <- BEST' if r.accepted else ''} | {r.hypothesis[:160]}")
        else:
            lines.append(f"iter {r.idx}: FAILED ({(r.error or '')[:300]}) "
                         f"| {r.hypothesis[:120]}")
    best = f"{state.best_primary:.4f}" if state.history else "n/a"
    return f"Best val primary so far: {best}\n" + "\n".join(lines)


def build_iteration_prompt(current_code: str, state) -> str:
    return (f"{TASK_BRIEF}\n\n{IDEA_BANK}\n\n"
            f"RUN HISTORY:\n{compress_history(state)}\n\n"
            f"CURRENT solution.py:\n```python\n{current_code}\n```\n\n"
            f"{REPLY_FORMAT}")


def parse_agent_reply(reply: str) -> tuple[str, str | None]:
    hyp_match = re.search(r"HYPOTHESIS:\s*(.+?)(?=```|\Z)", reply, re.DOTALL)
    hypothesis = hyp_match.group(1).strip() if hyp_match else "(no hypothesis)"
    code_match = re.search(r"```python\n(.*?)```", reply, re.DOTALL)
    code = code_match.group(1) if code_match else None
    return hypothesis, code
