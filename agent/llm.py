"""The agent's brain.

Two interchangeable implementations, same interface:
    complete(prompt, state) -> (reply_text, usage_dict)

- AnthropicLLM  : real Claude call, full token accounting. Used when
  ANTHROPIC_API_KEY is set (or --llm anthropic is passed).
- OfflinePlanner : deterministic research planner. Walks a ranked idea queue
  (each idea from the README's own "headroom" list), then hyper-parameter
  perturbations around the best variant, emitting a full solution.py every turn.
  Lets the whole loop run with zero external dependencies for the live demo.

Both return replies in the prompts.REPLY_FORMAT so the orchestrator parses them
identically.
"""
from __future__ import annotations

import json
import os
import textwrap

SOLUTION_TEMPLATE = '''\
# -*- coding: utf-8 -*-
"""solution.py - rewritten by the agent on iteration {iter}.

METHOD : {method}
{hypothesis_comment}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solutions.runner import run_variant

VARIANT = {variant!r}
PARAMS = {params!r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    metrics = run_variant(VARIANT, args.split, args.data_dir, PARAMS,
                          workspace=Path(__file__).resolve().parent)
    print("METRICS_JSON: " + json.dumps(metrics))


if __name__ == "__main__":
    main()
'''


def render_solution(iter_idx: int, method: str, hypothesis: str,
                    variant: str, params: dict) -> str:
    comment = textwrap.fill("HYP    : " + hypothesis, 78,
                            subsequent_indent="         ")
    return SOLUTION_TEMPLATE.format(
        iter=iter_idx, method=method, hypothesis_comment=comment,
        hypothesis=hypothesis, variant=variant, params=params)


# --------------------------------------------------------------------------- #
class OfflinePlanner:
    """Deterministic idea-queue planner. No network."""

    QUEUE = [
        dict(method="fm", variant="fm",
             params=dict(k=16, lr=0.001, epochs=25),
             hypothesis=(
                 "Anchor the run: reproduce the official Factorization Machine "
                 "baseline end-to-end (pointwise logloss, 5 categorical fields). "
                 "Every subsequent idea is judged against this number and the "
                 "oracle ceiling (0.8484), not against 1.0.")),
        dict(method="listwise_numpy", variant="listwise_numpy",
             params=dict(k=16, lr=0.0005, epochs=40, m_neg=4),
             hypothesis=(
                 "The eval metrics are ranking metrics but FM optimises "
                 "pointwise logloss. Same 5-field FM architecture, but for each "
                 "positive sample M=4 random negatives from the same user and "
                 "train with softmax cross-entropy over [positive, neg_1..neg_M] "
                 "(ListNet-style listwise loss; M=1 reduces exactly to BPR's "
                 "sigmoid(z_pos - z_neg), checked by hand). Validated in a "
                 "parallel harness: val primary 0.6039 vs FM's 0.6015, test "
                 "0.5973 vs FM's 0.5946 - the best score found with zero "
                 "human-written hypothesis in that run.")),
        dict(method="lambdarank", variant="lambdarank",
             params=dict(n_estimators=350, learning_rate=0.06, num_leaves=31),
             hypothesis=(
                 "The eval metrics (GAUC, nDCG@5) are ranking metrics but the FM "
                 "baseline optimises pointwise logloss. Switch to LightGBM "
                 "LambdaRank with groups = per-user impression sets and "
                 "label = long_view over leakage-guarded (leave-one-out) "
                 "target-encoded features, so the training objective directly "
                 "optimises NDCG. Organizers flagged this as the #1 untried idea.")),
        dict(method="history", variant="history",
             params=dict(n_estimators=350, learning_rate=0.06, num_leaves=31,
                         hist_last_n=20),
             hypothesis=(
                 "The model still sees zero behavioural history. Add DIN/SIM-style "
                 "per-user history features computed on train only: expanding and "
                 "rolling long_view rate, mean watch ratio, history length, and "
                 "user-author affinity. These give the ranker a personalised "
                 "prior that raw IDs cannot express for sparse users.")),
        dict(method="multitask", variant="multitask",
             params=dict(emb_dim=16, epochs=3, aux_weight=0.3, lr=1e-3),
             hypothesis=(
                 "is_click / is_like / play_time are logged but unused. Train a "
                 "shared-embedding network with auxiliary heads on those signals "
                 "(MMoE/ESMM-style) so the shared representation is regularised "
                 "by correlated engagement signals, improving the long_view head "
                 "especially where long_view is sparse.")),
        dict(method="combined", variant="combined",
             params=dict(n_seeds=3, blend=0.85, n_estimators=350,
                         learning_rate=0.06, num_leaves=31),
             hypothesis=(
                 "Blend the two most diverse models on per-user RANKS "
                 "(idea-bank #6): the pointwise FM (0.85) and the LambdaRank "
                 "model (0.15). Rank-averaging is scale-free and is exactly "
                 "what the metric rewards; the GBDT corrects the FM on the "
                 "slices where target-encoded video stats rank better than the "
                 "learned embedding.")),
        dict(method="censored", variant="censored",
             params=dict(emb_dim=16, epochs=3, complete_thresh=0.95, lr=1e-3),
             hypothesis=(
                 "A video watched to completion has a right-censored true watch "
                 "time, so squared error on watch ratio is biased. Apply CWM's "
                 "one-sided loss (penalise only under-prediction on completed "
                 "plays) and rank by predicted watch ratio as a long_view proxy.")),
    ]

    PERTURB = {
        "fm": [("k", 32), ("lr", 0.002), ("epochs", 40), ("k", 8)],
        "combined": [("blend", 0.75), ("blend", 0.95), ("blend", 1.0),
                     ("num_leaves", 63)],
        "_default": [("learning_rate", 0.03), ("num_leaves", 63),
                     ("n_estimators", 500), ("min_child_samples", 100)],
    }

    def __init__(self, inject_faults=()):
        self.turn = 0
        self._in = self._out = 0
        self.inject_faults = set(inject_faults or ())

    def complete(self, prompt: str, state) -> tuple[str, dict]:
        idx = len(state.history)
        if idx in self.inject_faults:
            # Person-3 robustness demo: hand the loop code that crashes at
            # runtime and confirm it recovers (rollback + traceback fed forward).
            bad = ('# -*- coding: utf-8 -*-\nimport sys\n'
                   'sys.stderr.write("injected fault: simulated bad edit\\n")\n'
                   'raise RuntimeError("planner produced an invalid model config")\n')
            reply = ("HYPOTHESIS: Try an aggressive config change (deliberately "
                     "faulty in this demo run) to exercise the recovery path.\n"
                     "METHOD: fault-injection\n```python\n" + bad + "\n```")
            return reply, {"input": 0, "output": 0}
        if self.turn < len(self.QUEUE):
            spec = self.QUEUE[self.turn]
        else:
            # perturbation phase around the current best variant
            k = self.turn - len(self.QUEUE)
            base = next((r for r in reversed(state.history)
                         if r.accepted and r.metrics), None)
            best_variant = getattr(base, "variant", "fm") if base else "fm"
            best_params = dict(getattr(base, "params", {}) or {})
            plist = self.PERTURB.get(best_variant, self.PERTURB["_default"])
            if k >= len(plist):
                spec = dict(method=best_variant, variant=best_variant,
                            params=best_params,
                            hypothesis=("No structural idea left in the queue and "
                                        "recent perturbations did not help; hold "
                                        "the current best configuration."))
            else:
                key, val = plist[k]
                best_params[key] = val
                spec = dict(
                    method=best_variant, variant=best_variant, params=best_params,
                    hypothesis=(f"Structural queue exhausted. Perturb the best "
                                f"variant ({best_variant}): set {key}={val} to "
                                f"probe whether a small regularisation/capacity "
                                f"change still moves validation primary."))
        self.turn += 1
        code = render_solution(idx, spec["method"], spec["hypothesis"],
                               spec["variant"], spec["params"])
        reply = (f"HYPOTHESIS: {spec['hypothesis']}\n"
                 f"METHOD: {spec['method']}\n```python\n{code}\n```")
        return reply, {"input": 0, "output": 0}

    def total_tokens(self) -> dict:
        return {"input": 0, "output": 0, "total": 0}


# --------------------------------------------------------------------------- #
class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 8000):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self._in = self._out = 0

    def complete(self, prompt: str, state=None) -> tuple[str, dict]:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}])
        self._in += resp.usage.input_tokens
        self._out += resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, {"input": resp.usage.input_tokens,
                      "output": resp.usage.output_tokens}

    def total_tokens(self) -> dict:
        return {"input": self._in, "output": self._out,
                "total": self._in + self._out}


def make_llm(kind: str, inject_faults=()):
    if kind == "auto":
        kind = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "offline"
    if kind == "anthropic":
        return AnthropicLLM()
    return OfflinePlanner(inject_faults=inject_faults)
