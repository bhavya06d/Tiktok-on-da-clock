"""Person 4 — reference solution: LightGBM LambdaRank grouped by user.

Purpose: (a) prove by hand that this beats the FM baseline BEFORE trusting the
agent to find it; (b) serve as the MockLLM payload so Persons 1-3 can test the
harness; (c) live in the idea bank as method #1.

ADAPT THE DATA LOADING to the starter kit's `data.load()` — column names below
are from the KuaiRand log schema; verify against the actual CSVs:
log_standard_4_08_to_4_21_pure.csv / log_standard_4_22_to_5_08_pure.csv.
Honors the solution.py contract (METRICS_JSON line + submission_<split>.csv).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

LABEL = "long_view"
CATS = ["user_id", "video_id", "tab", "hourly_time"]  # verify against real schema


def load(data_dir: Path):
    a = pd.read_csv(data_dir / "log_standard_4_08_to_4_21_pure.csv")
    b = pd.read_csv(data_dir / "log_standard_4_22_to_5_08_pure.csv")
    df = pd.concat([a, b], ignore_index=True)
    train = df[df.date.between(20220408, 20220421)].reset_index(drop=True)
    val = df[df.date.between(20220422, 20220428)].reset_index(drop=True)
    test = df[df.date.between(20220429, 20220508)].reset_index(drop=True)
    return train, val, test


def engineer(train: pd.DataFrame, eval_df: pd.DataFrame):
    """Time-aware stats computed on TRAIN ONLY (leakage guard)."""
    global_rate = train[LABEL].mean()
    u = train.groupby("user_id")[LABEL].agg(["mean", "count"])
    v = train.groupby("video_id")[LABEL].agg(["mean", "count"])

    def add(df):
        out = df.copy()
        out["u_rate"] = df.user_id.map(u["mean"]).fillna(global_rate)
        out["u_cnt"] = df.user_id.map(u["count"]).fillna(0)
        out["v_rate"] = df.video_id.map(v["mean"]).fillna(global_rate)
        out["v_cnt"] = df.video_id.map(v["count"]).fillna(0)
        # TODO(P4): hour-of-day, day-of-week, user/video side-feature joins,
        # smoothed (Bayesian) target encoding, play_time-derived train features.
        return out

    feats = ["u_rate", "u_cnt", "v_rate", "v_cnt"] + \
            [c for c in CATS if c in train.columns]
    return add(train), add(eval_df), feats


def gauc_ndcg5(user_ids, labels, scores):
    """Mirror of the pinned evaluate.py conventions — for local iteration only.
    THE agent-scored metrics must come from the starter kit's evaluate.py;
    Person 4: replace this with `import evaluate` from the kit once unpacked."""
    df = pd.DataFrame({"u": user_ids, "y": labels, "s": scores})
    gaucs, weights, ndcgs = [], [], []
    for _, g in df.groupby("u"):
        y = g.y.to_numpy()
        order = np.argsort(-g.s.to_numpy(), kind="stable")
        yr = y[order][:5]
        gains = (2.0 ** yr - 1) / np.log2(np.arange(2, len(yr) + 2))
        ideal = np.sort(y)[::-1][:5]
        ig = (2.0 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))
        ndcgs.append(gains.sum() / ig.sum() if ig.sum() > 0 else 0.0)
        npos = int(y.sum())
        if 0 < npos < len(y):
            pos = g.s.to_numpy()[y == 1][:, None]
            neg = g.s.to_numpy()[y == 0][None, :]
            auc = ((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size / 1)
            gaucs.append(auc); weights.append(npos)
    gauc = float(np.average(gaucs, weights=weights)) if gaucs else 0.0
    ndcg = float(np.mean(ndcgs))
    return gauc, ndcg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--data-dir", type=Path, default=Path("../data"))
    args = ap.parse_args()

    train, val, test = load(args.data_dir)
    eval_df = val if args.split == "val" else test
    tr, ev, feats = engineer(train, eval_df)

    tr = tr.sort_values("user_id", kind="stable")
    groups = tr.groupby("user_id", sort=False).size().to_numpy()
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=400,
                           learning_rate=0.05, num_leaves=63,
                           label_gain=[0, 1], random_state=0)
    model.fit(tr[feats], tr[LABEL], group=groups)
    scores = model.predict(ev[feats])

    gauc, ndcg = gauc_ndcg5(ev.user_id, ev[LABEL], scores)
    primary = (gauc + ndcg) / 2
    print(f"METRICS_JSON: "
          f"{json.dumps({'gauc': gauc, 'ndcg5': ndcg, 'primary': primary})}")

    out = pd.DataFrame({"row_id": np.arange(len(ev)), "user_id": ev.user_id,
                        "video_id": ev.video_id, "score": scores})
    out.to_csv(f"submission_{args.split}.csv", index=False)


if __name__ == "__main__":
    main()
