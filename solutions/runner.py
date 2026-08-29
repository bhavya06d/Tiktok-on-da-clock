"""Variant registry — every idea in the agent's idea bank, one function each.

Contract (shared by all): return {"gauc", "ndcg5", "primary"} on the eval split
and write submission_<split>.csv aligned to the official row order.

    fm          numpy Factorization Machine (starter-kit baseline, the bar)
    lambdarank  LightGBM LambdaRank grouped by user   (idea: ranking loss)
    bpr         alias of lambdarank (pairwise/listwise objective)
    history     LambdaRank + per-user behavioural history features (DIN-lite)
    multitask   torch shared-embedding net, aux heads on click/like/watch (MMoE-lite)
    censored    torch one-sided watch-time regression (CWM-style duration debiasing)
    combined    LambdaRank over history features + multitask score as a stacked feature
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate  # noqa: E402  (official scorer, never modified)
from ml.features import (AUX_LABELS, FeatureConfig, build_features,  # noqa: E402
                         load_raw)

DATA_DIR_DEFAULT = str(ROOT / "KuaiRand-Pure" / "data")


def _pack(res: dict) -> dict:
    return {"gauc": float(res["GAUC"]), "ndcg5": float(res["nDCG@5"]),
            "primary": float(res["primary"])}


def _write_submission(path: Path, ev: pd.DataFrame, scores: np.ndarray) -> None:
    out = pd.DataFrame({
        "row_id": np.arange(len(ev)),
        "user_id": ev["user_id"].to_numpy(),
        "video_id": ev["video_id"].to_numpy(),
        "score": np.asarray(scores, dtype=float),
    })
    out.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
def _run_lgbm(raw, split, params, use_history, use_ids=False):
    import lightgbm as lgb

    cfg = FeatureConfig(use_history=use_history,
                        hist_last_n=int(params.get("hist_last_n", 20)),
                        smoothing=float(params.get("smoothing", 20.0)))
    fb = build_features(raw, split, cfg)
    feat_cols = list(fb.feats)
    cat_cols = list(fb.cat_feats)
    if use_ids:
        for c in ["user_id_code", "video_id_code", "author_id_code"]:
            feat_cols.append(c)
            cat_cols.append(c)
    Xtr = fb.Xtr[feat_cols].iloc[fb.tr_sort_idx].reset_index(drop=True)
    ytr = fb.ytr[fb.tr_sort_idx]

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=int(params.get("n_estimators", 350)),
        learning_rate=float(params.get("learning_rate", 0.06)),
        num_leaves=int(params.get("num_leaves", 31)),
        min_child_samples=int(params.get("min_child_samples", 50)),
        subsample=float(params.get("subsample", 0.8)),
        subsample_freq=1,
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        max_bin=int(params.get("max_bin", 255)),
        random_state=0, n_jobs=-1, verbose=-1,
    )
    model.fit(Xtr, ytr.astype(int), group=fb.group_tr,
              categorical_feature=cat_cols)
    scores = model.predict(fb.Xev[feat_cols])
    res = evaluate(raw[split]["user_id"].tolist(), fb.yev.tolist(),
                   scores.tolist())
    return res, raw[split], scores


def run_fm(split, data_dir, params):
    from data import encode, load
    import baseline as B
    splits = load(data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xev, yev, uev = enc[split if split != "val" else "valid"]
    seed = int(params.get("seed", 0))
    m = B.FM(dim, k=int(params.get("k", 16)),
             lr=float(params.get("lr", 0.001)), seed=seed)
    rng = np.random.default_rng(seed)
    Xva, yva, uva = enc["valid"]
    best, state, bad = -1.0, None, 0
    for _ in range(int(params.get("epochs", 40))):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            m.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
        p = evaluate(uva, yva, m.predict(Xva))["primary"]
        if p > best + 1e-5:
            best, bad, state = p, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4:
                break
    m.V, m.W, m.b = state
    scores = m.predict(Xev)
    res = evaluate(uev, yev, scores)
    raw_ev = pd.DataFrame({"user_id": uev, "video_id": [x[2] for x in
                           splits["valid" if split != "test" else "test"]]})
    return res, raw_ev, scores


def run_bpr(split, data_dir, params, workspace):
    """Person-1 idea: replace the FM baseline's POINTWISE logloss with a
    LISTWISE ranking objective (per-user softmax / ListNet), with a BPR
    fallback. Same architecture and feature space as the FM baseline
    (data.encode's 5 fields) so the only variable is the loss function —
    which is the organizers' #1 untried lever, and the eval metrics
    (GAUC / nDCG@5) are themselves ranking metrics.

    Training: sample B users per step, up to K impressions each (>=1 positive),
    score them, take a masked softmax over each user's K slots, and push the
    probability mass onto that user's long_view impressions.
    """
    import torch
    import torch.nn as nn
    from data import encode, load

    torch.manual_seed(0)
    np.random.seed(0)
    splits = load(data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    ev_name = "valid" if split != "test" else "test"
    Xva, yva, uva = enc["valid"]
    Xev, yev, uev = enc[ev_name]

    k = int(params.get("k", 32))
    loss_kind = params.get("loss", "softmax")
    epochs = int(params.get("epochs", 25))
    steps_per_epoch = int(params.get("steps_per_epoch", 200))
    n_users_b = int(params.get("users_per_batch", 1024))
    K = int(params.get("list_len", 24))
    lr = float(params.get("lr", 0.02))
    wd = float(params.get("weight_decay", 2e-6))

    # group train rows by user; keep users that have both a pos and a neg.
    # Build padded per-user positive / negative row-index pools for fully
    # vectorised list sampling (no python loop in the training step).
    P_CAP, N_CAP = 48, 96
    order = np.argsort(utr, kind="stable")
    utr_s = np.asarray(utr)[order]
    ytr_s = ytr[order]
    Xtr_s = Xtr[order]
    uniq, starts = np.unique(utr_s, return_index=True)
    ends = np.r_[starts[1:], len(utr_s)]
    rng = np.random.default_rng(0)
    pos_pool, neg_pool, pos_cnt, neg_cnt = [], [], [], []
    for s, e in zip(starts, ends):
        rows = np.arange(s, e)
        p = rows[ytr_s[s:e] > 0.5]
        n = rows[ytr_s[s:e] < 0.5]
        if len(p) == 0 or len(n) == 0:
            continue
        if len(p) > P_CAP:
            p = rng.choice(p, P_CAP, replace=False)
        if len(n) > N_CAP:
            n = rng.choice(n, N_CAP, replace=False)
        pos_cnt.append(len(p))
        neg_cnt.append(len(n))
        pos_pool.append(np.pad(p, (0, P_CAP - len(p))))
        neg_pool.append(np.pad(n, (0, N_CAP - len(n))))
    pos_pool = np.asarray(pos_pool)
    neg_pool = np.asarray(neg_pool)
    pos_cnt = np.asarray(pos_cnt)
    neg_cnt = np.asarray(neg_cnt)
    G = len(pos_cnt)
    n_pos = max(1, K // 4)
    n_neg = K - n_pos

    Xtr_t = torch.as_tensor(Xtr_s, dtype=torch.long)

    class FMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(dim, k)
            self.bias = nn.Embedding(dim, 1)
            nn.init.normal_(self.emb.weight, std=0.01)
            nn.init.zeros_(self.bias.weight)

        def forward(self, x):                       # x: (..., F)
            e = self.emb(x)
            s = e.sum(-2)
            inter = 0.5 * (s.pow(2).sum(-1) - e.pow(2).sum((-1, -2)))
            return self.bias(x).sum((-1, -2)) + inter

    net = FMNet()

    # ---- pointwise warm-start: FM converges fast & data-efficiently on the
    # full logloss objective; we then rank-tune the same weights so the loss
    # matches the (ranking) eval metric without throwing away that head start.
    if bool(params.get("warmstart", True)):
        wopt = torch.optim.Adam(net.parameters(), lr=float(params.get("warm_lr", 3e-3)),
                                weight_decay=wd)
        bce = nn.BCEWithLogitsLoss()
        Xall = torch.as_tensor(Xtr_s, dtype=torch.long)
        yall = torch.as_tensor(ytr_s, dtype=torch.float32)
        Nn = len(yall)
        wbs = 8192
        wbest, wstate, wbad = -1.0, None, 0
        for wep in range(int(params.get("warm_epochs", 15))):
            net.train()
            perm = torch.randperm(Nn)
            for i in range(0, Nn, wbs):
                b = perm[i:i + wbs]
                wopt.zero_grad()
                loss = bce(net(Xall[b]), yall[b])
                loss.backward()
                wopt.step()
            net.eval()
            with torch.no_grad():
                sv = net(torch.as_tensor(Xva, dtype=torch.long)).numpy()
            wp = evaluate(uva, yva, sv)["primary"]
            if wp > wbest + 1e-5:
                wbest, wbad = wp, 0
                wstate = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
            else:
                wbad += 1
                if wbad >= 3:
                    break
        if wstate:
            net.load_state_dict(wstate)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    lab_np = np.concatenate([np.ones((1, n_pos), np.float32),
                             np.zeros((1, n_neg), np.float32)], 1)

    def sample_batch():
        gi = np.random.randint(0, G, size=n_users_b)
        ps = (np.random.random((n_users_b, n_pos)) *
              pos_cnt[gi][:, None]).astype(np.int64)
        ns = (np.random.random((n_users_b, n_neg)) *
              neg_cnt[gi][:, None]).astype(np.int64)
        pidx = np.take_along_axis(pos_pool[gi], ps, axis=1)
        nidx = np.take_along_axis(neg_pool[gi], ns, axis=1)
        idx = np.concatenate([pidx, nidx], axis=1)
        lab = np.broadcast_to(lab_np, (n_users_b, K))
        return (torch.as_tensor(idx),
                torch.as_tensor(np.ascontiguousarray(lab)),
                None)

    def evaluate_on(Xarr, uarr, yarr):
        net.eval()
        with torch.no_grad():
            sc = net(torch.as_tensor(Xarr, dtype=torch.long)).numpy()
        return evaluate(uarr, yarr, sc), sc

    best_primary, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        net.train()
        for _ in range(steps_per_epoch):
            idx, lab, _ = sample_batch()
            opt.zero_grad()
            sc = net(Xtr_t[idx])                          # (B,K)
            if loss_kind == "bpr":
                pos_s = (sc * lab).sum(1) / lab.sum(1).clamp(min=1)
                neg_s = (sc * (1 - lab)).sum(1) / (1 - lab).sum(1).clamp(min=1)
                loss = -nn.functional.logsigmoid(pos_s - neg_s).mean()
            else:  # listwise softmax / ListNet top-1
                tgt = lab / lab.sum(1, keepdim=True).clamp(min=1)
                loss = -(tgt * torch.log_softmax(sc, dim=1)).sum(1).mean()
            loss.backward()
            opt.step()
        (r, _) = evaluate_on(Xva, uva, yva)
        p = r["primary"]
        if p > best_primary + 1e-5:
            best_primary, bad = p, 0
            best_state = {kk: v.detach().clone()
                          for kk, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= int(params.get("patience", 5)):
                break
    if best_state:
        net.load_state_dict(best_state)
    res, scores = evaluate_on(Xev, uev, yev)
    ev_df = pd.DataFrame({"user_id": uev,
                          "video_id": [x[2] for x in splits[ev_name]]})
    return res, ev_df, scores


# --------------------------------------------------------------------------- #
def run_variant(variant: str, split: str, data_dir: str | None = None,
                params: dict | None = None, workspace: Path | None = None):
    params = params or {}
    data_dir = data_dir or DATA_DIR_DEFAULT
    workspace = Path(workspace or ".")
    split_key = "valid" if split == "val" else split

    if variant == "fm":
        res, ev, scores = run_fm(split, data_dir, params)
    else:
        if variant == "bpr":
            res, ev, scores = run_bpr(split, data_dir, params, workspace)
            _write_submission(workspace / f"submission_{split}.csv", ev, scores)
            return _pack(res)
        raw = load_raw(data_dir)
        if variant == "lambdarank":
            res, ev, scores = _run_lgbm(raw, split_key, params, use_history=False)
        elif variant == "history":
            res, ev, scores = _run_lgbm(raw, split_key, params, use_history=True)
        elif variant == "multitask":
            res, ev, scores = _run_multitask(raw, split_key, params)
        elif variant == "censored":
            res, ev, scores = _run_censored(raw, split_key, params)
        elif variant == "combined":
            res, ev, scores = _run_combined(raw, split, data_dir, params)
        else:
            raise ValueError(f"unknown variant {variant!r}")

    _write_submission(workspace / f"submission_{split}.csv", ev, scores)
    return _pack(res)


# --------------------------------------------------------------------------- #
# torch models
# --------------------------------------------------------------------------- #
def _torch_prep(raw, split_key, cfg):
    fb = build_features(raw, split_key, cfg)
    cat_cols = fb.id_feats
    Xtr_cat = fb.Xtr[cat_cols].to_numpy()
    Xev_cat = fb.Xev[cat_cols].to_numpy()
    # clamp unseen (-1) codes to a dedicated slot per column
    card = []
    for j in range(Xtr_cat.shape[1]):
        mx = int(max(Xtr_cat[:, j].max(), 0)) + 2
        Xtr_cat[:, j] = np.where(Xtr_cat[:, j] < 0, mx - 1, Xtr_cat[:, j])
        Xev_cat[:, j] = np.where((Xev_cat[:, j] < 0) | (Xev_cat[:, j] >= mx),
                                 mx - 1, Xev_cat[:, j])
        card.append(mx)
    return fb, Xtr_cat.astype(np.int64), Xev_cat.astype(np.int64), card


def _run_multitask(raw, split_key, params):
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    cfg = FeatureConfig(use_history=bool(params.get("use_history", False)))
    fb, Xtr_cat, Xev_cat, card = _torch_prep(raw, split_key, cfg)
    dim = int(params.get("emb_dim", 16))
    epochs = int(params.get("epochs", 3))
    bs = int(params.get("batch_size", 4096))
    aux_w = float(params.get("aux_weight", 0.3))

    class MTL(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList(
                [nn.Embedding(c, dim) for c in card])
            h = dim * len(card)
            self.shared = nn.Sequential(
                nn.Linear(h, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
            self.head_lv = nn.Linear(64, 1)
            self.head_click = nn.Linear(64, 1)
            self.head_like = nn.Linear(64, 1)
            self.head_watch = nn.Linear(64, 1)

        def forward(self, x):
            e = torch.cat([emb(x[:, j]) for j, emb in enumerate(self.embs)], 1)
            z = self.shared(e)
            return (self.head_lv(z).squeeze(1), self.head_click(z).squeeze(1),
                    self.head_like(z).squeeze(1), self.head_watch(z).squeeze(1))

    model = MTL()
    opt = torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 1e-3)))
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    Xtr_t = torch.from_numpy(Xtr_cat)
    y_lv = torch.from_numpy(fb.ytr)
    y_click = torch.from_numpy(fb.aux_tr.get("is_click", fb.ytr))
    y_like = torch.from_numpy(fb.aux_tr.get("is_like", fb.ytr))
    y_watch = torch.from_numpy(np.clip(fb.aux_tr["watch_ratio"], 0, 1))
    n = len(y_lv)

    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            p_lv, p_c, p_l, p_w = model(Xtr_t[b])
            loss = (bce(p_lv, y_lv[b])
                    + aux_w * bce(p_c, y_click[b])
                    + aux_w * bce(p_l, y_like[b])
                    + aux_w * mse(torch.sigmoid(p_w), y_watch[b]))
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(Xev_cat))[0].numpy()
    res = evaluate(raw[split_key]["user_id"].tolist(), fb.yev.tolist(),
                   scores.tolist())
    return res, raw[split_key], scores


def _run_censored(raw, split_key, params):
    """One-sided (censored) watch-time regression: if a video played (near) to
    completion the 'would-have-watched' time is right-censored, so only penalise
    under-prediction there (CWM, Zhao et al. KDD'24)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    cfg = FeatureConfig()
    fb, Xtr_cat, Xev_cat, card = _torch_prep(raw, split_key, cfg)
    dim = int(params.get("emb_dim", 16))
    epochs = int(params.get("epochs", 3))
    bs = int(params.get("batch_size", 4096))

    tr = raw["train"]
    wr = np.clip((tr["play_time_ms"] / tr["duration_ms"].clip(lower=1)).to_numpy(),
                 0, 1).astype(np.float32)
    censored = (wr >= float(params.get("complete_thresh", 0.95))).astype(np.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(c, dim) for c in card])
            self.mlp = nn.Sequential(
                nn.Linear(dim * len(card), 128), nn.ReLU(),
                nn.Linear(128, 1))

        def forward(self, x):
            e = torch.cat([emb(x[:, j]) for j, emb in enumerate(self.embs)], 1)
            return self.mlp(e).squeeze(1)

    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 1e-3)))
    Xtr_t = torch.from_numpy(Xtr_cat)
    wr_t = torch.from_numpy(wr)
    cen_t = torch.from_numpy(censored)
    n = len(wr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pred = torch.sigmoid(model(Xtr_t[b]))
            diff = pred - wr_t[b]
            # squared error, but for censored rows only when pred < observed
            w = torch.where(cen_t[b] > 0, (diff < 0).float(), torch.ones_like(diff))
            loss = (w * diff.pow(2)).mean()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(Xev_cat))).numpy()
    res = evaluate(raw[split_key]["user_id"].tolist(), fb.yev.tolist(),
                   scores.tolist())
    return res, raw[split_key], scores


def _peruser_rank(users, scores):
    """0..1 within-user percentile rank (higher score -> higher rank)."""
    s = pd.DataFrame({"u": np.asarray(users), "s": np.asarray(scores)})
    return s.groupby("u")["s"].rank(pct=True, method="average").to_numpy()


def _run_combined(raw, split, data_dir, params):
    """Ensemble of diverse models, blended on per-user RANKS (idea-bank #6):
    a seed-ensemble of the pointwise FM (strong pointwise signal) + a LightGBM
    LambdaRank over the leakage-guarded target-encoded features (different
    inductive bias). Rank-averaging is robust to the models' different score
    scales and is what the eval metric actually cares about."""
    split_key = "valid" if split == "val" else split
    users = raw[split_key]["user_id"].to_numpy()
    y = raw[split_key]["long_view"].to_numpy()

    n_seeds = int(params.get("n_seeds", 1))
    blend = float(params.get("blend", 0.85))         # weight on FM

    # cache the (expensive) per-seed FM rank vectors so hyper-parameter
    # perturbations that only change `blend` don't retrain the ensemble.
    from ml.features import _CACHE_DIR
    fm_ranks = []
    for sd in range(n_seeds):
        cpath = os.path.join(_CACHE_DIR, f"fmrank_{split}_seed{sd}.npy")
        if os.path.exists(cpath):
            fm_ranks.append(np.load(cpath))
            continue
        _, _, s = run_fm(split, data_dir,
                         {"k": 16, "lr": 0.001, "epochs": 25, "seed": sd})
        r = _peruser_rank(users, s)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        np.save(cpath, r)
        fm_ranks.append(r)
    fm_rank = np.mean(fm_ranks, axis=0)

    lkey = "-".join(str(params.get(k, "")) for k in
                    ("n_estimators", "learning_rate", "num_leaves"))
    lpath = os.path.join(_CACHE_DIR, f"lgbmrank_{split}_{lkey}.npy")
    if os.path.exists(lpath):
        lgbm_rank = np.load(lpath)
    else:
        _, _, lr_scores = _run_lgbm(raw, split_key, params, use_history=False)
        lgbm_rank = _peruser_rank(users, lr_scores)
        np.save(lpath, lgbm_rank)

    scores = blend * fm_rank + (1 - blend) * lgbm_rank
    res = evaluate(users.tolist(), y.tolist(), scores.tolist())
    return res, raw[split_key], scores


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="lambdarank")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    metrics = run_variant(a.variant, a.split, a.data_dir, json.loads(a.params))
    print("METRICS_JSON: " + json.dumps(metrics))
