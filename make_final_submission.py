"""Generate submission.csv from the current best model (the overall
champion, not necessarily the autonomous-only one - a final submission
should use the best score actually achievable).

submit.py's own --make only knows how to run the plain official FM
baseline; it doesn't know about champions discovered under experiments/.
This script trains the current champion's exact configuration (mirroring
experiments/hour_of_day.py) and writes real per-row predictions through
submit.py's own write_submission, so the output format is guaranteed
identical/valid.

Usage:
    python3 make_final_submission.py [--split test] [--out submission.csv]

Re-run this whenever the champion changes - check agent_summary.json's
"champion" field first. Current champion (as of the hour_of_day result,
val 0.6052 / test 0.5986): listwise softmax loss + hour-of-day feature.
"""
import argparse
import csv
import os

import numpy as np

from baseline import FM
from submit import write_submission

SPLITS = {'train': (20220408, 20220421), 'valid': (20220422, 20220428), 'test': (20220429, 20220508)}
LABEL = 'long_view'


def _load_with_hour(data_dir):
    """Same as experiments/hour_of_day.py's loader - kept in sync there;
    duplicated here rather than imported so this script has no dependency
    on the experiments/ package layout."""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                hour = int(r['hourmin']) // 100
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                            vid2author.get(r['video_id'], 'UNK'), r['tab'],
                            float(r['duration_ms']), 1 if r[LABEL] != '0' else 0, hour))
    return {name: [x for x in rows if lo <= x[0] <= hi] for name, (lo, hi) in SPLITS.items()}


def _encode_with_hour(splits):
    tr = splits['train']
    edges = np.quantile(np.asarray([x[5] for x in tr]), np.linspace(0, 1, 10)[1:-1])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5]))), str(x[7])]

    vocabs = [dict() for _ in range(6)]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), 6), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


def _build_user_pos_neg(users, y):
    import collections
    pos_by_user, neg_by_user = collections.defaultdict(list), collections.defaultdict(list)
    for i, (u, yi) in enumerate(zip(users, y)):
        (pos_by_user if yi > 0 else neg_by_user)[u].append(i)
    eligible = [u for u in pos_by_user if u in neg_by_user]
    flat_pos = [(u, i) for u in eligible for i in pos_by_user[u]]
    neg_by_user = {u: np.array(neg_by_user[u]) for u in eligible}
    return flat_pos, neg_by_user


def train_champion(data_dir, k=16, lr=0.0005, epochs=40, bs=8192, patience=4, seed=0, m_neg=4):
    """Trains the current champion (listwise softmax + hour-of-day field)
    and returns the trained model + encoding, so callers can predict on
    any split."""
    from evaluate import evaluate
    splits = _load_with_hour(data_dir)
    enc, dim = _encode_with_hour(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    flat_pos, neg_by_user = _build_user_pos_neg(utr, ytr)
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        order = rng.permutation(len(flat_pos))
        for start in range(0, len(order), bs):
            batch = [flat_pos[j] for j in order[start:start + bs]]
            B = len(batch)
            rows_idx = np.empty((B, 1 + m_neg), dtype=np.int64)
            rows_idx[:, 0] = [i for _, i in batch]
            for r, (u, _) in enumerate(batch):
                pool = neg_by_user[u]
                rows_idx[r, 1:] = pool[rng.integers(len(pool), size=m_neg)]
            X = Xtr[rows_idx.reshape(-1)]
            z, E, S = m.logits(X)
            zg = z.reshape(B, 1 + m_neg)
            zg = zg - zg.max(1, keepdims=True)
            ez = np.exp(zg)
            p = ez / ez.sum(1, keepdims=True)
            coef = p.copy()
            coef[:, 0] -= 1.0
            coef = (coef / B).astype(np.float32)
            m._apply_grad(X, E, S, coef.reshape(-1))
        va = evaluate(uva, yva, m.predict(Xva))['primary']
        if va > best + 1e-5:
            best, bad, best_state = va, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    return m, enc, splits


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    ap.add_argument('--out', default='submission.csv')
    a = ap.parse_args()

    print("training champion config (listwise softmax + hour-of-day, k=16, lr=0.0005) ...")
    m, enc, splits = train_champion(a.data_dir)

    X, y, u = enc[a.split]
    scores = m.predict(X)
    write_submission(a.out, splits[a.split], scores)
    print(f"wrote {a.out}: {len(splits[a.split]):,} rows (split={a.split})")

    if a.split == 'valid':
        from evaluate import evaluate
        r = evaluate(u, y.tolist(), scores)
        print(f"  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
