"""Autonomous agent iteration #4 - first one to touch features, not the loss.

Every prior autonomous attempt (hard-negative mining, listwise softmax) only
changed the loss function. The champion (listwise_numpy) is still trained on
the same 5 fields as the original baseline: user_id, video_id, author_id,
tab, dur_bucket. Nothing has touched what the model actually SEES yet - and
the README's own headroom list has an untried item for exactly that: "time
features and distribution drift" (hourmin, date). Nobody (human or agent)
has used hourmin so far.

Hypothesis: recommendation behavior often has time-of-day structure (day-
parting) - e.g. a user's taste in video length or type may shift between
morning/commute, afternoon, evening, late night. Adding hour-of-day as a 6th
categorical field lets the FM's pairwise interaction term learn crosses like
"this user favors longer videos at night" that raw user_id alone can't
express. Everything else (listwise softmax loss, M=4, lr, k, epochs) is held
identical to the champion, so hour-of-day is the only variable being tested.

hourmin is HHMM (e.g. 1900 = 7pm); bucketed to the hour (0-23) as a plain
categorical field, same treatment as the other 5 fields.
"""
import csv
import os

import numpy as np

from baseline import FM
from evaluate import evaluate

PRIORITY = 5
DESCRIPTION = 'Add hour-of-day as a 6th categorical field (time-of-day/day-parting), listwise loss unchanged.'
AUTHOR = 'agent'

SPLITS = {'train': (20220408, 20220421), 'valid': (20220422, 20220428), 'test': (20220429, 20220508)}
LABEL = 'long_view'


def _load_with_hour(data_dir):
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

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


def _encode_with_hour(splits):
    tr = splits['train']
    edges = np.quantile(np.asarray([x[5] for x in tr]), np.linspace(0, 1, 10)[1:-1])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5]))), str(x[7])]

    fields = 6
    vocabs = [dict() for _ in range(fields)]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), fields), dtype=np.int32)
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


def run(splits_unused, data_dir='./KuaiRand-Pure/data', k=16, lr=0.0005, epochs=40,
       bs=8192, patience=4, seed=0, m_neg=4):
    splits = _load_with_hour(data_dir)
    enc, dim = _encode_with_hour(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    flat_pos, neg_by_user = _build_user_pos_neg(utr, ytr)
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, state, bad = -1.0, None, 0
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
            best, bad, state = va, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}
