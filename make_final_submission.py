"""Generate submission.csv from the official checkpoint.

submit.py's own --make only knows how to run the plain official FM
baseline; it doesn't know about champions discovered under experiments/.
This script trains the official checkpoint's exact configuration (mirroring
experiments/listwise_softmax.py) and writes real per-row predictions through
submit.py's own write_submission, so the output format is guaranteed
identical/valid.

Usage:
    python3 make_final_submission.py [--split test] [--out submission.csv]

Re-run this whenever the official checkpoint changes - check
agent_summary.json's "official_checkpoint" field first (this is the
validation-best checkpoint bounded to attempts up through the convergence
trigger, per the rule's "at that point" wording - see agent.py's comments -
not necessarily the single highest raw score found anywhere in the log).
Current official checkpoint: listwise softmax loss (val 0.6039 / test 0.5973),
the run's autonomous champion - see README.md.
"""
import argparse

import numpy as np

from baseline import FM, build_user_pos_neg
from data import load, encode
from submit import write_submission


def train_official_checkpoint(data_dir, k=16, lr=0.0005, epochs=40, bs=8192,
                              patience=4, seed=0, m_neg=4):
    """Trains the official checkpoint (listwise softmax loss, 5-field FM,
    matching experiments/listwise_softmax.py exactly) and returns the
    trained model + encoding, so callers can predict on any split."""
    from evaluate import evaluate
    splits = load(data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    flat_pos, neg_by_user = build_user_pos_neg(utr, ytr)
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

    print("training official checkpoint (listwise softmax, 5-field FM, k=16, lr=0.0005) ...")
    m, enc, splits = train_official_checkpoint(a.data_dir)

    X, y, u = enc[a.split]
    scores = m.predict(X)
    write_submission(a.out, splits[a.split], scores)
    print(f"wrote {a.out}: {len(splits[a.split]):,} rows (split={a.split})")

    if a.split == 'valid':
        from evaluate import evaluate
        r = evaluate(u, y.tolist(), scores)
        print(f"  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
