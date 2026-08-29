"""Generate submission.csv from the current best model (the overall
champion, not necessarily the autonomous-only one - a final submission
should use the best score actually achievable).

submit.py's own --make only knows how to run the plain official FM
baseline; it doesn't know about champions discovered under experiments/.
This script trains the current champion's exact configuration (mirroring
baseline.py's run_fm_bpr) and writes real per-row predictions through
submit.py's own write_submission, so the output format is guaranteed
identical/valid.

Usage:
    python3 make_final_submission.py [--split test] [--out submission.csv]

Re-run this whenever the champion changes - check agent_summary.json's
"champion" field first.
"""
import argparse

import numpy as np

from baseline import FM, build_user_pos_neg
from data import load, encode
from submit import write_submission


def train_champion_bpr(splits, k=16, lr=0.0005, epochs=40, bs=8192, patience=4, seed=0):
    """Mirrors baseline.py's run_fm_bpr exactly, but returns the trained
    model instead of just evaluation metrics, so we can call .predict()
    on the submission split."""
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    flat_pos, neg_by_user = build_user_pos_neg(utr, ytr)
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    from evaluate import evaluate
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(flat_pos))
        for start in range(0, len(order), bs):
            batch = [flat_pos[j] for j in order[start:start + bs]]
            pos_rows = np.array([i for _, i in batch])
            neg_rows = np.array([neg_by_user[u][rng.integers(len(neg_by_user[u]))] for u, _ in batch])
            m.step_bpr(Xtr[pos_rows], Xtr[neg_rows])
        va = evaluate(uva, yva, m.predict(Xva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    return m, enc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    ap.add_argument('--out', default='submission.csv')
    a = ap.parse_args()

    print("loading data ...")
    splits = load(a.data_dir)
    rows = splits[a.split]

    print("training champion config (BPR loss, k=16, lr=0.0005) ...")
    m, enc = train_champion_bpr(splits)

    X, y, u = enc[a.split]
    scores = m.predict(X)
    write_submission(a.out, rows, scores)
    print(f"wrote {a.out}: {len(rows):,} rows (split={a.split})")

    if a.split == 'valid':
        from evaluate import evaluate
        r = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
        print(f"  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
