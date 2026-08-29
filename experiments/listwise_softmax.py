"""Autonomous agent iteration #3.

Both hard-negative-mining attempts (bpr_hard_negative*.py) underperformed
plain BPR. Stepping back to a different idea from the README's own headroom
list, distinct from BPR: a *listwise* loss instead of pairwise. For each
positive, sample M negatives from the same user and train with a softmax
cross-entropy over [positive, neg_1, ..., neg_M] - "the positive should
outscore all M of these at once", rather than BPR's one-negative-at-a-time
comparison. This gives a denser gradient per step than pairwise BPR without
the instability hard-negative mining ran into (negatives here are still
uniformly random, only the loss aggregation changes).

Sanity check: with M=1 negative, softmax over 2 items reduces exactly to
BPR's sigmoid(z_pos - z_neg) - confirmed by hand before trusting this file's
M=4 result below.

Follow-up sweep (not separate files - results were statistically flat, not
worth the clutter): M=8 gave valid/test primary 0.6038/0.5977, indistinguishable
from M=4's 0.6039/0.5973 (baseline's own seed std is 0.0008) - more negatives
per step doesn't help further. lr in {0.0008, 0.001} (vs. the 0.0005 used
below, which Person 1 tuned for BPR specifically) gave valid 0.6033 both
times, slightly below 0.0005's 0.6039 - the BPR-tuned learning rate happens
to transfer fine here too. M=4, lr=0.0005 (this file's defaults) is the
result of that sweep, not an arbitrary first guess.
"""
import numpy as np

from baseline import FM, build_user_pos_neg
from data import encode
from evaluate import evaluate

PRIORITY = 4
DESCRIPTION = 'Listwise softmax loss (1 positive vs M=4 random negatives per user) instead of pairwise BPR.'
AUTHOR = 'agent'


def run_fm_listwise(splits, k=16, lr=0.0005, epochs=40, bs=8192, patience=4,
                     seed=0, m_neg=4, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    flat_pos, neg_by_user = build_user_pos_neg(utr, ytr)
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(flat_pos))
        losses = []
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
            losses.append(float(-np.log(p[:, 0] + 1e-9).mean()))
            coef = p.copy()
            coef[:, 0] -= 1.0
            coef = (coef / B).astype(np.float32)
            m._apply_grad(X, E, S, coef.reshape(-1))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f}")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


def run(splits):
    return run_fm_listwise(splits, verbose=False)
