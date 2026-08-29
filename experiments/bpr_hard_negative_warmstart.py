"""Autonomous agent iteration #2 (follow-up to bpr_hard_negative.py).

Prior result: hard-negative mining from epoch 1 underperformed plain BPR
(valid primary 0.5803 vs 0.6037) - training loss stayed stuck near 0.693,
consistent with the known failure mode where hard-negative mining applied
before the model has learned anything picks negatives based on noisy,
untrained scores instead of a useful signal.

Revised hypothesis: warm-start with a few epochs of plain (uniform-random
negative) BPR first, so the model has a real notion of relevance before
switching to hard-negative mining for the remaining epochs. If this still
doesn't clear the champion, that's evidence hard-negative mining isn't
worth pursuing further on this dataset/model, at least in this form.
"""
import numpy as np

from baseline import FM, build_user_pos_neg
from data import encode
from evaluate import evaluate

PRIORITY = 3
DESCRIPTION = 'BPR: 5 warm-up epochs with random negatives, then switch to hard-negative mining (K=5).'
AUTHOR = 'agent'  # proposed and written live by Claude, not hand-coded by a teammate


def run_fm_bpr_hard_warmstart(splits, k=16, lr=0.0005, epochs=40, bs=8192, patience=4,
                               seed=0, hard_k=5, warmup_epochs=5, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    flat_pos, neg_by_user = build_user_pos_neg(utr, ytr)
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        hard = ep > warmup_epochs
        order = rng.permutation(len(flat_pos))
        losses = []
        for start in range(0, len(order), bs):
            batch = [flat_pos[j] for j in order[start:start + bs]]
            pos_rows = np.array([i for _, i in batch])
            if hard:
                cand_rows = np.stack([
                    neg_by_user[u][rng.integers(len(neg_by_user[u]), size=hard_k)]
                    for u, _ in batch
                ])
                cand_scores = m.predict(Xtr[cand_rows.reshape(-1)]).reshape(len(batch), hard_k)
                neg_rows = cand_rows[np.arange(len(batch)), cand_scores.argmax(1)]
            else:
                neg_rows = np.array([neg_by_user[u][rng.integers(len(neg_by_user[u]))] for u, _ in batch])
            losses.append(m.step_bpr(Xtr[pos_rows], Xtr[neg_rows]))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | {'hard' if hard else 'warm'} | loss {np.mean(losses):.4f} "
                  f"| valid primary {va['primary']:.4f}")
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
    return run_fm_bpr_hard_warmstart(splits, verbose=False)
