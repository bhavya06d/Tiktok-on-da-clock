"""Autonomous agent iteration (live, not human-authored by a team member).

Hypothesis: the current champion (bpr_loss) trains on pairs of
(positive, negative) where the negative is picked uniformly at random from
the user's non-viewed videos. Most random negatives are "easy" - the model
already scores them low, so the gradient signal is weak. Hard-negative
mining (Rendle et al., BPR; widely used since) instead samples a small
pool of K random candidate negatives per step and trains against whichever
one the model currently scores *highest* - the hardest to distinguish from
the positive - giving a stronger, more informative gradient each step.

This is a natural next step on top of bpr_loss, not yet tried by the team
or the organizers. If it doesn't clear the eps bar, that's still a useful,
honest negative result for the log.
"""
import numpy as np

from baseline import FM, build_user_pos_neg
from data import encode
from evaluate import evaluate

PRIORITY = 2
DESCRIPTION = 'BPR with hard-negative mining (K=5 candidates, pick highest-scoring) instead of uniform random negatives.'
AUTHOR = 'agent'  # proposed and written live by Claude, not hand-coded by a teammate


def run_fm_bpr_hard(splits, k=16, lr=0.0005, epochs=40, bs=8192, patience=4,
                     seed=0, hard_k=5, verbose=True):
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
            pos_rows = np.array([i for _, i in batch])
            # K random candidate negatives per positive, drawn from that user's pool
            cand_rows = np.stack([
                neg_by_user[u][rng.integers(len(neg_by_user[u]), size=hard_k)]
                for u, _ in batch
            ])  # (B, hard_k)
            cand_scores = m.predict(Xtr[cand_rows.reshape(-1)]).reshape(len(batch), hard_k)
            hardest = cand_rows[np.arange(len(batch)), cand_scores.argmax(1)]
            losses.append(m.step_bpr(Xtr[pos_rows], Xtr[hardest]))
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
    return run_fm_bpr_hard(splits, verbose=False)
