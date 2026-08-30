"""Autonomous agent iteration - multi-task learning (Person 3's assigned idea
from PLAN.md, built agent-side after coordinating with the team).

PLAN.md's idea: "is_click, is_like, is_follow, is_comment, is_forward,
play_time_ms are all sitting in the logs unused as auxiliary signals for the
long_view main task." Checked the actual positive rates in the raw log before
picking one (sampled 300k rows of log_standard_4_08_to_4_21_pure.csv):

    is_click   45.59%      <- dense, chosen as the auxiliary task
    long_view  33.07%      (main task)
    is_like     1.77%
    is_comment  0.24%
    is_follow   0.09%
    is_forward  0.09%
    is_hate     0.04%

is_like/follow/comment/forward are all under 2% positive - the primer
appendix's own "data sparsity" warning applies directly to those, they'd
mostly add noise at this row count. is_click is the only auxiliary signal
dense enough to plausibly help, and it's a natural precursor to long_view
(you can't long-view something you didn't click), so shared representations
between the two tasks are a reasonable bet.

Architecture: ESMM-style shared-embeddings/separate-heads, per PLAN.md's own
description. Copies baseline.py's FM and gives it a second linear head
(W_aux, b_aux) for is_click while keeping the embedding table V fully shared -
gradients from both losses flow into V, only their own loss updates each
head's linear weights. aux_weight=0.3 is a reasonable ESMM-typical default,
not tuned - flagging that honestly rather than presenting it as optimized.

data.py's load()/encode() don't carry is_click (not part of the 5-field
contract), so this file re-reads the same two log CSVs itself rather than
editing data.py, same "own file, no shared edits" convention experiments/
README.md asks for. Assumes the default ./KuaiRand-Pure/data path, matching
every other script in this repo.
"""
import csv
import os

import numpy as np

from baseline import sigmoid
from data import SPLITS, LABEL, encode
from evaluate import evaluate

PRIORITY = 7     # right after user_history.py - same reasoning, see that
                 # file's PRIORITY comment
DESCRIPTION = ('Multi-task FM: shared embeddings, separate linear heads for '
               'long_view (main) and is_click (auxiliary, aux_weight=0.3).')
AUTHOR = 'agent'


def _load_with_click(data_dir='./KuaiRand-Pure/data'):
    """Mirrors data.load() exactly, plus is_click. Duplicated rather than
    imported since data.load() has no is_click column and shouldn't be
    edited for one experiment's needs."""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             1 if r['is_click'] != '0' else 0))
    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


class MultiTaskFM:
    """Same FM interaction term as baseline.py, shared across two linear
    heads. See baseline.FM for the single-task version this extends."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, aux_weight=0.3):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)   # shared
        self.W_main = np.zeros(dim, dtype=np.float32); self.b_main = np.float32(0.0)
        self.W_aux = np.zeros(dim, dtype=np.float32); self.b_aux = np.float32(0.0)
        self.lr, self.l2, self.aux_weight = lr, l2, aux_weight
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW_main = np.zeros_like(self.W_main); self.vW_main = np.zeros_like(self.W_main)
        self.mW_aux = np.zeros_like(self.W_aux); self.vW_aux = np.zeros_like(self.W_aux)
        self.t = 0

    def _interaction(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return inter, E, S

    def logits_main(self, X):
        inter, E, S = self._interaction(X)
        return self.b_main + self.W_main[X].sum(1) + inter, E, S

    def _adam(self, P, G, M, Vv):
        b1, b2, eps = 0.9, 0.999, 1e-8
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def step(self, X, y_main, y_aux):
        B = len(y_main)
        inter, E, S = self._interaction(X)
        z_main = self.b_main + self.W_main[X].sum(1) + inter
        z_aux = self.b_aux + self.W_aux[X].sum(1) + inter
        coef_main = ((sigmoid(z_main) - y_main) / B).astype(np.float32)
        coef_aux = ((sigmoid(z_aux) - y_aux) / B).astype(np.float32) * self.aux_weight
        coef_shared = coef_main + coef_aux   # both losses flow into the shared V

        gV = np.zeros_like(self.V)
        np.add.at(gV, X, coef_shared[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V

        gW_main = np.zeros_like(self.W_main)
        np.add.at(gW_main, X, coef_main[:, None])
        gW_main += self.l2 * self.W_main

        gW_aux = np.zeros_like(self.W_aux)
        np.add.at(gW_aux, X, coef_aux[:, None])
        gW_aux += self.l2 * self.W_aux

        self.t += 1
        self._adam(self.V, gV, self.mV, self.vV)
        self._adam(self.W_main, gW_main, self.mW_main, self.vW_main)
        self._adam(self.W_aux, gW_aux, self.mW_aux, self.vW_aux)
        self.b_main -= self.lr * coef_main.sum()
        self.b_aux -= self.lr * coef_aux.sum()

        return float(-np.mean(y_main * np.log(sigmoid(z_main) + 1e-9) +
                               (1 - y_main) * np.log(1 - sigmoid(z_main) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits_main(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def run_fm_multitask(splits_with_click, k=16, lr=0.001, aux_weight=0.3, epochs=40,
                      bs=8192, patience=4, seed=0, verbose=True):
    enc, dim = encode(splits_with_click)   # ignores the extra is_click element, still works
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    yclick_tr = np.array([x[7] for x in splits_with_click['train']], dtype=np.float32)

    m = MultiTaskFM(dim, k=k, lr=lr, seed=seed, aux_weight=aux_weight)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = []
        for i in range(0, len(idx), bs):
            b_idx = idx[i:i + bs]
            losses.append(m.step(Xtr[b_idx], ytr[b_idx], yclick_tr[b_idx]))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f}")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W_main.copy(), np.float32(m.b_main))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W_main, m.b_main = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


def run(splits):
    d = _load_with_click()   # splits arg has no is_click column - reload with it
    return run_fm_multitask(d, verbose=False)
