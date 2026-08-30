"""Autonomous agent iteration (proposed + written live by Claude, AUTHOR='agent').

CONTEXT
-------
Person 2 has already shown that generic pooled user-history vectors (uniform
mean AND DIN-style candidate-aware attention over the last-N video embeddings)
do NOT beat the FM baseline on validation, on either the pointwise or the BPR
loss. Instruction for this iteration: do not try more pooled-history variants;
find a sequence signal that carries information the static `user_id` embedding
cannot, with broad enough eval coverage to actually move a within-user ranking
metric.

WHY MOST SEQUENCE FEATURES CAN'T HELP HERE (measured, not assumed)
-----------------------------------------------------------------
The train/valid split is temporal, so valid candidates are mostly fresh items:
  * candidate video seen anywhere in the user's train history : 1.6% of valid rows
  * candidate author seen anywhere in the user's train history: 3.4% of valid rows
Any "re-exposure / repeat / last-outcome" feature is therefore capped at ~3%
coverage and cannot shift validation primary by 0.002 on a within-user metric.
Embedding-similarity-to-history has full coverage but is the exact family that
already came back flat.

HYPOTHESIS (this experiment)
---------------------------
`dur_bucket` (video length, 10 levels) is a low-cardinality taste axis with
FULL candidate coverage. The FM already learns each user's *long-term* length
preference via the user_id x dur_bucket interaction. What it cannot represent
is *short-term drift*: "this user has been long-viewing longer videos than
usual in their most recent sessions". Feature = for the candidate's dur_bucket,
the user's long_view rate within a trailing window of their last K=15
interactions, minus their overall long_view rate in that same window. This
delta is candidate-varying (so it contributes to within-user ranking), has
~75% coverage on validation, and is a pure recent-outcome statistic on a
coarse attribute rather than a pooled content embedding.

Probe on validation (no training): among covered rows, long_view rate is
0.282 for "recently cold on this length" vs 0.365 for "recently hot",
0.305 neutral -> there is a within-user gradient to try to capture.

Injected as one extra bucketised categorical FM field (9 levels: no-history /
insufficient-window / dur-absent-from-window / 6 signed-delta bins). Everything
else (encode, evaluate, FM hyper-params, early-stop rule) is byte-for-byte the
pointwise FM baseline, so the validation number is directly comparable to the
official 0.6016.

DECISION RULE: keep only if valid primary gain >= 0.002 over the same-seed
baseline; otherwise log as an honest negative and stop.

RESULTS (seed 0, same-seed pointwise FM baseline; official valid primary 0.6016)
------------------------------------------------------------------------------
                  valid GAUC  valid nDCG@5  valid primary   test primary
  baseline FM        0.6671      0.5358        0.6015           0.5953
  + dur_drift field  0.6675      0.5358        0.6017 (+0.0002) 0.5944 (-0.0009)

DECISION: DISCARD. Valid gain +0.0002 is an order of magnitude below the
0.002 gate (and inside the 0.0008 FM seed std). Test moved -0.0009. Stop;
do not chain a follow-up variant.

REFLECTION (for the autonomous agent)
-------------------------------------
* The within-user drift signal is real but tiny: the probe showed an
  8-point long_view-rate spread across drift bins, yet it converts to only
  +0.0002 primary. Reason: the FM's user_id x dur_bucket interaction, with a
  median of 31 interactions/user, already fits each user's length preference
  well; the *recent deviation* from it is mostly noise at K=15, and dur_bucket
  is too coarse (10 levels) for "which length" to reorder a slate much.
* This now closes the practical space for Person 2 on the FM baseline:
  pooled-embedding history (mean/attention) = flat; re-exposure features =
  <3.4% coverage, structurally can't move the metric; low-cardinality
  outcome-drift (this) = +0.0002. All three failure modes are distinct and
  all three are now measured.
* The remaining theoretical lever is a *trained* sequence encoder (its own
  params, gradient through the pooling, e.g. GRU/self-attention over the
  history with the candidate as query) rather than any hand-computed summary
  fed to the FM. That is a large build with low prior (the non-parametric
  attention proxy was already flat) and should rank below the loss-function
  and multi-task directions, which have shown real movement (BPR +0.0032
  test over pointwise).
* Recommended next action for the loop: stop spending iterations on user
  history; the highest-EV unexplored lever is listwise/softmax loss or
  multi-task auxiliary heads.
"""
import collections
import time

import numpy as np

from baseline import FM
from data import encode
from evaluate import evaluate

PRIORITY = 3
DESCRIPTION = ('Short-term duration-preference drift: recent long_view rate for the '
               'candidate video-length minus recent baseline, as an extra FM field.')
AUTHOR = 'agent'

K_WINDOW = 15          # trailing interactions that count as "recent"
MIN_HIST = 5           # need at least this many prior interactions to trust the window
_DELTA_EDGES = np.array([-0.35, -0.15, -0.03, 0.03, 0.15, 0.35])   # -> 7 signed bins
# category codes: 0 no history, 1 window < MIN_HIST, 2 candidate dur absent from window,
#                 3..9 signed-delta bin (len(_DELTA_EDGES)+1 = 7 bins)
_N_CODES = 3 + (len(_DELTA_EDGES) + 1)


def _codes_for_window(win_dur, win_y, cand_dur):
    """win_dur/win_y: trailing window arrays (most recent last). cand_dur: int.
    Returns the category code for this (window, candidate) pair."""
    if len(win_y) == 0:
        return 0
    if len(win_y) < MIN_HIST:
        return 1
    same = win_dur == cand_dur
    if not same.any():
        return 2
    base = win_y.mean()
    delta = win_y[same].mean() - base
    return 3 + int(np.searchsorted(_DELTA_EDGES, delta))


def _build_drift_field(enc):
    """Per row -> category code (int), no future data.
    train row i : the K interactions of the same user strictly before i.
    valid/test  : that user's last K train interactions.
    """
    utr = enc['train'][2]
    dur_tr = enc['train'][0][:, 4].astype(np.int64)     # encoded dur_bucket id per train row
    y_tr = enc['train'][1].astype(np.int64)

    hist = collections.defaultdict(collections.deque)   # user -> deque[(dur, y)] (maxlen K)
    code_tr = np.zeros(len(utr), dtype=np.int64)
    for i, (u, d, y) in enumerate(zip(utr, dur_tr, y_tr)):
        w = hist[u]
        wd = np.fromiter((t[0] for t in w), dtype=np.int64, count=len(w))
        wy = np.fromiter((t[1] for t in w), dtype=np.int64, count=len(w))
        code_tr[i] = _codes_for_window(wd, wy, int(d))
        w.append((int(d), int(y)))
        if len(w) > K_WINDOW:
            w.popleft()

    user_win = {u: (np.fromiter((t[0] for t in w), dtype=np.int64, count=len(w)),
                    np.fromiter((t[1] for t in w), dtype=np.int64, count=len(w)))
                for u, w in hist.items()}

    def eval_codes(name):
        users = enc[name][2]
        dur = enc[name][0][:, 4].astype(np.int64)
        out = np.zeros(len(users), dtype=np.int64)
        for i, (u, d) in enumerate(zip(users, dur)):
            w = user_win.get(u)
            if w is None:
                out[i] = 0
            else:
                out[i] = _codes_for_window(w[0], w[1], int(d))
        return out

    return {'train': code_tr, 'valid': eval_codes('valid'), 'test': eval_codes('test')}


def _augment(X, codes, base_dim):
    """Append the drift code as one extra encoded field."""
    col = (codes + base_dim).astype(np.int32)[:, None]
    return np.concatenate([X, col], axis=1)


def run_fm_durdrift(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4,
                    seed=0, verbose=False):
    enc, base_dim = encode(splits)
    codes = _build_drift_field(enc)
    dim = base_dim + _N_CODES

    Xtr = _augment(enc['train'][0], codes['train'], base_dim); ytr = enc['train'][1]
    Xva = _augment(enc['valid'][0], codes['valid'], base_dim)
    yva, uva = enc['valid'][1], enc['valid'][2]
    Xte = _augment(enc['test'][0], codes['test'], base_dim)
    yte, ute = enc['test'][1], enc['test'][2]

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


def run(splits):
    return run_fm_durdrift(splits, verbose=False)


if __name__ == '__main__':
    import argparse
    from data import load, FIELDS
    from baseline import run_fm

    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    print("\n--- pointwise FM baseline (same seed) ---")
    base = run_fm(splits, seed=a.seed, verbose=True)
    print(f"\n--- FM + recent duration-drift field (K={K_WINDOW}, same seed) ---")
    res = run_fm_durdrift(splits, seed=a.seed, verbose=True)

    print("\n==================== SUMMARY (seed %d) ====================" % a.seed)
    for name, r in (('baseline', base), ('+dur_drift', res)):
        for sp in ('valid', 'test'):
            x = r[sp]
            print(f"  {name:11s} {sp:5s}  GAUC {x['GAUC']:.4f} | nDCG@5 {x['nDCG@5']:.4f} | primary {x['primary']:.4f}")
    dv = res['valid']['primary'] - base['valid']['primary']
    dt = res['test']['primary'] - base['test']['primary']
    print(f"\n  valid delta: {dv:+.4f}  (gate: keep iff >= +0.0020)  ->  {'KEEP' if dv >= 0.002 else 'DISCARD'}")
    print(f"  test  delta: {dt:+.4f}  (reported after decision, not used for it)")
