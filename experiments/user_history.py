"""Person 2 — User interaction history / sequence features.

README headroom idea #2 ("用户历史序列"). For every example we build that
user's history from ONLY interactions that happened before it:

  * train row i : that user's earlier train rows (train is in chronological
    file order); the current row is appended AFTER its history is snapshotted.
  * valid/test  : the user's last N train interactions (all of train predates
    valid/test), so there is no future-data leakage.

Two history representations live here:

  mode='mean'  (v1, simplest) : h_u = mean of the FM video embeddings of the
      user's last N interacted videos. A single per-user interest vector.

  mode='attn'  (v2, DIN-style): h_u,candidate = sum_j softmax_j(<v_j, v_cand>
      / sqrt(k)) * v_j  — a candidate-AWARE weighted average of the history,
      i.e. "how much does this candidate look like the videos this user
      actually engaged with". Optional exponential recency bias (`decay`).
      No new parameters — reuses the FM's own video embeddings.

The history vector is injected as one extra term in the FM interaction sum,
so it crosses with the current video (dot(h, v_candidate)). Per README a
pure user-side term is a within-user constant and cannot change ranking —
the signal only comes from that history x item cross. h is stop-gradient
(constant for the Adam step); gradients still flow into every real field
embedding, keeping the update rule identical to baseline FM.

The feature can sit on top of either the pointwise BCE loss (loss='point',
= baseline) or a pairwise BPR loss (loss='bpr', copied from baseline.py) —
BPR is where candidate-aware attention has the most room to help, since the
pos/neg videos in a pair produce different h vectors for the same user.

data.encode, evaluate.evaluate, the early-stopping rule and the FM
hyper-params are byte-for-byte the baseline so scores are directly comparable.

RESULTS (seed 0, same-seed baselines; FM seed std ~0.0008):

  pointwise BCE loss
    baseline FM ................... valid 0.6015   test 0.5953
    + history mean (N=50) ......... valid 0.6010   test 0.5952   (-0.0001)
    + history mean (N=50, pos) .... valid 0.6015   test 0.5953   ( 0.0000)
    + history attn (N=50) ......... valid 0.6010   test 0.5953   (-0.0000)
  pairwise BPR loss
    fm_bpr (no history) .......... valid 0.6037   test 0.5985
    + history attn (N=50) ........ valid 0.6037   test 0.5977   (-0.0008)

Conclusion: neither history representation (uniform mean or DIN-style
candidate-aware attention), on either loss, beats the same-loss baseline.
Under BPR the history term is a slight net drag (-0.0008, ~1 std). The
real lever in this comparison is the BPR loss itself (+0.0032 test over
pointwise), which is Person 1's idea, not history. Interpretation: the
FM's user_id embedding already encodes the user's average taste, so an
averaged / soft-attended history vector built from the same video
embeddings is close to redundant with it; only per-item history signals
the FM cannot already reconstruct (recency, specific repeat-watch, true
sequence order) would add anything, and those need a learned history
encoder rather than a stop-gradient pool.

Run:  python -m experiments.user_history --pool attn --bpr
"""
import collections
import time

import numpy as np

from data import encode
from evaluate import evaluate

PRIORITY = 6   # placed after the agent's own earlier exploration (bpr_hard_negative*,
               # listwise_softmax, hour_of_day) and merged in alongside multitask.py -
               # see agent.py's converged_at note on why the loop no longer hard-stops
               # before reaching genuinely later-added ideas
DESCRIPTION = ('User history: DIN-style candidate-aware attention over the user\'s '
               'last-N pre-example videos, crossed with the current video in the FM.')
AUTHOR = 'human'

N_HISTORY = 50
_SEED = 0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------- history construction (no future data) ----------------
def _build_history_ids(enc, n, pos_only=False):
    """Per row -> encoded video ids of that user's last <=n PRIOR interactions.
    Left-packed: column 0 is the oldest in the window, column cnt-1 the most
    recent. Returns {split: (H int32 (rows, n), cnt int32 (rows,))}."""
    tr_users = enc['train'][2]
    tr_vid = enc['train'][0][:, 1].astype(np.int32)     # encoded video id per train row
    tr_y = enc['train'][1]

    seen = collections.defaultdict(list)
    Htr = np.zeros((len(tr_users), n), dtype=np.int32)
    Ctr = np.zeros(len(tr_users), dtype=np.int32)
    for i, (u, v, yi) in enumerate(zip(tr_users, tr_vid, tr_y)):
        past = seen[u]
        if past:
            take = past[-n:]
            Htr[i, :len(take)] = take
            Ctr[i] = len(take)
        if not pos_only or yi > 0:
            past.append(int(v))

    user_full = {u: np.asarray(lst[-n:], dtype=np.int32) for u, lst in seen.items()}

    def eval_split(name):
        users = enc[name][2]
        H = np.zeros((len(users), n), dtype=np.int32)
        C = np.zeros(len(users), dtype=np.int32)
        for i, u in enumerate(users):
            h = user_full.get(u)
            if h is not None and len(h):
                H[i, :len(h)] = h
                C[i] = len(h)
        return H, C

    return {'train': (Htr, Ctr), 'valid': eval_split('valid'), 'test': eval_split('test')}


def _hist_vec(V, Hb, cnt_b, q_ids=None, mode='mean', temp=None, decay=0.0):
    """History pooling. (B,n) padded ids -> (B,k) vector.

    mode='mean' : uniform average over the real history.
    mode='attn' : softmax(<v_j, v_query> / temp) weights, optional exp recency
                  bias `decay` on item age (0 = most recent real item).
    Rows with no history return a zero vector.
    """
    n = Hb.shape[1]
    mask = np.arange(n)[None, :] < cnt_b[:, None]            # (B,n) bool
    emb = V[Hb]                                              # (B,n,k)
    if mode == 'mean':
        w = mask.astype(np.float32)
    else:
        q = V[q_ids]                                         # (B,k)
        t = temp if temp else np.sqrt(emb.shape[2])
        logit = np.einsum('bnk,bk->bn', emb, q) / t          # (B,n)
        if decay > 0:
            age = cnt_b[:, None] - 1 - np.arange(n)[None, :]
            logit = logit - decay * np.clip(age, 0, None)
        logit = np.where(mask, logit, -1e30)
        logit = logit - logit.max(1, keepdims=True)
        w = np.exp(logit) * mask
    denom = w.sum(1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    w = (w / denom).astype(np.float32)
    return np.einsum('bn,bnk->bk', w, emb).astype(np.float32)


class FM:
    """Baseline FM (copied from baseline.py) + optional stop-gradient history
    vector `h`, added as one more term in the FM sum."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X, h=None):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        sq = (E ** 2).sum((1, 2))                       # (B,)
        if h is not None:
            S = S + h
            sq = sq + (h ** 2).sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - sq)
        return self.b + self.W[X].sum(1) + inter, E, S

    def _apply_grad(self, X, E, S, coef):
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, coef[:, None])
        np.add.at(gV, X, coef[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * coef.sum()

    def step(self, X, y, h=None):
        B = len(y)
        z, E, S = self.logits(X, h)
        coef = ((sigmoid(z) - y) / B).astype(np.float32)
        self._apply_grad(X, E, S, coef)
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def step_bpr(self, Xpos, Xneg, hpos=None, hneg=None):
        B = len(Xpos)
        X = np.concatenate([Xpos, Xneg], axis=0)
        h = None if hpos is None else np.concatenate([hpos, hneg], axis=0)
        z, E, S = self.logits(X, h)
        diff = z[:B] - z[B:]
        g = ((sigmoid(diff) - 1.0) / B).astype(np.float32)
        coef = np.concatenate([g, -g])
        self._apply_grad(X, E, S, coef)
        return float(-np.mean(np.log(sigmoid(diff) + 1e-9)))

    def predict(self, X, H=None, cnt=None, pool=None, bs=100_000):
        out = []
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]
            hb = None
            if H is not None:
                mode = pool['mode'] if pool else 'mean'
                q = xb[:, 1] if mode != 'mean' else None
                hb = _hist_vec(self.V, H[i:i + bs], cnt[i:i + bs], q_ids=q, mode=mode,
                               temp=(pool or {}).get('temp'), decay=(pool or {}).get('decay', 0.0))
            out.append(self.logits(xb, hb)[0])
        return np.concatenate(out)


def _build_user_pos_neg(users, y):
    """Copied from baseline.build_user_pos_neg — per-user long_view 1/0 row pools."""
    pos_by_user, neg_by_user = collections.defaultdict(list), collections.defaultdict(list)
    for i, (u, yi) in enumerate(zip(users, y)):
        (pos_by_user if yi > 0 else neg_by_user)[u].append(i)
    eligible = [u for u in pos_by_user if u in neg_by_user]
    flat_pos = [(u, i) for u in eligible for i in pos_by_user[u]]
    neg_by_user = {u: np.array(neg_by_user[u]) for u in eligible}
    return flat_pos, neg_by_user


# ---------------- training ----------------
def run_fm_hist(splits, n=N_HISTORY, pool_mode='attn', temp=None, decay=0.0,
                loss='point', k=16, lr=None, epochs=40, bs=8192, patience=4,
                seed=_SEED, pos_only=False, history=True, verbose=False):
    if lr is None:
        lr = 0.001 if loss == 'point' else 0.0005
    enc, dim = encode(splits)
    hist = _build_history_ids(enc, n, pos_only=pos_only) if history else None
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    if history:
        Htr, Ctr = hist['train']; Hva, Cva = hist['valid']; Hte, Cte = hist['test']
    pool = {'mode': pool_mode, 'temp': temp, 'decay': decay} if history else None

    def hbatch(rows, q_rows):
        if not history:
            return None
        q = Xtr[q_rows, 1] if pool_mode != 'mean' else None
        return _hist_vec(m.V, Htr[rows], Ctr[rows], q_ids=q, mode=pool_mode, temp=temp, decay=decay)

    def vpred():
        return m.predict(Xva, Hva if history else None, Cva if history else None, pool)

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    if loss == 'bpr':
        flat_pos, neg_by_user = _build_user_pos_neg(utr, ytr)

    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        if loss == 'point':
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                losses.append(m.step(Xtr[b], ytr[b], h=hbatch(b, b)))
        else:
            order = rng.permutation(len(flat_pos))
            for s in range(0, len(order), bs):
                batch = [flat_pos[j] for j in order[s:s + bs]]
                pr = np.array([i for _, i in batch])
                nr = np.array([neg_by_user[u][rng.integers(len(neg_by_user[u]))] for u, _ in batch])
                losses.append(m.step_bpr(Xtr[pr], Xtr[nr], hbatch(pr, pr), hbatch(nr, nr)))
        va = evaluate(uva, yva, vpred())
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
    return {'valid': evaluate(uva, yva, vpred()),
            'test':  evaluate(ute, yte, m.predict(Xte, Hte if history else None,
                                                  Cte if history else None, pool))}


def run(splits):
    """Experiment-contract entry point: strongest Person-2 config."""
    return run_fm_hist(splits, pool_mode='attn', loss='point', verbose=False)


if __name__ == '__main__':
    import argparse
    from data import load, FIELDS

    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--n', type=int, default=N_HISTORY)
    ap.add_argument('--pool', default='attn', choices=['mean', 'attn'])
    ap.add_argument('--decay', type=float, default=0.0, help='exp recency bias on attn weights')
    ap.add_argument('--seed', type=int, default=_SEED)
    ap.add_argument('--pos_only', action='store_true')
    ap.add_argument('--bpr', action='store_true', help='also run the BPR-loss configs')
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    results = []

    print("\n--- [point] baseline FM (no history) ---")
    results.append(('point  baseline',
                    run_fm_hist(splits, loss='point', history=False, seed=a.seed, verbose=True)))

    print(f"\n--- [point] + history {a.pool} (N={a.n}, decay={a.decay}) ---")
    results.append((f'point  +hist {a.pool}',
                    run_fm_hist(splits, loss='point', pool_mode=a.pool, decay=a.decay,
                                n=a.n, pos_only=a.pos_only, seed=a.seed, verbose=True)))

    if a.bpr:
        print("\n--- [bpr] fm_bpr (no history) ---")
        results.append(('bpr    baseline',
                        run_fm_hist(splits, loss='bpr', history=False, seed=a.seed, verbose=True)))
        print(f"\n--- [bpr] + history {a.pool} (N={a.n}, decay={a.decay}) ---")
        results.append((f'bpr    +hist {a.pool}',
                        run_fm_hist(splits, loss='bpr', pool_mode=a.pool, decay=a.decay,
                                    n=a.n, pos_only=a.pos_only, seed=a.seed, verbose=True)))

    print("\n==================== SUMMARY (seed %d) ====================" % a.seed)
    for name, r in results:
        print(f"  {name:20s}  valid {r['valid']['primary']:.4f}   "
              f"test {r['test']['primary']:.4f}  "
              f"(GAUC {r['test']['GAUC']:.4f} nDCG@5 {r['test']['nDCG@5']:.4f})")
    base_t = results[0][1]['test']['primary']
    print(f"\n  point delta vs baseline: {results[1][1]['test']['primary'] - base_t:+.4f}")
    if a.bpr:
        bpr_b = results[2][1]['test']['primary']
        print(f"  bpr   delta vs bpr-baseline: {results[3][1]['test']['primary'] - bpr_b:+.4f}")
        print(f"  bpr+hist vs point-baseline:  {results[3][1]['test']['primary'] - base_t:+.4f}")
