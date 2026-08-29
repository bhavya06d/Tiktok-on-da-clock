"""Shared data loading + feature engineering for every agent variant.

Row order is kept identical to the starter kit's `data.load()` (file A then
file B, date-filtered, original order preserved) so `submission_<split>.csv`
stays aligned with the official evaluator.

Leakage guard: every historical / target-encoded statistic is computed on the
TRAIN split only. Nothing from valid/test ever feeds a feature.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LABEL = "long_view"
AUX_LABELS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward"]

SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}

LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)

# high-cardinality ids are captured by target encoding, not by one-hot/native
# categorical splits (27k users / 7.5k videos makes LightGBM crawl).
CAT_FIELDS = ["tab", "dur_bucket"]
ID_FIELDS = ["user_id", "video_id", "author_id"]

_CACHE_DIR = os.path.join(
    os.environ.get("TEMP", "/tmp"), "kuairand_agent_cache")


# --------------------------------------------------------------------------- #
# raw load
# --------------------------------------------------------------------------- #
def load_raw(data_dir: str, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Return {'train','valid','test': DataFrame} with side features joined."""
    cache = os.path.join(_CACHE_DIR, "raw.pkl")
    if use_cache and os.path.exists(cache):
        try:
            return pd.read_pickle(cache)
        except Exception:  # noqa: BLE001
            pass
    out = _load_raw_uncached(data_dir)
    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        try:
            pd.to_pickle(out, cache)
        except Exception:  # noqa: BLE001
            pass
    return out


def _load_raw_uncached(data_dir: str) -> dict[str, pd.DataFrame]:
    usecols = [
        "user_id", "video_id", "date", "hourmin", "time_ms",
        "is_click", "is_like", "is_follow", "is_comment", "is_forward",
        LABEL, "play_time_ms", "duration_ms", "tab",
    ]
    frames = []
    for f in LOG_FILES:
        frames.append(pd.read_csv(os.path.join(data_dir, f), usecols=usecols))
    df = pd.concat(frames, ignore_index=True)

    vbasic = pd.read_csv(
        os.path.join(data_dir, "video_features_basic_pure.csv"),
        usecols=["video_id", "author_id", "video_type", "upload_type",
                 "music_id", "video_duration"],
    )
    df = df.merge(vbasic, on="video_id", how="left")
    df["author_id"] = df["author_id"].fillna(-1).astype(np.int64)

    users = pd.read_csv(
        os.path.join(data_dir, "user_features_pure.csv"),
        usecols=["user_id", "user_active_degree", "follow_user_num_range",
                 "fans_user_num_range", "friend_user_num_range",
                 "register_days_range"],
    )
    df = df.merge(users, on="user_id", how="left")

    vstat = pd.read_csv(
        os.path.join(data_dir, "video_features_statistic_pure.csv"),
        usecols=["video_id", "play_progress", "show_cnt", "like_cnt",
                 "complete_play_cnt", "play_cnt"],
    )
    vstat["v_like_ratio"] = vstat["like_cnt"] / vstat["play_cnt"].clip(lower=1)
    vstat["v_complete_ratio"] = (
        vstat["complete_play_cnt"] / vstat["play_cnt"].clip(lower=1)
    )
    df = df.merge(
        vstat[["video_id", "play_progress", "v_like_ratio", "v_complete_ratio",
               "show_cnt"]],
        on="video_id", how="left",
    )

    # temporal
    df["hour"] = (df["hourmin"] // 100).astype(np.int16)
    df["dow"] = (pd.to_datetime(df["date"], format="%Y%m%d").dt.dayofweek
                 .astype(np.int16))
    df["watch_ratio"] = (df["play_time_ms"] /
                         df["duration_ms"].clip(lower=1)).clip(0, 3)

    # duration bucket from TRAIN quantiles (matches starter kit spirit)
    tr_mask = df["date"].between(*SPLITS["train"])
    edges = np.quantile(df.loc[tr_mask, "duration_ms"].to_numpy(),
                        np.linspace(0, 1, 11)[1:-1])
    df["dur_bucket"] = np.searchsorted(edges, df["duration_ms"].to_numpy())

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = df[df["date"].between(lo, hi)].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# feature engineering
# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    use_stats: bool = True          # smoothed target encodings
    use_temporal: bool = True       # hour / dow
    use_side: bool = True           # user + video side features
    use_history: bool = False       # per-user behavioural history features
    hist_last_n: int = 20
    smoothing: float = 20.0


@dataclass
class FeatureBundle:
    Xtr: pd.DataFrame
    Xev: pd.DataFrame
    feats: list[str]
    cat_feats: list[str]
    id_feats: list[str]
    ytr: np.ndarray
    yev: np.ndarray
    aux_tr: dict[str, np.ndarray] = field(default_factory=dict)
    group_tr: np.ndarray | None = None   # per-user group sizes (train, sorted)
    tr_sort_idx: np.ndarray | None = None


def _smooth_rate(series_sum, series_cnt, prior, m):
    return (series_sum + m * prior) / (series_cnt + m)


def build_features(raw: dict, split: str, cfg: FeatureConfig) -> FeatureBundle:
    train = raw["train"].copy()
    ev = raw[split].copy()
    prior = float(train[LABEL].mean())
    m = cfg.smoothing

    # code every id/categorical field: fit the mapping on TRAIN, unseen -> -1
    for c in CAT_FIELDS + ID_FIELDS:
        codes, uniques = pd.factorize(train[c], sort=True)
        train[c + "_code"] = codes.astype(np.int64)
        lut = {v: i for i, v in enumerate(uniques)}
        ev[c + "_code"] = ev[c].map(lut).fillna(-1).astype(np.int64)
    cat_feats = [c + "_code" for c in CAT_FIELDS]              # LGBM native cat
    id_feats = [c + "_code" for c in CAT_FIELDS + ID_FIELDS]   # torch embeddings
    feats = list(cat_feats)                                    # LGBM features

    if cfg.use_stats:
        # leave-one-out target encoding: for a train row the row's own label is
        # removed from its group stat, so the encoding can't leak the target.
        # eval rows get the full-group stat. This is the difference between
        # "helps" and "destroys the ranker" for the sparse cross keys.
        for keys, name in [(["user_id"], "u"), (["video_id"], "v"),
                           (["author_id"], "a"), (["user_id", "author_id"], "ua"),
                           (["user_id", "tab"], "ut"),
                           (["user_id", "dur_bucket"], "ud")]:
            gkey = keys[0] if len(keys) == 1 else keys
            g = train.groupby(gkey)[LABEL].agg(["sum", "count"])
            if len(keys) == 1:
                tsum = train[keys[0]].map(g["sum"]).to_numpy()
                tcnt = train[keys[0]].map(g["count"]).to_numpy()
                ecnt_series = ev[keys[0]].map(g["count"])
                esum_series = ev[keys[0]].map(g["sum"])
            else:
                tk = pd.MultiIndex.from_frame(train[keys])
                ek = pd.MultiIndex.from_frame(ev[keys])
                tsum = g["sum"].reindex(tk).to_numpy()
                tcnt = g["count"].reindex(tk).to_numpy()
                esum_series = g["sum"].reindex(ek)
                ecnt_series = g["count"].reindex(ek)
            y = train[LABEL].to_numpy()
            train[f"{name}_rate"] = ((tsum - y + m * prior) /
                                     (tcnt - 1 + m))
            ev[f"{name}_rate"] = ((esum_series.to_numpy() + m * prior) /
                                  (ecnt_series.to_numpy() + m))
            ev[f"{name}_rate"] = pd.Series(ev[f"{name}_rate"],
                                           index=ev.index).fillna(prior)
            feats.append(f"{name}_rate")
            if len(keys) == 1:
                for df, cnt in ((train, tcnt), (ev, ecnt_series.to_numpy())):
                    df[f"{name}_cnt"] = np.log1p(np.nan_to_num(cnt))
                feats.append(f"{name}_cnt")

        # per-video mean watch ratio (LOO on train)
        gw = train.groupby("video_id")["watch_ratio"].agg(["sum", "count"])
        wsum = train["video_id"].map(gw["sum"]).to_numpy()
        wcnt = train["video_id"].map(gw["count"]).to_numpy()
        wr_prior = float(train["watch_ratio"].mean())
        train["v_watch_ratio_te"] = ((wsum - train["watch_ratio"].to_numpy()
                                      + 5 * wr_prior) / (wcnt - 1 + 5))
        ev["v_watch_ratio_te"] = ev["video_id"].map(
            gw["sum"] / gw["count"]).fillna(wr_prior)
        feats.append("v_watch_ratio_te")

    if cfg.use_temporal:
        for df in (train, ev):
            pass
        feats += ["hour", "dow"]

    if cfg.use_side:
        side_num = ["video_duration", "play_progress", "v_like_ratio",
                    "v_complete_ratio", "show_cnt"]
        for c in side_num:
            for df in (train, ev):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        feats += side_num
        for c in ["user_active_degree", "follow_user_num_range",
                  "fans_user_num_range", "register_days_range"]:
            cats = pd.Categorical(train[c].astype(str)).categories
            for df in (train, ev):
                df[c + "_code"] = pd.Categorical(
                    df[c].astype(str), categories=cats).codes
            feats.append(c + "_code")
            cat_feats.append(c + "_code")

    if cfg.use_history:
        _add_history_features(train, ev, cfg, feats)

    aux_tr = {a: train[a].to_numpy(np.float32) for a in AUX_LABELS
              if a in train.columns}
    aux_tr["watch_ratio"] = train["watch_ratio"].to_numpy(np.float32)

    # group by user for ranking objectives
    tr_sorted = train.sort_values("user_id", kind="stable")
    sort_idx = tr_sorted.index.to_numpy()
    group_tr = tr_sorted.groupby("user_id", sort=False).size().to_numpy()

    keep = sorted(set(feats) | set(id_feats))
    return FeatureBundle(
        Xtr=train[keep], Xev=ev[keep], feats=feats, cat_feats=cat_feats,
        id_feats=id_feats,
        ytr=train[LABEL].to_numpy(np.float32),
        yev=ev[LABEL].to_numpy(np.float32),
        aux_tr=aux_tr, group_tr=group_tr, tr_sort_idx=sort_idx,
    )


def _add_history_features(train, ev, cfg, feats):
    """Behavioural history features (DIN-lite): what each user did BEFORE,
    computed from train only, ordered by time_ms."""
    t = train.sort_values(["user_id", "time_ms"], kind="stable")

    # expanding (all prior interactions) long_view rate — shift(1) => no leak
    grp = t.groupby("user_id", sort=False)
    t["u_hist_lv_rate"] = (grp[LABEL].cumsum().shift(1) /
                           grp.cumcount().replace(0, np.nan))
    t["u_hist_watch_mean"] = (grp["watch_ratio"].cumsum().shift(1) /
                              grp.cumcount().replace(0, np.nan))
    t["u_hist_len"] = np.log1p(grp.cumcount())
    # recent window
    t["u_recent_lv_rate"] = (
        grp[LABEL].transform(
            lambda s: s.shift(1).rolling(cfg.hist_last_n, min_periods=1).mean())
    )
    prior = float(train[LABEL].mean())
    t = t[["u_hist_lv_rate", "u_hist_watch_mean", "u_hist_len",
           "u_recent_lv_rate"]].reindex(train.index)
    for c in t.columns:
        train[c] = t[c].fillna(prior if "rate" in c else 0.0)

    # final-state snapshot per user -> map onto eval rows
    final = train.groupby("user_id").agg(
        u_hist_lv_rate=(LABEL, "mean"),
        u_hist_watch_mean=("watch_ratio", "mean"),
        u_hist_len=(LABEL, "count"),
    )
    ev["u_hist_lv_rate"] = ev["user_id"].map(final["u_hist_lv_rate"]).fillna(prior)
    ev["u_hist_watch_mean"] = ev["user_id"].map(
        final["u_hist_watch_mean"]).fillna(train["watch_ratio"].mean())
    ev["u_hist_len"] = np.log1p(
        ev["user_id"].map(final["u_hist_len"]).fillna(0))
    ev["u_recent_lv_rate"] = ev["user_id"].map(
        final["u_hist_lv_rate"]).fillna(prior)

    # author affinity: has this user long-viewed this author before?
    ua = (train[train[LABEL] == 1].groupby(["user_id", "author_id"]).size()
          .rename("ua_pos").reset_index())
    for df in (train, ev):
        merged = df.merge(ua, on=["user_id", "author_id"], how="left")
        df["u_author_affinity"] = np.log1p(merged["ua_pos"].fillna(0).to_numpy())

    feats += ["u_hist_lv_rate", "u_hist_watch_mean", "u_hist_len",
              "u_recent_lv_rate", "u_author_affinity"]
