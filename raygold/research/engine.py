"""Vectorised configuration sweep.

For every entry variant we build its setup events once, simulate each
exit spec over all events once (numba, parallel), and then evaluate every
filter mask in a single parallel numba pass that enforces one-position-
at-a-time and accumulates per-calendar-month statistics.  Any window
(train, validation, walk-forward folds) is then a sum over months, which
makes 100k-configuration studies cheap.

The sweep never receives data at or after the test start date: the caller
slices the frame before features are built.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit, prange

from raygold.backtest.simulator import Costs, ExitSpec, simulate
from raygold.features import Features
from raygold.research.grid import ExitCoord, FilterSpec
from raygold.signals.events import Variant
from raygold.signals.multitf import events_for
from raygold.signals.scoring import ScoreWeights, history_scores, score

log = logging.getLogger(__name__)

# monthly stat channels
S_N, S_WIN, S_SUM, S_SQ, S_GP, S_GL = range(6)
N_STATS = 6

REFERENCE_EXIT = ExitSpec(sl_mode=1, sl_param=1.0, tp_mode=0, tp_param=1.0, max_hold=120)


@njit(cache=True, parallel=True)
def eval_filters(entry_idx, exit_idx, r, valid, masks, month_pos, n_months):
    """One-position-at-a-time evaluation of many filter masks.

    Events must be sorted by ``entry_idx``.  Returns ``stats[f, month, 6]``
    and ``dd[f]`` (max drawdown in R over the full sequence).
    """
    nf = masks.shape[0]
    n = r.shape[0]
    stats = np.zeros((nf, n_months, 6), np.float64)
    dd = np.zeros(nf)
    for fi in prange(nf):
        last_exit = -1
        cum = 0.0
        peak = 0.0
        mdd = 0.0
        for k in range(n):
            if not valid[k] or not masks[fi, k]:
                continue
            if entry_idx[k] <= last_exit:
                continue
            last_exit = exit_idx[k]
            m = month_pos[k]
            x = r[k]
            stats[fi, m, 0] += 1.0
            if x > 1e-9:
                stats[fi, m, 1] += 1.0
                stats[fi, m, 4] += x
            elif x < -1e-9:
                stats[fi, m, 5] -= x
            stats[fi, m, 2] += x
            stats[fi, m, 3] += x * x
            cum += x
            if cum > peak:
                peak = cum
            if peak - cum > mdd:
                mdd = peak - cum
        dd[fi] = mdd
    return stats, dd


@njit(cache=True)
def select_sequential(entry_idx, exit_idx, mask):
    """Indices of events taken under one-position-at-a-time."""
    out = np.empty(entry_idx.shape[0], np.int64)
    c = 0
    last_exit = -1
    for k in range(entry_idx.shape[0]):
        if mask[k] and entry_idx[k] > last_exit:
            out[c] = k
            c += 1
            last_exit = exit_idx[k]
    return out[:c]


def attach_scores(f: Features, ev: pd.DataFrame, costs: Costs, weights: ScoreWeights = ScoreWeights()) -> pd.DataFrame:
    """Add the past-only 'historical performance' component and the confidence score."""
    if len(ev) == 0:
        ev["hist_score"] = []
        ev["score"] = []
        return ev
    n_bars = len(f)
    idx = ev["idx"].to_numpy()
    can_sim = idx + 1 < n_bars  # the live bar has no entry bar yet: its outcome is unknown
    minute = f["minute"]
    big = np.iinfo(np.int64).max
    entry_min = np.where(can_sim, minute[np.minimum(idx + 1, n_bars - 1)], minute[np.minimum(idx, n_bars - 1)] + 1)
    exit_min = np.full(len(ev), big, np.int64)
    win = np.zeros(len(ev), bool)
    k = np.flatnonzero(can_sim)
    if len(k):
        sim = simulate(f, idx[k], ev["dir"].to_numpy()[k], ev["ref"].to_numpy()[k], ev["atr"].to_numpy()[k],
                       ev["liq_target"].to_numpy()[k], REFERENCE_EXIT, costs, min_stop_price=0.0)
        exit_min[k] = np.where(sim["exit_idx"] >= 0, minute[np.clip(sim["exit_idx"], 0, None)], big)
        # trades still open at the end of the data have no known outcome yet
        exit_min[k] = np.where(sim["reason"] == 5, big, exit_min[k])
        win[k] = (sim["r"] > 0) & sim["valid"]
    sess_grp = np.where(np.isin(ev["session"].to_numpy(), (2, 3, 4)), "main", "other")
    key = (ev["variant"].astype(str) + "|" + ev["dir"].astype(str) + "|" + sess_grp).to_numpy()
    wr = history_scores(entry_min, exit_min, key, win)
    ev = ev.copy()
    ev["hist_wr"] = wr
    ev["hist_score"] = np.clip((wr - 0.35) / 0.3, 0, 1)
    ev["score"] = score(ev, weights)
    return ev


@dataclass
class SweepResult:
    configs: pd.DataFrame  # one row per configuration (variant, exit, filter descriptors)
    monthly: np.ndarray  # [config, month, stat] float32
    dd_full: np.ndarray  # [config] max DD (R) across the full sweep window
    months: list[pd.Period]
    meta: dict = field(default_factory=dict)


def run_sweep(f: Features, variants: list[Variant], exits: list[tuple[ExitSpec, ExitCoord]],
              filters: list[FilterSpec], costs: Costs, warmup: int = 1500, tf: str = "M1") -> SweepResult:
    t0 = time.time()
    month_ids = f["month_id"]
    m0 = int(month_ids.min())
    n_months = int(month_ids.max()) - m0 + 1
    months = [pd.Period(year=(m0 + i) // 12, month=(m0 + i) % 12 + 1, freq="M") for i in range(n_months)]
    rows = []
    blocks = []
    dds = []
    for vi, v in enumerate(variants):
        ev = events_for(f, v, tf, warmup=warmup)
        ev = attach_scores(f, ev, costs)
        if len(ev) == 0:
            log.warning("variant %s produced no events", v.key)
            continue
        masks = np.vstack([flt.mask(ev) for flt in filters])
        month_pos = (month_ids[ev["idx"].to_numpy() + 1] - m0).astype(np.int64)
        idx = ev["idx"].to_numpy()
        for ei, (spec, coord) in enumerate(exits):
            sim = simulate(f, idx, ev["dir"].to_numpy(), ev["ref"].to_numpy(), ev["atr"].to_numpy(),
                           ev["liq_target"].to_numpy(), spec, costs, min_stop_price=0.0)
            # ``min_stop_atr`` already enforces a minimum distance; also require the stop to
            # exceed twice the signal-bar spread so costs cannot dominate by construction.
            valid = sim["valid"] & (sim["risk"] >= 2.0 * ev["spread"].to_numpy())
            stats, dd = eval_filters(sim["entry_idx"].astype(np.int64), sim["exit_idx"], sim["r"], valid, masks,
                                     month_pos, n_months)
            blocks.append(stats.astype(np.float32))
            dds.append(dd)
            for fi, flt in enumerate(filters):
                rows.append((v.family, tf, v.key, ei, spec.key, coord.sl_idx, coord.tp_idx, coord.mgmt, fi, flt.session,
                             flt.min_htf, flt.min_score, flt.vol_filter))
        log.info(f"[{tf}] " + "variant %d/%d %s: %d events, %.0fs elapsed", vi + 1, len(variants), v.key, len(ev), time.time() - t0)
    configs = pd.DataFrame(rows, columns=["family", "tf", "variant", "exit_id", "exit", "sl_idx", "tp_idx", "mgmt",
                                          "filter_id", "session", "min_htf", "min_score", "vol_filter"])
    monthly = np.concatenate(blocks, axis=0)
    return SweepResult(configs, monthly, np.concatenate(dds), months, {"seconds": time.time() - t0})


def window_metrics(monthly: np.ndarray, month_mask: np.ndarray) -> pd.DataFrame:
    """Aggregate per-config metrics over the selected months."""
    s = monthly[:, month_mask, :].astype(np.float64).sum(axis=1)
    n = s[:, S_N]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s[:, S_SUM] / n
        var = s[:, S_SQ] / n - mean**2
        sd = np.sqrt(np.maximum(var, 0) * n / np.maximum(n - 1, 1))
        t = mean / (sd / np.sqrt(n))
        pf = s[:, S_GP] / s[:, S_GL]
        wr = s[:, S_WIN] / n
    # Monthly consistency: fraction of active months with positive total R.
    act = monthly[:, month_mask, S_N] > 0
    pos = monthly[:, month_mask, S_SUM] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_frac = pos.sum(axis=1) / act.sum(axis=1)
    return pd.DataFrame({"n": n, "win_rate": wr, "exp_r": mean, "sd_r": sd, "t": t, "pf": pf,
                         "total_r": s[:, S_SUM], "pos_months": pos_frac})
