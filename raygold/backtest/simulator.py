"""Bar-level trade simulator on M1 bid/ask OHLC.

Execution model (all conservative):

* Signal at close of bar ``i``; market entry at the open of ``i + 1``:
  longs buy at the **ask** open plus slippage, shorts sell at the **bid**
  open minus slippage.  Longs exit on the bid series, shorts on the ask
  series, so the real historical spread is paid on every round trip.
* Stops are stop-market orders: filled at the stop price minus slippage,
  or at the (worse) open when price gaps through the stop.
* Take-profits are limit orders: filled exactly at the limit, never better.
* If a bar touches both the stop and a profit target, the **stop is
  assumed to have been hit first** (no intrabar path information).
* Break-even and trailing adjustments are computed from a completed bar
  and only take effect from the next bar.
* Commission is charged per round turn on the full position.
* A trade still open before a gap longer than ``max_gap_minutes``
  (weekends) is closed at that bar's close.

Prices and P&L are per unit (1 oz for XAUUSD); conversion to money and
lots happens in :mod:`raygold.risk.sizing` / :mod:`raygold.backtest.account`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numba import njit, prange

SL_STRUCTURE, SL_ATR = 0, 1
TP_R, TP_ATR, TP_LIQUIDITY, TP_NETPOS, TP_NONE = 0, 1, 2, 3, 4
R_SL, R_TP, R_TIME, R_WEEKEND, R_DATA_END, R_INVALID = 1, 2, 3, 4, 5, -1

REASONS = {R_SL: "stop", R_TP: "target", R_TIME: "time", R_WEEKEND: "weekend", R_DATA_END: "data_end", R_INVALID: "invalid"}


@dataclass(frozen=True)
class ExitSpec:
    sl_mode: int = SL_ATR
    sl_param: float = 1.0  # ATR multiple (SL_ATR) or ATR buffer beyond ref (SL_STRUCTURE)
    tp_mode: int = TP_R
    tp_param: float = 1.0  # R multiple / ATR multiple / min R for liquidity / profit cushion for net-positive
    be_trigger_r: float = 0.0  # 0 disables
    trail_atr: float = 0.0  # 0 disables
    partial_r: float = 0.0  # 0 disables
    partial_frac: float = 0.5
    max_hold: int = 240
    min_stop_atr: float = 0.3
    max_stop_atr: float = 4.0

    @property
    def key(self) -> str:
        sl = f"SLstruct+{self.sl_param:g}" if self.sl_mode == SL_STRUCTURE else f"SLatr{self.sl_param:g}"
        tp = {TP_R: f"TP{self.tp_param:g}R", TP_ATR: f"TPatr{self.tp_param:g}", TP_LIQUIDITY: f"TPliq>={self.tp_param:g}R",
              TP_NETPOS: f"TPnet+{self.tp_param:g}", TP_NONE: "TPnone"}[self.tp_mode]
        extra = []
        if self.be_trigger_r > 0:
            extra.append(f"BE@{self.be_trigger_r:g}R")
        if self.trail_atr > 0:
            extra.append(f"trail{self.trail_atr:g}atr")
        if self.partial_r > 0:
            extra.append(f"part{self.partial_frac:g}@{self.partial_r:g}R")
        extra.append(f"T{self.max_hold}")
        return "|".join([sl, tp] + extra)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Costs:
    commission: float = 0.07  # price units per unit, round turn ($7/lot/100oz)
    slippage: float = 0.05  # price units per market fill
    max_gap_minutes: int = 180


@njit(cache=True)
def _simulate_one(e, d, sl0, tp, tp1, frac1, be_trig, trail, max_hold, end_limit,
                  o, h, l, c, ao, ah, al, ac, minute, comm, slip, max_gap):
    """Simulate a single trade entered at the open of bar ``e``.

    Returns (exit_idx, pnl, reason, mfe, mae, entry_price, partial_done).
    """
    if d > 0:
        entry = ao[e] + slip
    else:
        entry = o[e] - slip
    risk = (entry - sl0) * d
    if not (risk > 0.0):
        return e, 0.0, R_INVALID, 0.0, 0.0, entry, False
    sl = sl0
    realised = 0.0
    remaining = 1.0
    part_done = tp1 != tp1  # no partial configured -> treat as done
    mfe = 0.0
    mae = 0.0
    be_done = be_trig != be_trig
    best = entry
    last = min(e + max_hold - 1, end_limit)
    for j in range(e, last + 1):
        if d > 0:
            bo_, bh_, bl_, bc_ = o[j], h[j], l[j], c[j]
        else:
            bo_, bh_, bl_, bc_ = ao[j], ah[j], al[j], ac[j]
        # 1) gap through the stop at the open (not possible on the entry bar)
        if j > e:
            if (d > 0 and bo_ <= sl) or (d < 0 and bo_ >= sl):
                fill = bo_ - slip * d
                pnl = realised + remaining * (fill - entry) * d - comm
                return j, pnl, R_SL, mfe, mae, entry, part_done
        if d > 0:
            sl_hit = bl_ <= sl
            tp_hit = tp == tp and bh_ >= tp
            tp1_hit = (not part_done) and bh_ >= tp1
            fav = bh_ - entry
            adv = entry - bl_
        else:
            sl_hit = bh_ >= sl
            tp_hit = tp == tp and bl_ <= tp
            tp1_hit = (not part_done) and bl_ <= tp1
            fav = entry - bl_
            adv = bh_ - entry
        if sl_hit:
            # conservative: stop first even if a target was also touched
            fill = sl - slip * d
            pnl = realised + remaining * (fill - entry) * d - comm
            if fav > mfe:
                mfe = fav
            if adv > mae:
                mae = adv
            return j, pnl, R_SL, mfe, mae, entry, part_done
        if tp_hit:
            if not part_done:
                realised += frac1 * (tp1 - entry) * d
                remaining -= frac1
                part_done = True
            pnl = realised + remaining * (tp - entry) * d - comm
            if fav > mfe:
                mfe = fav
            if adv > mae:
                mae = adv
            return j, pnl, R_TP, mfe, mae, entry, part_done
        if tp1_hit:
            realised += frac1 * (tp1 - entry) * d
            remaining -= frac1
            part_done = True
            be_level = entry + (comm + slip) * d
            if (be_level - sl) * d > 0:
                sl = be_level
        if fav > mfe:
            mfe = fav
        if adv > mae:
            mae = adv
        # end-of-bar management (effective next bar)
        if not be_done and mfe >= be_trig * risk:
            be_level = entry + (comm + slip) * d
            if (be_level - sl) * d > 0:
                sl = be_level
            be_done = True
        if trail > 0.0:
            if d > 0:
                if bh_ > best:
                    best = bh_
                cand = best - trail
                if cand > sl:
                    sl = cand
            else:
                if bl_ < best:
                    best = bl_
                cand = best + trail
                if cand < sl:
                    sl = cand
        # weekend / long gap: flatten at this bar's close
        if j < end_limit and minute[j + 1] - minute[j] > max_gap:
            fill = bc_ - slip * d
            return j, realised + remaining * (fill - entry) * d - comm, R_WEEKEND, mfe, mae, entry, part_done
    j = last
    if d > 0:
        fill = c[j] - slip
    else:
        fill = ac[j] + slip
    reason = R_TIME if last == e + max_hold - 1 else R_DATA_END
    return j, realised + remaining * (fill - entry) * d - comm, reason, mfe, mae, entry, part_done


@njit(cache=True, parallel=True)
def simulate_batch(entry_idx, direction, sl, tp, tp1, frac1, be_trig, trail, max_hold, end_limit,
                   o, h, l, c, ao, ah, al, ac, minute, comm, slip, max_gap):
    n = entry_idx.shape[0]
    exit_idx = np.empty(n, np.int64)
    pnl = np.empty(n)
    reason = np.empty(n, np.int8)
    mfe = np.empty(n)
    mae = np.empty(n)
    entry = np.empty(n)
    for k in prange(n):
        r = _simulate_one(entry_idx[k], direction[k], sl[k], tp[k], tp1[k], frac1, be_trig, trail[k], max_hold,
                          end_limit, o, h, l, c, ao, ah, al, ac, minute, comm, slip, max_gap)
        exit_idx[k] = r[0]
        pnl[k] = r[1]
        reason[k] = r[2]
        mfe[k] = r[3]
        mae[k] = r[4]
        entry[k] = r[5]
    return exit_idx, pnl, reason, mfe, mae, entry


def plan_levels(f, ev_idx: np.ndarray, ev_dir: np.ndarray, ev_ref: np.ndarray, ev_atr: np.ndarray,
                ev_liq: np.ndarray, spec: ExitSpec, costs: Costs, min_stop_price: float = 0.0,
                entry: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Compute entry-bar index and SL/TP levels for each event (vectorised).

    The stop is anchored to the *expected* entry (next-bar open, incl.
    spread/slippage), which is what a live order would use: the trade is
    placed at bar open with absolute SL/TP prices computed from data known
    at the signal bar plus that open.  Live trading passes ``entry`` (the
    current quote) explicitly.
    """
    e = ev_idx + 1
    d = ev_dir.astype(np.int64)
    if entry is None:
        entry = np.where(d > 0, f.ao[e] + costs.slippage, f.o[e] - costs.slippage)
    if spec.sl_mode == SL_STRUCTURE:
        sl = ev_ref - d * spec.sl_param * ev_atr
    else:
        sl = entry - d * spec.sl_param * ev_atr
    risk = (entry - sl) * d
    min_risk = np.maximum(spec.min_stop_atr * ev_atr, min_stop_price)
    widen = risk < min_risk
    sl = np.where(widen, entry - d * min_risk, sl)
    risk = (entry - sl) * d
    valid = (risk > 0) & (risk <= spec.max_stop_atr * ev_atr) & np.isfinite(risk)
    if spec.tp_mode == TP_R:
        tp = entry + d * spec.tp_param * risk
    elif spec.tp_mode == TP_ATR:
        tp = entry + d * spec.tp_param * ev_atr
    elif spec.tp_mode == TP_LIQUIDITY:
        # front-run the pool by a tenth of ATR; require at least tp_param R
        tp = ev_liq - d * 0.1 * ev_atr
        valid &= np.isfinite(tp) & ((tp - entry) * d >= spec.tp_param * risk)
    elif spec.tp_mode == TP_NETPOS:
        tp = entry + d * (costs.commission + spec.tp_param)
    else:
        tp = np.full(len(e), np.nan)
    tp1 = entry + d * spec.partial_r * risk if spec.partial_r > 0 else np.full(len(e), np.nan)
    trail = np.where(spec.trail_atr > 0, spec.trail_atr * ev_atr, 0.0)
    return {"e": e, "d": d, "sl": sl, "tp": tp, "tp1": tp1, "trail": trail, "risk": risk, "valid": valid,
            "entry_plan": entry}


def simulate(f, ev_idx, ev_dir, ev_ref, ev_atr, ev_liq, spec: ExitSpec, costs: Costs,
             end_limit: int | None = None, min_stop_price: float = 0.0) -> dict[str, np.ndarray]:
    """Simulate all events under one exit spec. Invalid events get reason R_INVALID."""
    n_bars = len(f)
    end_limit = n_bars - 1 if end_limit is None else end_limit
    plan = plan_levels(f, ev_idx, ev_dir, ev_ref, ev_atr, ev_liq, spec, costs, min_stop_price)
    ok = plan["valid"] & (plan["e"] <= end_limit)
    k = np.flatnonzero(ok)
    out_exit = np.full(len(ev_idx), -1, np.int64)
    out_pnl = np.zeros(len(ev_idx))
    out_reason = np.full(len(ev_idx), R_INVALID, np.int8)
    out_mfe = np.zeros(len(ev_idx))
    out_mae = np.zeros(len(ev_idx))
    out_entry = plan["entry_plan"].copy()
    if len(k):
        be = spec.be_trigger_r if spec.be_trigger_r > 0 else np.nan
        x, p, r, mf, ma, en = simulate_batch(
            plan["e"][k].astype(np.int64), plan["d"][k], plan["sl"][k], plan["tp"][k], plan["tp1"][k],
            spec.partial_frac, be, plan["trail"][k], spec.max_hold, end_limit,
            f.o, f.h, f.l, f.c, f.ao, f.ah, f.al, f.ac, f["minute"], costs.commission, costs.slippage,
            costs.max_gap_minutes,
        )
        out_exit[k], out_pnl[k], out_reason[k], out_mfe[k], out_mae[k], out_entry[k] = x, p, r, mf, ma, en
    risk = plan["risk"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_mult = np.where(out_reason != R_INVALID, out_pnl / risk, 0.0)
    return {"entry_idx": plan["e"], "exit_idx": out_exit, "pnl": out_pnl, "reason": out_reason, "r": r_mult,
            "risk": risk, "mfe_r": out_mfe / risk, "mae_r": out_mae / risk, "entry": out_entry,
            "sl": plan["sl"], "tp": plan["tp"], "valid": out_reason != R_INVALID}
