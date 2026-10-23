"""Position monitoring: paper-position simulation and stop management.

Paper positions are resolved against completed M1 bars with the same
conservative rules as the backtester (stop before target on ambiguous
bars, exits on the correct side of the spread, commission charged).
Break-even and trailing adjustments are applied after bar close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from raygold.backtest.simulator import ExitSpec

log = logging.getLogger(__name__)


@dataclass
class ManagedPosition:
    trade_id: int
    ticket: int
    direction: int
    lots: float
    entry: float
    sl: float
    tp: float | None
    risk: float  # initial stop distance (price)
    open_time: pd.Timestamp
    atr: float
    be_done: bool = False
    best: float = 0.0
    bars: int = 0


@dataclass
class CloseEvent:
    trade_id: int
    time: pd.Timestamp
    price: float
    reason: str
    pnl_per_unit: float


def update_paper_position(p: ManagedPosition, bar: pd.Series, spec: ExitSpec, commission: float,
                          slippage: float) -> CloseEvent | None:
    """Advance one completed bar; return a CloseEvent if the position closed."""
    d = p.direction
    if d > 0:
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    else:
        o, h, l, c = bar["ask_open"], bar["ask_high"], bar["ask_low"], bar["ask_close"]
    t = pd.Timestamp(bar["time"])
    p.bars += 1
    if p.best == 0.0:
        p.best = p.entry
    if (d > 0 and o <= p.sl) or (d < 0 and o >= p.sl):
        fill = o - slippage * d
        return CloseEvent(p.trade_id, t, fill, "stop(gap)", (fill - p.entry) * d - commission)
    sl_hit = (l <= p.sl) if d > 0 else (h >= p.sl)
    tp_hit = p.tp is not None and ((h >= p.tp) if d > 0 else (l <= p.tp))
    if sl_hit:
        fill = p.sl - slippage * d
        return CloseEvent(p.trade_id, t, fill, "stop", (fill - p.entry) * d - commission)
    if tp_hit:
        return CloseEvent(p.trade_id, t, p.tp, "target", (p.tp - p.entry) * d - commission)
    fav = (h - p.entry) if d > 0 else (p.entry - l)
    if spec.be_trigger_r > 0 and not p.be_done and fav >= spec.be_trigger_r * p.risk:
        be = p.entry + (commission + slippage) * d
        if (be - p.sl) * d > 0:
            p.sl = be
        p.be_done = True
    if spec.trail_atr > 0:
        if d > 0:
            p.best = max(p.best, h)
            p.sl = max(p.sl, p.best - spec.trail_atr * p.atr)
        else:
            p.best = min(p.best, l)
            p.sl = min(p.sl, p.best + spec.trail_atr * p.atr)
    if p.bars >= spec.max_hold:
        fill = c - slippage * d
        return CloseEvent(p.trade_id, t, fill, "time", (fill - p.entry) * d - commission)
    return None


def mt5_positions(conn, magic: int) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    pos = conn.mt5.positions_get(symbol=conn.cfg.symbol)
    if pos is None:
        return pd.DataFrame()
    df = pd.DataFrame([p._asdict() for p in pos])
    return df[df["magic"] == magic] if len(df) else df
