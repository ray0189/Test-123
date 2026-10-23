"""Signal generation on the most recent completed bar, using research code paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from raygold.backtest.simulator import Costs, plan_levels
from raygold.features import build_features
from raygold.research.engine import attach_scores
from raygold.research.strategy import StrategySpec
from raygold.signals.events import ensure_context
from raygold.signals.multitf import events_for
from raygold.signals.scoring import components


@dataclass
class LiveSignal:
    bar_time: pd.Timestamp
    direction: int
    strategy: str
    confidence: float
    entry: float
    sl: float
    tp: float
    rr: float
    risk_price: float
    spread: float
    atr: float
    components: dict

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bar_time"] = str(self.bar_time)
        d["direction"] = "BUY" if self.direction > 0 else "SELL"
        return d


@dataclass
class MarketSnapshot:
    bar_time: pd.Timestamp
    bid: float
    ask: float
    trend_m1: int
    trend_m5: float
    trend_m15: float
    trend_h1: float
    atr: float
    session: int
    patterns: list[str]


def snapshot(df: pd.DataFrame, bid: float, ask: float) -> tuple[MarketSnapshot, object]:
    f = build_features(df)
    ensure_context(f)
    i = len(f) - 1
    pats = []
    for tag, d in (("up", "bullish"), ("dn", "bearish")):
        if f[f"since_sweep_{tag}"][i] <= 30:
            pats.append(f"{d} liquidity sweep ({int(f[f'since_sweep_{tag}'][i])} bars ago)")
        if f[f"since_choch_{tag}"][i] <= 30:
            pats.append(f"{d} CHoCH ({int(f[f'since_choch_{tag}'][i])} bars ago)")
        elif f[f"since_mss_{tag}"][i] <= 30:
            pats.append(f"{d} BOS ({int(f[f'since_mss_{tag}'][i])} bars ago)")
        if f[f"since_fvg_{tag}"][i] <= 10:
            pats.append(f"{d} FVG retest ({int(f[f'since_fvg_{tag}'][i])} bars ago)")
    if f["rej"][i] != 0:
        pats.append("bullish rejection candle" if f["rej"][i] > 0 else "bearish rejection candle")
    snap = MarketSnapshot(
        bar_time=f.time.iloc[i], bid=bid, ask=ask, trend_m1=int(f["trend_m1"][i]), trend_m5=float(f["m5_trend"][i]),
        trend_m15=float(f["m15_trend"][i]), trend_h1=float(f["h1_trend"][i]), atr=float(f["atr"][i]),
        session=int(f["session"][i]), patterns=pats,
    )
    return snap, f


def latest_signal(f, spec: StrategySpec, costs: Costs, bid: float, ask: float) -> LiveSignal | None:  # type: ignore[no-untyped-def]
    """Signal on the last completed bar of ``f`` (a Features bundle), or None."""
    n = len(f)
    ev = events_for(f, spec.variant, spec.tf, warmup=300, end_idx=n)
    if len(ev) == 0 or ev["idx"].iloc[-1] != n - 1:
        return None
    # score the whole window so the 'historical performance' component of the
    # newest event is computed from earlier setups whose outcome is known
    ev = attach_scores(f, ev.reset_index(drop=True), costs).iloc[[-1]].reset_index(drop=True)
    if not spec.filter.mask(ev)[0]:
        return None
    d = int(ev["dir"].iloc[0])
    entry = np.array([ask + costs.slippage if d > 0 else bid - costs.slippage])
    plan = plan_levels(f, ev["idx"].to_numpy() - 1, ev["dir"].to_numpy(), ev["ref"].to_numpy(), ev["atr"].to_numpy(),
                       ev["liq_target"].to_numpy(), spec.exit, costs, entry=entry)
    if not plan["valid"][0] or plan["risk"][0] < 2.0 * (ask - bid):
        return None
    tp = float(plan["tp"][0])
    risk = float(plan["risk"][0])
    rr = abs(tp - entry[0]) / risk if tp == tp else float("nan")
    comp = components(ev).iloc[0].round(3).to_dict()
    return LiveSignal(bar_time=f.time.iloc[n - 1], direction=d, strategy=spec.key, confidence=float(ev["score"].iloc[0]),
                      entry=float(entry[0]), sl=float(plan["sl"][0]), tp=tp, rr=rr, risk_price=risk,
                      spread=float(ask - bid), atr=float(ev["atr"].iloc[0]), components=comp)
