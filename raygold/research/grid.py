"""The research configuration space.

A configuration = entry variant x exit spec x filter set.  Every
configuration is evaluated with *at most one open position at a time* so
its trades are sequential and never overlap (overlapping trades would
otherwise be counted as independent evidence).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from raygold.backtest.simulator import (SL_ATR, SL_STRUCTURE, TP_ATR, TP_LIQUIDITY, TP_NETPOS, TP_NONE, TP_R,
                                        ExitSpec)
from raygold.patterns import sessions
from raygold.signals.events import Variant, _v, default_variants

SESSION_SETS = ("all", "no_asia", "ldn_ny", "overlap")
HTF_MIN = (-9, 1, 2)
SCORE_MIN = (0, 45, 55, 65)
VOL_FILTER = (0, 1)


@dataclass(frozen=True)
class FilterSpec:
    session: str = "all"
    min_htf: int = -9
    min_score: float = 0.0
    vol_filter: int = 0

    @property
    def key(self) -> str:
        return f"sess={self.session}|htf>={self.min_htf}|score>={self.min_score:g}|vol={self.vol_filter}"

    def as_dict(self) -> dict:
        return asdict(self)

    def mask(self, ev: pd.DataFrame) -> np.ndarray:
        s = ev["session"].to_numpy()
        if self.session == "all":
            m = np.ones(len(ev), bool)
        elif self.session == "no_asia":
            m = (s != sessions.ASIA) & (s != sessions.OFF)
        elif self.session == "ldn_ny":
            m = np.isin(s, (sessions.LONDON, sessions.NEWYORK, sessions.OVERLAP))
        elif self.session == "overlap":
            m = s == sessions.OVERLAP
        else:
            raise ValueError(self.session)
        if self.min_htf > -9:
            m &= ev["htf_align"].fillna(0).to_numpy() >= self.min_htf
        if self.min_score > 0:
            m &= ev["score"].to_numpy() >= self.min_score
        if self.vol_filter:
            p = ev["atr_pct"].to_numpy()
            m &= (p >= 0.2) & (p <= 0.95)
        return m


def default_filters() -> list[FilterSpec]:
    return [FilterSpec(s, h, sc, v) for s, h, sc, v in itertools.product(SESSION_SETS, HTF_MIN, SCORE_MIN, VOL_FILTER)]


SL_CHOICES = ((SL_STRUCTURE, 0.1), (SL_ATR, 0.75), (SL_ATR, 1.5))
TP_R_CHOICES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class ExitCoord:
    sl_idx: int
    tp_idx: int  # index into TP_R_CHOICES, -1 for non-R targets
    mgmt: str


# Holding-time limits in M1 bars, scaled with the signal timeframe.
MAX_HOLD = {"M1": (240, 60), "M5": (960, 300), "M15": (1440, 900)}


def default_exits(tf: str = "M1") -> list[tuple[ExitSpec, ExitCoord]]:
    hold, short = MAX_HOLD[tf]
    out: list[tuple[ExitSpec, ExitCoord]] = []
    for si, (slm, slp) in enumerate(SL_CHOICES):
        for ti, r in enumerate(TP_R_CHOICES):
            out.append((ExitSpec(slm, slp, TP_R, r, max_hold=hold), ExitCoord(si, ti, "none")))
        out.append((ExitSpec(slm, slp, TP_LIQUIDITY, 1.0, max_hold=hold), ExitCoord(si, -1, "liquidity")))
        out.append((ExitSpec(slm, slp, TP_NETPOS, 0.10, max_hold=hold), ExitCoord(si, -1, "netpos")))
        out.append((ExitSpec(slm, slp, TP_ATR, 1.0, max_hold=hold), ExitCoord(si, -1, "atr_tp")))
        out.append((ExitSpec(slm, slp, TP_R, 2.0, be_trigger_r=1.0, max_hold=hold), ExitCoord(si, -1, "be1_tp2")))
        out.append((ExitSpec(slm, slp, TP_R, 1.0, be_trigger_r=0.5, max_hold=hold), ExitCoord(si, -1, "be05_tp1")))
        out.append((ExitSpec(slm, slp, TP_NONE, 0.0, trail_atr=1.5, max_hold=hold), ExitCoord(si, -1, "trail")))
        out.append((ExitSpec(slm, slp, TP_R, 2.0, partial_r=1.0, partial_frac=0.5, max_hold=hold), ExitCoord(si, -1, "partial")))
        out.append((ExitSpec(slm, slp, TP_NONE, 0.0, max_hold=short), ExitCoord(si, -1, "time_short")))
    return out


def research_variants(n_random: int = 3, tf: str = "M1") -> list[Variant]:
    """Pattern variants plus ``n_random`` random-entry baselines (rate scaled to ~same events per day)."""
    from raygold.data.timeframes import TF_MINUTES

    vs = [v for v in default_variants() if v.family != "random_baseline"]
    for s in range(n_random):
        vs.append(_v("random_baseline", rate=min(0.004 * TF_MINUTES[tf], 0.5), seed=s))
    return vs
