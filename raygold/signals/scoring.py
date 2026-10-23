"""Confidence score (0-100) for a setup event.

The score is a weighted average of component sub-scores in [0, 1].  It is
a *ranking heuristic*, not a probability: :func:`calibration_table`
reports how the realised win rate varies by score bucket on out-of-sample
data, and nothing should treat ``score / 100`` as P(win) unless that
table supports it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from raygold.patterns import sessions


@dataclass(frozen=True)
class ScoreWeights:
    htf: float = 2.0
    sweep: float = 1.5
    mss: float = 1.5
    fvg: float = 1.0
    volume: float = 0.5
    volatility: float = 0.75
    session: float = 1.0
    cost: float = 1.0
    zone: float = 0.75
    history: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


RECENT_BARS = 30


def components(ev: pd.DataFrame) -> pd.DataFrame:
    """Sub-scores in [0, 1] from the event context columns."""
    out = pd.DataFrame(index=ev.index)
    out["htf"] = (ev["htf_align"].fillna(0).clip(-3, 3) + 3.0) / 6.0
    out["sweep"] = (ev["since_sweep"] <= RECENT_BARS).astype(float)
    out["mss"] = (ev["since_mss"] <= RECENT_BARS).astype(float)
    out["fvg"] = (ev["since_fvg"] <= 3).astype(float)
    out["volume"] = (ev["vol_ratio"].fillna(1.0) / 2.0).clip(0, 1)
    # Prefer middling-to-high volatility; dead and extreme markets score low.
    p = ev["atr_pct"].fillna(0.5)
    out["volatility"] = np.clip(1.0 - np.abs(p - 0.65) / 0.45, 0, 1)
    s = ev["session"].to_numpy()
    out["session"] = np.select(
        [s == sessions.OVERLAP, s == sessions.LONDON, s == sessions.NEWYORK, s == sessions.ASIA],
        [1.0, 0.8, 0.8, 0.3], 0.0,
    )
    # Spread as a fraction of ATR: 0 -> 1.0, >= 0.5 ATR -> 0.
    out["cost"] = np.clip(1.0 - (ev["spread"] / ev["atr"]) / 0.5, 0, 1).fillna(0)
    out["zone"] = np.clip(1.0 - ev["pd_pos"].fillna(0.5), 0, 1)
    out["history"] = ev["hist_score"] if "hist_score" in ev else 0.5
    return out


def score(ev: pd.DataFrame, w: ScoreWeights = ScoreWeights()) -> np.ndarray:
    comp = components(ev)
    wd = w.as_dict()
    total = sum(wd.values())
    s = sum(comp[k].to_numpy() * wv for k, wv in wd.items())
    return np.round(100.0 * s / total, 2)


def calibration_table(scores: np.ndarray, wins: np.ndarray, bins: tuple[float, ...] = (0, 40, 50, 60, 70, 80, 101)) -> pd.DataFrame:
    """Realised win rate per score bucket (use on out-of-sample trades only)."""
    df = pd.DataFrame({"score": scores, "win": wins.astype(float)})
    df["bucket"] = pd.cut(df["score"], bins=list(bins), right=False)
    g = df.groupby("bucket", observed=True)["win"]
    return pd.DataFrame({"trades": g.size(), "win_rate": g.mean()}).reset_index()


def history_scores(
    entry_min: np.ndarray, exit_min: np.ndarray, key: np.ndarray, win: np.ndarray, prior: float = 0.5,
    prior_n: float = 20.0, window: int = 50,
) -> np.ndarray:
    """Past-only smoothed win rate of the last ``window`` earlier setups with the same ``key``.

    For each event e, uses only events of the same key whose *exit* time
    is strictly before e's entry time — i.e. outcomes already known when
    e fires.  Bayesian shrinkage toward ``prior`` with weight ``prior_n``.
    A rolling (not expanding) window keeps the live scanner, which only
    sees a finite lookback, consistent with the backtest.
    """
    from collections import deque

    n = len(entry_min)
    out = np.full(n, prior)
    order_exit = np.argsort(exit_min, kind="stable")
    order_entry = np.argsort(entry_min, kind="stable")
    hist: dict[object, deque] = {}
    j = 0
    for e in order_entry:
        t = entry_min[e]
        while j < n and exit_min[order_exit[j]] < t:
            k = key[order_exit[j]]
            hist.setdefault(k, deque(maxlen=window)).append(float(win[order_exit[j]]))
            j += 1
        h = hist.get(key[e])
        w = sum(h) if h else 0.0
        c = len(h) if h else 0
        out[e] = (w + prior * prior_n) / (c + prior_n)
    return out
