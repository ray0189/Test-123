"""Performance statistics with honest uncertainty estimates.

Trades from the same trading day are correlated (same volatility regime,
often overlapping).  Confidence intervals for expectancy therefore use a
*day-clustered* bootstrap: whole days are resampled, never single trades.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


def wilson_ci(wins: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def max_drawdown(pnl: np.ndarray, start_equity: float = 0.0) -> float:
    """Largest peak-to-trough decline of cumulative P&L (positive number)."""
    if len(pnl) == 0:
        return 0.0
    eq = start_equity + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate(([start_equity], eq)))[1:]
    return float(np.max(peak - eq))


def cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 11,
                           alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for the per-trade mean, resampling whole clusters (days)."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    k = len(uniq)
    draws = rng.integers(0, k, size=(n_boot, k))
    s = sums[draws].sum(axis=1)
    c = counts[draws].sum(axis=1)
    means = s / c
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def clustered_t(values: np.ndarray, clusters: np.ndarray) -> float:
    """t-statistic of the mean using cluster-robust (by day) standard error."""
    n = len(values)
    if n < 3:
        return float("nan")
    mu = values.mean()
    uniq, inv = np.unique(clusters, return_inverse=True)
    g = len(uniq)
    if g < 3:
        return float("nan")
    resid_sum = np.bincount(inv, weights=values - mu)
    var = (g / (g - 1)) * np.sum(resid_sum**2) / n**2
    return float(mu / math.sqrt(var)) if var > 0 else float("nan")


@dataclass
class Summary:
    trades: int
    wins: int
    losses: int
    scratches: int
    win_rate: float
    win_rate_lo: float
    win_rate_hi: float
    expectancy_r: float
    expectancy_r_lo: float
    expectancy_r_hi: float
    t_stat: float
    profit_factor: float
    avg_win_r: float
    avg_loss_r: float
    total_r: float
    max_dd_r: float
    trading_days: int
    trades_per_day: float
    breakeven_win_rate: float

    def as_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def summarise(r: np.ndarray, day: np.ndarray, n_boot: int = 2000) -> Summary:
    """Summary of a chronologically ordered sequence of trade R-multiples."""
    r = np.asarray(r, float)
    n = len(r)
    wins = int((r > 1e-9).sum())
    losses = int((r < -1e-9).sum())
    scratches = n - wins - losses
    lo, hi = wilson_ci(wins, n)
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    exp_lo, exp_hi = cluster_bootstrap_mean(r, day, n_boot) if n >= 2 else (float("nan"), float("nan"))
    days = len(np.unique(day)) if n else 0
    avg_w = float(r[r > 1e-9].mean()) if wins else 0.0
    avg_l = float(r[r < -1e-9].mean()) if losses else 0.0
    be = (-avg_l) / (avg_w - avg_l) if wins and losses else float("nan")
    return Summary(
        trades=n, wins=wins, losses=losses, scratches=scratches,
        win_rate=wins / n if n else float("nan"), win_rate_lo=lo, win_rate_hi=hi,
        expectancy_r=float(r.mean()) if n else float("nan"), expectancy_r_lo=exp_lo, expectancy_r_hi=exp_hi,
        t_stat=clustered_t(r, day) if n else float("nan"),
        profit_factor=float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else float("nan")),
        avg_win_r=avg_w, avg_loss_r=avg_l, total_r=float(r.sum()), max_dd_r=max_drawdown(r),
        trading_days=days, trades_per_day=n / days if days else 0.0, breakeven_win_rate=be,
    )
