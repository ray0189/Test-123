from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raygold import indicators as ind
from raygold.patterns import smc, technical
from raygold.patterns.sessions import LONDON, NEWYORK, OVERLAP, session_codes
from raygold.research.stats import cluster_bootstrap_mean, max_drawdown, summarise, wilson_ci


def test_wilson_matches_reference() -> None:
    lo, hi = wilson_ci(484, 1587)  # the user's earlier 30.5% result
    assert lo == pytest.approx(0.2828, abs=1e-3) and hi == pytest.approx(0.3284, abs=1e-3)


def test_drawdown_and_summary() -> None:
    assert max_drawdown(np.array([1, -2, 1, -3, 5])) == 4
    r = np.array([1.0, -1.0, 1.0, 0.0])
    s = summarise(r, np.array([0, 0, 1, 1]))
    assert s.trades == 4 and s.wins == 2 and s.scratches == 1 and s.win_rate == 0.5


def test_cluster_bootstrap_wider_than_naive_for_correlated_days() -> None:
    rng = np.random.default_rng(0)
    day_effect = np.repeat(rng.normal(0, 1, 50), 20)
    vals = day_effect + rng.normal(0, 0.2, 1000)
    lo, hi = cluster_bootstrap_mean(vals, np.repeat(np.arange(50), 20))
    naive = 1.96 * vals.std() / np.sqrt(len(vals))
    assert (hi - lo) / 2 > 2 * naive


def test_indicators_against_pandas() -> None:
    x = np.random.default_rng(1).normal(0, 1, 500).cumsum() + 100
    e = ind.ema(x, 20)
    ref = pd.Series(x).ewm(span=20, adjust=False).mean().to_numpy()
    # seeded with SMA; converges to the pandas EMA
    assert np.allclose(e[200:], ref[200:], atol=1e-6)
    np.testing.assert_allclose(ind.rolling_max(x, 10)[9:], pd.Series(x).rolling(10).max().to_numpy()[9:])
    r = ind.rsi(x, 14)
    assert np.nanmin(r) >= 0 and np.nanmax(r) <= 100


def test_sessions_dst() -> None:
    t = pd.Series(pd.to_datetime(["2025-01-15 08:30", "2025-07-15 07:30", "2025-07-15 13:00", "2025-01-15 19:00"], utc=True))
    s = session_codes(t)
    assert s[0] == LONDON  # winter London 08:30 local
    assert s[1] == LONDON  # summer: 07:30 UTC = 08:30 BST
    assert s[2] == OVERLAP  # 13:00 UTC = 09:00 EDT, London open
    assert s[3] == NEWYORK


def _arr(rows):
    o, h, l, c = (np.array(v, float) for v in zip(*rows))
    return o, h, l, c


def test_fvg_formation_and_retest() -> None:
    rows = [(10, 10.5, 9.5, 10), (10, 11, 9.9, 11), (11, 12.5, 10.8, 12), (12, 12.6, 11.8, 12.4), (12.4, 12.5, 10.6, 11.0)]
    o, h, l, c = _arr(rows)
    atr = np.full(len(rows), 1.0)
    formed, retest, ref = smc.fair_value_gaps(h, l, c, o, atr, 0.1, 50, 4)
    assert formed[2] == 1  # low[2]=10.8 > high[0]=10.5
    assert retest[4] == 1 and ref[4] == 10.5  # dipped to 10.6 <= top 10.8, closed above bottom


def test_sweep_requires_reclaim() -> None:
    h = np.array([10, 10, 10, 10, 10.0])
    l = np.array([9, 8, 9, 9, 7.8])
    c = np.array([9.5, 9.5, 9.5, 9.5, 8.5])
    piv_l = np.array([np.nan, np.nan, 8.0, np.nan, np.nan])  # pivot low 8.0 confirmed at bar 2
    sig, ref, lvl = smc.liquidity_sweep(h, l, c, np.ones(5), np.full(5, np.nan), piv_l, 1.5, 100)
    assert sig[4] == 1 and ref[4] == 7.8 and lvl[4] == 8.0
    c2 = c.copy()
    c2[4] = 7.9  # closed below: a breakdown, not a sweep
    sig2, _, _ = smc.liquidity_sweep(h, l, c2, np.ones(5), np.full(5, np.nan), piv_l, 1.5, 100)
    assert sig2[4] == 0


def test_bos_then_choch() -> None:
    n = 8
    h = np.array([10, 10, 10, 10, 10, 10, 10, 10.0])
    l = np.array([9, 9, 9, 9, 9, 9, 9, 9.0])
    c = np.array([9.5, 10.5, 9.5, 8.5, 9.5, 9.5, 9.5, 9.5])
    piv_h = np.full(n, np.nan)
    piv_l = np.full(n, np.nan)
    piv_h[0] = 10.0
    piv_l[2] = 9.0
    bos, choch, ref, state, brk = smc.market_structure(h, l, c, piv_h, piv_l)
    assert bos[1] == 1 and state[1] == 1
    assert choch[3] == -1 and state[3] == -1


def test_rejection_candles() -> None:
    o = np.array([10.0, 10.0])
    h = np.array([10.5, 10.2])
    l = np.array([9.5, 8.0])
    c = np.array([10.0, 10.1])
    r = technical.rejection_candles(o, h, l, c, np.ones(2))
    assert r[1] == 1  # long lower wick pin bar


def test_breakout_retest() -> None:
    h = np.array([10, 10, 10, 11.2, 11.0, 11.3])
    l = np.array([9, 9, 9, 10.1, 10.05, 10.6])
    c = np.array([9.5, 9.5, 9.5, 11.0, 10.5, 11.1])
    sig, ref = technical.breakout_retest(h, l, c, np.ones(6), 3, 10, 0.2)
    assert sig[4] == 1 and ref[4] == 10.0


def test_equal_lows_sweep() -> None:
    n = 6
    piv_l = np.full(n, np.nan)
    piv_l[1], piv_l[3] = 9.0, 9.05
    h = np.full(n, 10.0)
    l = np.array([9.5, 9.5, 9.5, 9.5, 9.5, 8.9])
    c = np.array([9.7, 9.7, 9.7, 9.7, 9.7, 9.2])
    sig, ref, eqh, eql = smc.equal_levels(np.full(n, np.nan), piv_l, np.ones(n), 0.1, h, l, c, 100)
    assert eql[3] == 9.0 and sig[5] == 1
