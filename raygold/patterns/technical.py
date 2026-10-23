"""Classical technical-analysis pattern definitions (causal, deterministic).

Same conventions as :mod:`raygold.patterns.smc`: values at bar ``i`` are
known at its close; ``sig`` is +1/-1/0 and ``ref`` the invalidation price.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def ema_trend(close: np.ndarray, e20: np.ndarray, e50: np.ndarray, e100: np.ndarray, e200: np.ndarray) -> np.ndarray:
    """+1 when ``C > EMA20 > EMA50 > EMA100 > EMA200``, -1 for the mirror, else 0."""
    up = (close > e20) & (e20 > e50) & (e50 > e100) & (e100 > e200)
    dn = (close < e20) & (e20 < e50) & (e50 < e100) & (e100 < e200)
    return up.astype(np.int8) - dn.astype(np.int8)


def ema_slope_trend(e_fast: np.ndarray, e_slow: np.ndarray) -> np.ndarray:
    """Looser trend: +1 when EMA_fast > EMA_slow, -1 when below."""
    out = np.sign(e_fast - e_slow)
    out[np.isnan(out)] = 0
    return out.astype(np.int8)


def rejection_candles(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, atr: np.ndarray,
                      wick_body: float = 2.0, wick_range: float = 0.55, min_range_atr: float = 0.5) -> np.ndarray:
    """Pin bars and engulfing candles.

    Bullish pin: lower wick ``>= wick_body * body`` and ``>= wick_range *
    range`` with range ``>= min_range_atr * A``.  Bullish engulfing:
    previous bar bearish, current bullish, ``O_i <= C_{i-1}`` and
    ``C_i >= O_{i-1}`` with range ``>= min_range_atr * A``.  Mirror for bearish.
    """
    rng = h - l
    body = np.abs(c - o)
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    big = rng >= min_range_atr * atr
    bull_pin = big & (lower >= wick_body * body) & (lower >= wick_range * rng)
    bear_pin = big & (upper >= wick_body * body) & (upper >= wick_range * rng)
    po = np.concatenate(([np.nan], o[:-1]))
    pc = np.concatenate(([np.nan], c[:-1]))
    bull_eng = big & (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear_eng = big & (pc > po) & (c < o) & (o >= pc) & (c <= po)
    bull = bull_pin | bull_eng
    bear = bear_pin | bear_eng
    return (bull & ~bear).astype(np.int8) - (bear & ~bull).astype(np.int8)


@njit(cache=True)
def rsi_divergence(h, l, r, piv_h, piv_l, k, max_sep, min_rsi_diff, os_level, ob_level):
    """Regular RSI divergence at confirmed pivots.

    When a swing low pivot at bar p = i - k is confirmed at bar i, compare
    it with the previous confirmed swing low p' (``p - p' <= max_sep``):
    bullish if ``L_p < L_p'`` and ``RSI_p >= RSI_p' + min_rsi_diff`` and
    ``RSI_p' <= os_level``.  The signal fires at confirmation bar i;
    ``ref = L_p``.  Bearish mirror uses highs and ``ob_level``.
    """
    n = h.shape[0]
    sig = np.zeros(n, np.int8)
    ref = np.full(n, np.nan)
    pl_idx = -1
    ph_idx = -1
    for i in range(n):
        if piv_l[i] == piv_l[i]:
            p = i - k
            if pl_idx >= 0 and p - pl_idx <= max_sep:
                if l[p] < l[pl_idx] and r[p] >= r[pl_idx] + min_rsi_diff and r[pl_idx] <= os_level:
                    sig[i] = 1
                    ref[i] = l[p]
            pl_idx = p
        if piv_h[i] == piv_h[i]:
            p = i - k
            if ph_idx >= 0 and p - ph_idx <= max_sep:
                if h[p] > h[ph_idx] and r[p] <= r[ph_idx] - min_rsi_diff and r[ph_idx] >= ob_level:
                    if sig[i] == 1:
                        sig[i] = 0
                        ref[i] = np.nan
                    else:
                        sig[i] = -1
                        ref[i] = h[p]
            ph_idx = p
    return sig, ref


@njit(cache=True)
def breakout_retest(h, l, c, atr, n_lookback, retest_window, tol_atr):
    """Donchian breakout followed by a retest of the broken level.

    Breakout up at bar b: ``C_b > max(H_{b-N..b-1})``; level ``X`` is that
    max.  Retest at the first bar j in ``(b, b + retest_window]`` with
    ``L_j <= X + tol*A_j`` and ``C_j > X``; a close below ``X - tol*A``
    cancels.  ``ref = min(L_j, X) `` for longs.  Mirror for shorts.
    """
    n = h.shape[0]
    sig = np.zeros(n, np.int8)
    ref = np.full(n, np.nan)
    up_lvl = np.nan
    up_t = -1
    dn_lvl = np.nan
    dn_t = -1
    for i in range(n_lookback, n):
        a = atr[i]
        if a != a:
            continue
        if up_lvl == up_lvl:
            if i - up_t > retest_window or c[i] < up_lvl - tol_atr * a:
                up_lvl = np.nan
            elif l[i] <= up_lvl + tol_atr * a and c[i] > up_lvl:
                sig[i] = 1
                ref[i] = min(l[i], up_lvl)
                up_lvl = np.nan
        if dn_lvl == dn_lvl:
            if i - dn_t > retest_window or c[i] > dn_lvl + tol_atr * a:
                dn_lvl = np.nan
            elif h[i] >= dn_lvl - tol_atr * a and c[i] < dn_lvl:
                if sig[i] == 1:
                    sig[i] = 0
                    ref[i] = np.nan
                else:
                    sig[i] = -1
                    ref[i] = max(h[i], dn_lvl)
                dn_lvl = np.nan
        hh = h[i - n_lookback]
        ll = l[i - n_lookback]
        for j in range(i - n_lookback + 1, i):
            if h[j] > hh:
                hh = h[j]
            if l[j] < ll:
                ll = l[j]
        if c[i] > hh:
            up_lvl = hh
            up_t = i
        elif c[i] < ll:
            dn_lvl = ll
            dn_t = i
    return sig, ref


def ema_pullback(o: np.ndarray, l: np.ndarray, h: np.ndarray, c: np.ndarray,
                 e20: np.ndarray, e50: np.ndarray, e200: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trend-continuation pullback to EMA20.

    Long when ``EMA20 > EMA50 > EMA200``, ``L_i <= EMA20_i``, ``C_i > EMA20_i``,
    ``C_i > O_i`` and the previous close was above EMA50.  ``ref = min(L_i, EMA50_i)``.
    """
    pc = np.concatenate(([np.nan], c[:-1]))
    pe50 = np.concatenate(([np.nan], e50[:-1]))
    up = (e20 > e50) & (e50 > e200) & (l <= e20) & (c > e20) & (c > o) & (pc > pe50)
    dn = (e20 < e50) & (e50 < e200) & (h >= e20) & (c < e20) & (c < o) & (pc < pe50)
    sig = up.astype(np.int8) - dn.astype(np.int8)
    ref = np.where(up, np.minimum(l, e50), np.where(dn, np.maximum(h, e50), np.nan))
    return sig, ref


def sr_rejection(rej: np.ndarray, l: np.ndarray, h: np.ndarray, c: np.ndarray, atr: np.ndarray,
                 supports: np.ndarray, resistances: np.ndarray, tol_atr: float) -> tuple[np.ndarray, np.ndarray]:
    """Rejection candle at a support/resistance level.

    ``supports``/``resistances`` are ``(n, m)`` arrays of the latest
    confirmed swing lows/highs.  Long when a bullish rejection candle's low
    is within ``tol * A`` of any support and the close is above it.
    """
    tol = (tol_atr * atr)[:, None]
    near_s = (np.abs(l[:, None] - supports) <= tol) & (c[:, None] > supports)
    near_r = (np.abs(h[:, None] - resistances) <= tol) & (c[:, None] < resistances)
    up = (rej == 1) & near_s.any(axis=1)
    dn = (rej == -1) & near_r.any(axis=1)
    sig = up.astype(np.int8) - dn.astype(np.int8)
    ref = np.where(up, l, np.where(dn, h, np.nan))
    return sig, ref


def volume_spike(ratio: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """Tick volume >= ``threshold`` x median of the previous window."""
    return (ratio >= threshold).astype(np.int8)
