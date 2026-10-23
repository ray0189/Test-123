"""Causal technical indicators.

Every function returns an array whose element ``i`` depends only on inputs
``0..i``.  ``tests/test_no_lookahead.py`` verifies this by truncating the
input and checking that earlier outputs do not change.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def ema(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full(x.shape[0], np.nan)
    alpha = 2.0 / (period + 1.0)
    acc = np.nan
    n = 0
    s = 0.0
    for i in range(x.shape[0]):
        v = x[i]
        if np.isnan(v):
            out[i] = acc
            continue
        if n < period:
            s += v
            n += 1
            if n == period:
                acc = s / period
                out[i] = acc
        else:
            acc = acc + alpha * (v - acc)
            out[i] = acc
    return out


@njit(cache=True)
def wilder(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full(x.shape[0], np.nan)
    acc = 0.0
    n = 0
    for i in range(x.shape[0]):
        v = x[i]
        if np.isnan(v):
            if n >= period:
                out[i] = acc
            continue
        if n < period:
            acc += v
            n += 1
            if n == period:
                acc /= period
                out[i] = acc
        else:
            acc = (acc * (period - 1) + v) / period
            out[i] = acc
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    tr[0] = high[0] - low[0]
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    return wilder(true_range(high, low, close), period)


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=np.nan)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    up[0] = np.nan
    dn[0] = np.nan
    au = wilder(up, period)
    ad = wilder(dn, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = au / ad
        out = 100.0 - 100.0 / (1.0 + rs)
    out[(ad == 0) & (au > 0)] = 100.0
    return out


@njit(cache=True)
def rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    """Max of x[i-window+1 .. i] (inclusive of i)."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    dq = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(n):
        while tail > head and x[dq[tail - 1]] <= x[i]:
            tail -= 1
        dq[tail] = i
        tail += 1
        if dq[head] <= i - window:
            head += 1
        if i >= window - 1:
            out[i] = x[dq[head]]
    return out


def rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    return -rolling_max(-x, window)


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full(x.shape[0], np.nan)
    if k < x.shape[0]:
        out[k:] = x[:-k] if k > 0 else x
    return out


@njit(cache=True)
def rolling_median_ratio(x: np.ndarray, window: int) -> np.ndarray:
    """x[i] / median(x[i-window .. i-1]) — the current bar vs. the *previous* window."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    buf = np.empty(window)
    for i in range(window, n):
        for j in range(window):
            buf[j] = x[i - window + j]
        med = np.median(buf)
        if med > 0:
            out[i] = x[i] / med
    return out


@njit(cache=True)
def rolling_percentile_rank(x: np.ndarray, window: int) -> np.ndarray:
    """Fraction of the previous ``window`` values strictly below x[i] (past-only)."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    for i in range(window, n):
        v = x[i]
        if np.isnan(v):
            continue
        c = 0
        m = 0
        for j in range(i - window, i):
            if not np.isnan(x[j]):
                m += 1
                if x[j] < v:
                    c += 1
        if m > 0:
            out[i] = c / m
    return out


@njit(cache=True)
def confirmed_pivots(high: np.ndarray, low: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractal pivots that only become *known* ``k`` bars after the pivot.

    Returns two arrays aligned to the confirmation bar ``c = p + k``:
    ``ph[c]`` = pivot-high price confirmed at bar c (NaN otherwise), and
    ``pl[c]`` likewise for pivot lows.  A pivot high at p requires
    ``high[p] > high[p-k..p-1]`` and ``high[p] >= high[p+1..p+k]``.
    """
    n = high.shape[0]
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for p in range(k, n - k):
        hv = high[p]
        is_h = True
        for j in range(1, k + 1):
            if high[p - j] >= hv or high[p + j] > hv:
                is_h = False
                break
        if is_h:
            ph[p + k] = hv
        lv = low[p]
        is_l = True
        for j in range(1, k + 1):
            if low[p - j] <= lv or low[p + j] < lv:
                is_l = False
                break
        if is_l:
            pl[p + k] = lv
    return ph, pl


@njit(cache=True)
def last_n_levels(levels: np.ndarray, nkeep: int) -> np.ndarray:
    """Forward-fill the last ``nkeep`` confirmed pivot levels: out[i, 0] is the most recent."""
    n = levels.shape[0]
    out = np.full((n, nkeep), np.nan)
    cur = np.full(nkeep, np.nan)
    for i in range(n):
        if not np.isnan(levels[i]):
            for j in range(nkeep - 1, 0, -1):
                cur[j] = cur[j - 1]
            cur[0] = levels[i]
        for j in range(nkeep):
            out[i, j] = cur[j]
    return out
