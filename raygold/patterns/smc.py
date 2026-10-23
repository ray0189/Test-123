"""ICT / Smart-Money-Concept pattern definitions.

All detectors take bid OHLC arrays and return, per bar ``i``, values that
are fully determined at the *close* of bar ``i``.  A trade triggered by a
signal on bar ``i`` is entered at the open of bar ``i + 1``.

Notation: ``H, L, O, C`` are bar high, low, open, close; ``A`` is ATR(14)
at the same bar; a *confirmed swing* is a fractal pivot with strength
``k`` that is only usable ``k`` bars after the pivot bar (see
:func:`raygold.indicators.confirmed_pivots`).

Return conventions: ``sig`` is +1 (bullish), -1 (bearish) or 0; ``ref``
is the structural invalidation price (the level a stop belongs beyond).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def liquidity_sweep(h, l, c, atr, piv_h, piv_l, max_depth_atr, max_age):
    """Stop-hunt through the most recent unswept confirmed swing.

    Bullish at bar i when, for the latest confirmed swing low level ``S``
    (confirmed at or before bar i-1, not older than ``max_age`` bars, not
    already swept):  ``L_i < S``, ``C_i > S`` and ``S - L_i <= max_depth_atr * A_i``.
    ``ref = L_i``.  Bearish is the mirror image on swing highs.
    """
    n = h.shape[0]
    sig = np.zeros(n, np.int8)
    ref = np.full(n, np.nan)
    lvl_out = np.full(n, np.nan)
    sh = np.nan
    sh_t = -1
    sl = np.nan
    sl_t = -1
    for i in range(n):
        if sl == sl and i - sl_t <= max_age and l[i] < sl:
            if c[i] > sl and sl - l[i] <= max_depth_atr * atr[i]:
                sig[i] = 1
                ref[i] = l[i]
                lvl_out[i] = sl
            sl = np.nan  # level is consumed either way (swept or broken)
        if sh == sh and i - sh_t <= max_age and h[i] > sh:
            if c[i] < sh and h[i] - sh <= max_depth_atr * atr[i]:
                if sig[i] == 1:
                    sig[i] = 0  # outside bar sweeping both sides: ambiguous
                    ref[i] = np.nan
                else:
                    sig[i] = -1
                    ref[i] = h[i]
                    lvl_out[i] = sh
            sh = np.nan
        # pivots confirmed on bar i become usable from bar i+1
        if piv_l[i] == piv_l[i]:
            sl = piv_l[i]
            sl_t = i
        if piv_h[i] == piv_h[i]:
            sh = piv_h[i]
            sh_t = i
    return sig, ref, lvl_out


@njit(cache=True)
def market_structure(h, l, c, piv_h, piv_l):
    """Break of Structure / Change of Character state machine.

    Keeps the latest unbroken confirmed swing high ``SH`` and swing low
    ``SL`` and a structure state ``s`` in {-1, 0, +1}.
    * ``C_i > SH``: bullish break; CHoCH if ``s == -1`` else BOS; s := +1.
    * ``C_i < SL``: bearish break; CHoCH if ``s == +1`` else BOS; s := -1.
    A broken swing cannot trigger again.  ``ref`` for a bullish break is
    the latest confirmed swing low (the protective low of the new leg).

    Returns ``bos`` (+1/-1/0), ``choch`` (+1/-1/0), ``ref``, ``state`` and
    ``brk_level`` (the swing that was broken).
    """
    n = h.shape[0]
    bos = np.zeros(n, np.int8)
    choch = np.zeros(n, np.int8)
    ref = np.full(n, np.nan)
    state = np.zeros(n, np.int8)
    brk = np.full(n, np.nan)
    sh = np.nan
    sl = np.nan
    last_sl = np.nan
    last_sh = np.nan
    s = 0
    for i in range(n):
        if sh == sh and c[i] > sh:
            if s == -1:
                choch[i] = 1
            else:
                bos[i] = 1
            ref[i] = last_sl
            brk[i] = sh
            s = 1
            sh = np.nan
        elif sl == sl and c[i] < sl:
            if s == 1:
                choch[i] = -1
            else:
                bos[i] = -1
            ref[i] = last_sh
            brk[i] = sl
            s = -1
            sl = np.nan
        state[i] = s
        if piv_h[i] == piv_h[i]:
            sh = piv_h[i]
            last_sh = piv_h[i]
        if piv_l[i] == piv_l[i]:
            sl = piv_l[i]
            last_sl = piv_l[i]
    return bos, choch, ref, state, brk


@njit(cache=True)
def fair_value_gaps(h, l, c, o, atr, min_gap_atr, expiry, max_active):
    """Three-candle imbalance and its first retest.

    Bullish FVG formed at bar i: ``L_i > H_{i-2}`` and gap
    ``L_i - H_{i-2} >= min_gap_atr * A_i``; the zone is ``[H_{i-2}, L_i]``.
    It is *retested* at the first later bar j (``i < j <= i + expiry``)
    with ``L_j <= top`` and ``C_j >= bottom`` (touched, not closed
    through).  A close below ``bottom`` invalidates it.  Each gap fires at
    most once.  Returns ``formed`` (+1/-1 at formation bar), ``retest``
    (+1/-1 at retest bar) and ``ref`` (zone far edge).
    """
    n = h.shape[0]
    formed = np.zeros(n, np.int8)
    retest = np.zeros(n, np.int8)
    ref = np.full(n, np.nan)
    # active gaps: [bottom, top, created, alive]
    bull = np.full((max_active, 3), np.nan)
    bear = np.full((max_active, 3), np.nan)
    for i in range(2, n):
        # retests / invalidation of existing zones (formed on earlier bars)
        for k in range(max_active):
            b = bull[k, 0]
            if b == b:
                if i - bull[k, 2] > expiry or c[i] < b:
                    bull[k, 0] = np.nan
                elif l[i] <= bull[k, 1]:
                    if retest[i] == 0:
                        retest[i] = 1
                        ref[i] = b
                    bull[k, 0] = np.nan
            b = bear[k, 1]
            if b == b:
                if i - bear[k, 2] > expiry or c[i] > b:
                    bear[k, 1] = np.nan
                elif h[i] >= bear[k, 0]:
                    if retest[i] == 0:
                        retest[i] = -1
                        ref[i] = b
                    elif retest[i] == 1:
                        retest[i] = 0
                        ref[i] = np.nan
                    bear[k, 1] = np.nan
        # formation (zone becomes tradable from the next bar)
        a = atr[i]
        if a == a:
            if l[i] > h[i - 2] and l[i] - h[i - 2] >= min_gap_atr * a:
                formed[i] = 1
                slot = 0
                oldest = 1e18
                for k in range(max_active):
                    if bull[k, 0] != bull[k, 0]:
                        slot = k
                        break
                    if bull[k, 2] < oldest:
                        oldest = bull[k, 2]
                        slot = k
                bull[slot, 0] = h[i - 2]
                bull[slot, 1] = l[i]
                bull[slot, 2] = i
            elif h[i] < l[i - 2] and l[i - 2] - h[i] >= min_gap_atr * a:
                formed[i] = -1
                slot = 0
                oldest = 1e18
                for k in range(max_active):
                    if bear[k, 1] != bear[k, 1]:
                        slot = k
                        break
                    if bear[k, 2] < oldest:
                        oldest = bear[k, 2]
                        slot = k
                bear[slot, 0] = h[i]  # bottom
                bear[slot, 1] = l[i - 2]  # top
                bear[slot, 2] = i
    return formed, retest, ref


@njit(cache=True)
def order_blocks(h, l, o, c, brk_dir, lookback, expiry):
    """Order block = last opposite-coloured candle before a structure break.

    When a bullish BOS/CHoCH occurs at bar b (``brk_dir[b] == +1``), the
    bullish OB is the last bearish candle (``C < O``) in ``[b - lookback,
    b - 1]``; its zone is ``[L_ob, H_ob]``.  Retest at the first bar j in
    ``(b, b + expiry]`` with ``L_j <= H_ob`` and ``C_j >= L_ob``.  A close
    below ``L_ob`` invalidates.  Mirror for bearish.  Only the most recent
    OB per direction is tracked.
    """
    n = h.shape[0]
    sig = np.zeros(n, np.int8)
    ref = np.full(n, np.nan)
    bl = np.nan
    bh = np.nan
    bt = -1
    sl_ = np.nan
    shh = np.nan
    st = -1
    for i in range(n):
        if bl == bl:
            if i - bt > expiry or c[i] < bl:
                bl = np.nan
            elif l[i] <= bh:
                sig[i] = 1
                ref[i] = bl
                bl = np.nan
        if sl_ == sl_:
            if i - st > expiry or c[i] > shh:
                sl_ = np.nan
            elif h[i] >= sl_:
                if sig[i] == 1:
                    sig[i] = 0
                    ref[i] = np.nan
                else:
                    sig[i] = -1
                    ref[i] = shh
                sl_ = np.nan
        if brk_dir[i] == 1:
            for j in range(i - 1, max(-1, i - lookback - 1), -1):
                if c[j] < o[j]:
                    bl = l[j]
                    bh = h[j]
                    bt = i
                    break
        elif brk_dir[i] == -1:
            for j in range(i - 1, max(-1, i - lookback - 1), -1):
                if c[j] > o[j]:
                    sl_ = l[j]  # zone bottom (entry edge for shorts)
                    shh = h[j]  # zone top (invalidation)
                    st = i
                    break
    return sig, ref


@njit(cache=True)
def equal_levels(piv_h, piv_l, atr, tol_atr, h, l, c, max_age):
    """Equal highs / equal lows and their sweeps.

    When a new swing low is confirmed at bar i and differs from the
    previous confirmed swing low by at most ``tol_atr * A_i``, an EQL pool
    at ``min(both)`` is created.  A *sweep* of the pool at bar j is
    ``L_j < pool`` and ``C_j > pool`` (bullish).  A close through the pool
    destroys it.  Returns ``eq_sweep`` (+1/-1), ``ref`` and the currently
    active pool levels (``eqh``, ``eql``) for use as liquidity targets.
    """
    n = h.shape[0]
    sig = np.zeros(n, np.int8)
    ref = np.full(n, np.nan)
    eqh_o = np.full(n, np.nan)
    eql_o = np.full(n, np.nan)
    prev_h = np.nan
    prev_l = np.nan
    eqh = np.nan
    eqh_t = -1
    eql = np.nan
    eql_t = -1
    for i in range(n):
        if eql == eql:
            if i - eql_t > max_age:
                eql = np.nan
            elif l[i] < eql:
                if c[i] > eql:
                    sig[i] = 1
                    ref[i] = l[i]
                eql = np.nan
        if eqh == eqh:
            if i - eqh_t > max_age:
                eqh = np.nan
            elif h[i] > eqh:
                if c[i] < eqh:
                    if sig[i] == 1:
                        sig[i] = 0
                        ref[i] = np.nan
                    else:
                        sig[i] = -1
                        ref[i] = h[i]
                eqh = np.nan
        a = atr[i]
        if piv_l[i] == piv_l[i]:
            if prev_l == prev_l and a == a and abs(piv_l[i] - prev_l) <= tol_atr * a:
                eql = min(piv_l[i], prev_l)
                eql_t = i
            prev_l = piv_l[i]
        if piv_h[i] == piv_h[i]:
            if prev_h == prev_h and a == a and abs(piv_h[i] - prev_h) <= tol_atr * a:
                eqh = max(piv_h[i], prev_h)
                eqh_t = i
            prev_h = piv_h[i]
        eqh_o[i] = eqh
        eql_o[i] = eql
    return sig, ref, eqh_o, eql_o


def premium_discount(close: np.ndarray, range_high: np.ndarray, range_low: np.ndarray) -> np.ndarray:
    """Position of the close inside a (completed, higher-timeframe) dealing range.

    0 = range low (deep discount), 1 = range high (deep premium).  Buys are
    favoured below 0.5, sells above.
    """
    width = range_high - range_low
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = (close - range_low) / width
    pos[~(width > 0)] = np.nan
    return pos
