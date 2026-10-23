"""Build every causal feature array needed by the pattern engine.

The output is a :class:`Features` bundle of NumPy arrays aligned to the
M1 frame.  Nothing here looks at a bar later than the one it is attached
to; higher-timeframe values come only from *completed* candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from raygold import indicators as ind
from raygold.data.timeframes import align_to_m1, resample
from raygold.patterns import sessions
from raygold.patterns.technical import ema_slope_trend, ema_trend

PIVOT_STRENGTHS = (3, 5, 10)
N_LEVELS = 5


@dataclass
class Features:
    time: pd.Series
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    ao: np.ndarray
    ah: np.ndarray
    al: np.ndarray
    ac: np.ndarray
    vol: np.ndarray
    arr: dict[str, np.ndarray] = field(default_factory=dict)
    cache: dict = field(default_factory=dict)  # derived objects, e.g. higher-timeframe signal features

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"time": self.time, "open": self.o, "high": self.h, "low": self.l, "close": self.c,
                             "ask_open": self.ao, "ask_high": self.ah, "ask_low": self.al, "ask_close": self.ac,
                             "volume": self.vol})

    def __len__(self) -> int:
        return len(self.c)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arr[key]

    def __contains__(self, key: str) -> bool:
        return key in self.arr


def _htf_context(df: pd.DataFrame, tf: str, prefix: str) -> pd.DataFrame:
    bars = resample(df, tf)
    c = bars["close"].to_numpy()
    e20 = ind.ema(c, 20)
    e50 = ind.ema(c, 50)
    bars["trend"] = ema_slope_trend(e20, e50).astype(float)
    # Strong trend requires the close on the right side of EMA20 as well.
    bars["trend"] = np.where((bars["trend"] > 0) & (c > e20), 1.0, np.where((bars["trend"] < 0) & (c < e20), -1.0, 0.0))
    bars["atr"] = ind.atr(bars["high"].to_numpy(), bars["low"].to_numpy(), c, 14)
    cols = ["trend", "atr", "close"]
    if tf == "H1":
        bars["range_hi"] = ind.rolling_max(bars["high"].to_numpy(), 24)
        bars["range_lo"] = ind.rolling_min(bars["low"].to_numpy(), 24)
        ph, pl = ind.confirmed_pivots(bars["high"].to_numpy(), bars["low"].to_numpy(), 3)
        lh = ind.last_n_levels(ph, 3)
        ll = ind.last_n_levels(pl, 3)
        for j in range(3):
            bars[f"swing_hi{j}"] = lh[:, j]
            bars[f"swing_lo{j}"] = ll[:, j]
        cols += ["range_hi", "range_lo"] + [f"swing_hi{j}" for j in range(3)] + [f"swing_lo{j}" for j in range(3)]
    return align_to_m1(df["time"], bars, cols, prefix)


def _previous_day_levels(df: pd.DataFrame, tday: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """High/low of the previous *completed* trading day (rolls 17:00 New York)."""
    s = pd.DataFrame({"d": tday, "h": df["high"].to_numpy(), "l": df["low"].to_numpy()})
    daily = s.groupby("d", sort=True).agg(h=("h", "max"), l=("l", "min"))
    prev = daily.shift(1)
    return prev["h"].reindex(tday).to_numpy(copy=True), prev["l"].reindex(tday).to_numpy(copy=True)


def _asia_range(df: pd.DataFrame, tday: np.ndarray, sess: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Asia-session high/low of the current trading day, visible only after Asia closes."""
    is_asia = sess == sessions.ASIA
    s = pd.DataFrame({"d": tday, "h": np.where(is_asia, df["high"], np.nan), "l": np.where(is_asia, df["low"], np.nan)})
    rng = s.groupby("d").agg(h=("h", "max"), l=("l", "min"))
    hi = rng["h"].reindex(tday).to_numpy(copy=True)
    lo = rng["l"].reindex(tday).to_numpy(copy=True)
    utc_min = (df["time"].dt.hour * 60 + df["time"].dt.minute).to_numpy()
    visible = (utc_min >= 7 * 60) & (utc_min < 21 * 60)
    hi[~visible] = np.nan
    lo[~visible] = np.nan
    return hi, lo


def build_features(df: pd.DataFrame) -> Features:
    f = Features(
        time=df["time"].reset_index(drop=True),
        o=df["open"].to_numpy(float), h=df["high"].to_numpy(float), l=df["low"].to_numpy(float), c=df["close"].to_numpy(float),
        ao=df["ask_open"].to_numpy(float), ah=df["ask_high"].to_numpy(float), al=df["ask_low"].to_numpy(float),
        ac=df["ask_close"].to_numpy(float), vol=df["volume"].to_numpy(float),
    )
    a = f.arr
    a["atr"] = ind.atr(f.h, f.l, f.c, 14)
    a["atr_pct"] = ind.rolling_percentile_rank(a["atr"], 1440)
    for p in (20, 50, 100, 200):
        a[f"ema{p}"] = ind.ema(f.c, p)
    a["trend_m1"] = ema_trend(f.c, a["ema20"], a["ema50"], a["ema100"], a["ema200"])
    a["rsi"] = ind.rsi(f.c, 14)
    a["vol_ratio"] = ind.rolling_median_ratio(f.vol, 30)
    a["spread"] = f.ac - f.c
    a["session"] = sessions.session_codes(f.time)
    tday = sessions.trading_day(f.time)
    a["day_id"] = pd.factorize(tday)[0].astype(np.int64)
    a["month_id"] = (f.time.dt.year * 12 + f.time.dt.month - 1).to_numpy().astype(np.int64)
    # Minutes until the next bar (detects daily break / weekend gaps).
    a["minute"] = ((f.time - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy().astype(np.int64)
    for k in PIVOT_STRENGTHS:
        ph, pl = ind.confirmed_pivots(f.h, f.l, k)
        a[f"piv_h{k}"] = ph
        a[f"piv_l{k}"] = pl
    a["res_levels"] = ind.last_n_levels(a["piv_h5"], N_LEVELS)
    a["sup_levels"] = ind.last_n_levels(a["piv_l5"], N_LEVELS)
    a["res_levels10"] = ind.last_n_levels(a["piv_h10"], N_LEVELS)
    a["sup_levels10"] = ind.last_n_levels(a["piv_l10"], N_LEVELS)
    for tf, pre in (("M5", "m5_"), ("M15", "m15_"), ("H1", "h1_")):
        ctx = _htf_context(df.reset_index(drop=True), tf, pre)
        for col in ctx.columns:
            a[col] = ctx[col].to_numpy()
    a["pdh"], a["pdl"] = _previous_day_levels(df, tday)
    a["asia_hi"], a["asia_lo"] = _asia_range(df, tday, a["session"])
    from raygold.patterns.smc import premium_discount

    a["pd_pos"] = premium_discount(f.c, a["h1_range_hi"], a["h1_range_lo"])
    return f


def liquidity_targets(f: Features, idx: np.ndarray, direction: np.ndarray, price: np.ndarray, min_dist: np.ndarray) -> np.ndarray:
    """Nearest opposing liquidity level beyond ``price + min_dist`` (long) or below (short).

    Candidate pools: previous-day high/low, Asia range, completed-H1 swing
    levels, M1 swing levels (k=10) and active equal highs/lows.
    """
    a = f.arr
    highs = [a["pdh"][idx], a["asia_hi"][idx]] + [a[f"h1_swing_hi{j}"][idx] for j in range(3)]
    highs += [a["res_levels10"][idx, j] for j in range(N_LEVELS)]
    lows = [a["pdl"][idx], a["asia_lo"][idx]] + [a[f"h1_swing_lo{j}"][idx] for j in range(3)]
    lows += [a["sup_levels10"][idx, j] for j in range(N_LEVELS)]
    if "eqh" in a:
        highs.append(a["eqh"][idx])
        lows.append(a["eql"][idx])
    H = np.vstack(highs).T
    L = np.vstack(lows).T
    up_ok = H > (price + min_dist)[:, None]
    dn_ok = L < (price - min_dist)[:, None]
    up = np.where(up_ok, H, np.inf).min(axis=1)
    dn = np.where(dn_ok, L, -np.inf).max(axis=1)
    out = np.where(direction > 0, up, dn)
    out[~np.isfinite(out)] = np.nan
    return out
