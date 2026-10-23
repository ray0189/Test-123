"""Tick-level bid/ask replay.

When real tick data is available, a trade's exit can be resolved exactly
(which of stop or target was touched first) instead of using the
conservative "stop first" rule for ambiguous M1 bars.  Long positions
exit on the bid, shorts on the ask.

Dukascopy hourly tick files (``{hour:02d}h_ticks.bi5``) are 20-byte
big-endian records: ``int32 ms_offset, int32 ask, int32 bid, float32
ask_vol, float32 bid_vol`` (prices scaled like the candle files).
"""

from __future__ import annotations

import datetime as dt
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from raygold.data.dukascopy import POINT_DIVISOR, BASE_URL

TICK_REASON = {1: "stop", 2: "target", 3: "timeout", 0: "no_ticks"}


def decode_dukascopy_ticks(payload: bytes, hour_start: pd.Timestamp, point_divisor: float = 1000.0) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=["time", "bid", "ask"])
    raw = lzma.decompress(payload)
    arr = np.frombuffer(raw, dtype=np.dtype([("t", ">i4"), ("a", ">i4"), ("b", ">i4"), ("av", ">f4"), ("bv", ">f4")]))
    return pd.DataFrame({
        "time": hour_start + pd.to_timedelta(arr["t"].astype(np.int64), unit="ms"),
        "bid": arr["b"].astype(float) / point_divisor,
        "ask": arr["a"].astype(float) / point_divisor,
    })


def tick_url(symbol: str, hour_start: pd.Timestamp) -> str:
    return f"{BASE_URL}/{symbol}/{hour_start.year}/{hour_start.month - 1:02d}/{hour_start.day:02d}/{hour_start.hour:02d}h_ticks.bi5"


def load_tick_hours(symbol: str, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path) -> pd.DataFrame:
    """Load (and cache) Dukascopy ticks covering [start, end)."""
    import urllib.request

    frames = []
    h = start.floor("h")
    while h < end:
        path = cache_dir / symbol / f"{h:%Y/%m/%d}" / f"{h:%H}h_ticks.bi5"
        if not path.exists():
            req = urllib.request.Request(tick_url(symbol, h), headers={"User-Agent": "Mozilla/5.0 raygold-research"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
            except Exception:  # noqa: BLE001 - network errors leave a gap, reported by caller
                data = None
            if data is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        if path.exists():
            frames.append(decode_dukascopy_ticks(path.read_bytes(), h, POINT_DIVISOR.get(symbol, 1000.0)))
        h += pd.Timedelta(hours=1)
    if not frames:
        return pd.DataFrame(columns=["time", "bid", "ask"])
    return pd.concat(frames, ignore_index=True)


@njit(cache=True)
def _first_touch(t_ns, bid, ask, start_ns, end_ns, d, sl, tp):
    for k in range(t_ns.shape[0]):
        if t_ns[k] < start_ns:
            continue
        if t_ns[k] >= end_ns:
            return k, 3
        px = bid[k] if d > 0 else ask[k]
        if (d > 0 and px <= sl) or (d < 0 and px >= sl):
            return k, 1
        if tp == tp and ((d > 0 and px >= tp) or (d < 0 and px <= tp)):
            return k, 2
    return -1, 0


def resolve_trade(ticks: pd.DataFrame, entry_time: pd.Timestamp, exit_deadline: pd.Timestamp, direction: int,
                  sl: float, tp: float) -> tuple[pd.Timestamp | None, float | None, str]:
    """Return (time, price, reason) of the first stop/target touch on the tick stream."""
    t_ns = ticks["time"].dt.as_unit("ns").astype("int64").to_numpy()
    k, code = _first_touch(t_ns, ticks["bid"].to_numpy(float), ticks["ask"].to_numpy(float),
                           pd.Timestamp(entry_time).value, pd.Timestamp(exit_deadline).value, int(direction), float(sl),
                           float(tp))
    if k < 0:
        return None, None, TICK_REASON[code]
    px = ticks["bid"].iloc[k] if direction > 0 else ticks["ask"].iloc[k]
    return ticks["time"].iloc[k], float(px), TICK_REASON[code]


def ambiguous_bar_audit(trades: pd.DataFrame, symbol: str, cache_dir: Path, max_trades: int = 200) -> pd.DataFrame:
    """For stop-outs that happened on a bar which also touched the target, check the tick order.

    Returns one row per audited trade with the tick-resolved outcome; this
    measures how conservative the bar-level "stop first" rule is.
    """
    rows = []
    cand = trades[(trades["reason"] == "stop") & (trades["mfe_r"] * trades["risk"] >= (trades["tp"] - trades["entry"]).abs())]
    for _, t in cand.head(max_trades).iterrows():
        start = pd.Timestamp(t["entry_time"])
        end = pd.Timestamp(t["exit_time"]) + pd.Timedelta(minutes=1)
        ticks = load_tick_hours(symbol, start, end, cache_dir)
        when, px, why = resolve_trade(ticks, start, end, int(t["dir"]), float(t["sl"]), float(t["tp"]))
        rows.append({"entry_time": start, "bar_reason": "stop", "tick_reason": why, "tick_time": when, "tick_price": px})
    return pd.DataFrame(rows)


def ticks_hour_range(day: dt.date) -> list[pd.Timestamp]:
    base = pd.Timestamp(day, tz="UTC")
    return [base + pd.Timedelta(hours=h) for h in range(24)]
