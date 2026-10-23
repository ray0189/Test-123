"""Trading-session classification with correct daylight-saving handling.

Sessions are defined in local exchange time and converted per bar:
* Asia:    00:00-07:00 UTC
* London:  08:00-16:30 Europe/London
* New York: 08:00-17:00 America/New_York
* Overlap: London and New York both open
Codes: 0 off-hours, 1 Asia, 2 London only, 3 New York only, 4 overlap.

The trading day rolls at 17:00 New York (the gold daily close).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OFF, ASIA, LONDON, NEWYORK, OVERLAP = 0, 1, 2, 3, 4
SESSION_NAMES = {OFF: "off", ASIA: "asia", LONDON: "london", NEWYORK: "newyork", OVERLAP: "overlap"}


def _local_minutes(t: pd.Series, tz: str) -> np.ndarray:
    lt = t.dt.tz_convert(tz)
    return (lt.dt.hour * 60 + lt.dt.minute).to_numpy()


def session_codes(time_utc: pd.Series) -> np.ndarray:
    lon = _local_minutes(time_utc, "Europe/London")
    ny = _local_minutes(time_utc, "America/New_York")
    utc_min = (time_utc.dt.hour * 60 + time_utc.dt.minute).to_numpy()
    in_lon = (lon >= 8 * 60) & (lon < 16 * 60 + 30)
    in_ny = (ny >= 8 * 60) & (ny < 17 * 60)
    in_asia = utc_min < 7 * 60
    out = np.zeros(len(time_utc), dtype=np.int8)
    out[in_asia] = ASIA
    out[in_lon] = LONDON
    out[in_ny] = NEWYORK
    out[in_lon & in_ny] = OVERLAP
    return out


def session_allowed(codes: np.ndarray, allowed: tuple[str, ...]) -> np.ndarray:
    ok = np.zeros(len(codes), dtype=bool)
    names = {n.lower() for n in allowed}
    for code, name in SESSION_NAMES.items():
        if name in names:
            ok |= codes == code
    if "london" in names:
        ok |= codes == OVERLAP
    if "newyork" in names:
        ok |= codes == OVERLAP
    return ok


def trading_day(time_utc: pd.Series) -> np.ndarray:
    """Trading-day id: calendar date of (New York time + 7h), i.e. rolls at 17:00 NY."""
    ny = time_utc.dt.tz_convert("America/New_York") + pd.Timedelta(hours=7)
    return ny.dt.date.astype(str).to_numpy()
