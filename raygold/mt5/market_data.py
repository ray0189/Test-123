"""Market data collection from MT5 (or a replay source) into the canonical frame."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import pandas as pd

from raygold.mt5.connection import MT5Connection

log = logging.getLogger(__name__)


class BarSource(Protocol):
    def latest_bars(self, count: int) -> pd.DataFrame: ...

    def current_quote(self) -> tuple[float, float, pd.Timestamp]: ...


class MT5BarSource:
    """Completed M1 bars from the terminal.

    MT5 bars are bid-based and carry the bar's spread in points; the ask
    series is reconstructed as bid + spread * point.  The still-forming
    current bar is always dropped so signals use completed bars only.
    """

    def __init__(self, conn: MT5Connection, server_utc_offset_hours: float = 0.0) -> None:
        self.conn = conn
        self.offset = pd.Timedelta(hours=server_utc_offset_hours)

    def latest_bars(self, count: int) -> pd.DataFrame:
        mt5 = self.conn.mt5
        rates = mt5.copy_rates_from_pos(self.conn.cfg.symbol, mt5.TIMEFRAME_M1, 0, count + 1)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True) - self.offset
        df = df.iloc[:-1]  # drop the forming bar
        point = self.conn.cfg.point
        sp = np.maximum(df["spread"].to_numpy() * point, point)
        out = pd.DataFrame({"time": df["time"], "open": df["open"], "high": df["high"], "low": df["low"],
                            "close": df["close"], "ask_open": df["open"] + sp, "ask_high": df["high"] + sp,
                            "ask_low": df["low"] + sp, "ask_close": df["close"] + sp,
                            "volume": df["tick_volume"].astype(float)})
        return out.reset_index(drop=True)

    def current_quote(self) -> tuple[float, float, pd.Timestamp]:
        tick = self.conn.mt5.symbol_info_tick(self.conn.cfg.symbol)
        if tick is None:
            raise RuntimeError("no tick available")
        return float(tick.bid), float(tick.ask), pd.Timestamp(tick.time, unit="s", tz="UTC") - self.offset


class ReplayBarSource:
    """Replays a historical canonical frame bar by bar (offline paper trading / dashboard demo)."""

    def __init__(self, df: pd.DataFrame, start_index: int) -> None:
        self.df = df.reset_index(drop=True)
        self.pos = start_index

    def advance(self) -> bool:
        if self.pos + 1 >= len(self.df):
            return False
        self.pos += 1
        return True

    def latest_bars(self, count: int) -> pd.DataFrame:
        lo = max(0, self.pos - count)
        return self.df.iloc[lo: self.pos].reset_index(drop=True)  # completed bars before `pos`

    def current_quote(self) -> tuple[float, float, pd.Timestamp]:
        row = self.df.iloc[self.pos]
        return float(row["open"]), float(row["ask_open"]), pd.Timestamp(row["time"])
