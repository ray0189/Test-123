from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raygold.data.synthetic import synthetic_m1


@pytest.fixture(scope="session")
def synth() -> pd.DataFrame:
    return synthetic_m1(12_000, seed=3)


def make_bars(rows: list[tuple[float, float, float, float]], spread: float = 0.2, start: str = "2024-03-05 09:00") -> pd.DataFrame:
    """Hand-crafted bid OHLC bars with a constant spread, one minute apart."""
    t = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    o, h, l, c = (np.array(x, float) for x in zip(*rows))
    return pd.DataFrame({"time": t, "open": o, "high": h, "low": l, "close": c, "ask_open": o + spread,
                         "ask_high": h + spread, "ask_low": l + spread, "ask_close": c + spread,
                         "volume": np.full(len(rows), 100.0)})
