"""Detect patterns on a higher signal timeframe, execute on M1.

Patterns can be computed on M5/M15 bars (larger structures, wider stops,
so fixed costs are a smaller fraction of risk) while trades are still
simulated on M1 bid/ask bars.  A signal on a ``tf`` bar is observable only
when that bar has closed; it is mapped to the last M1 bar inside the bucket
so that entry happens at the open of the first M1 bar after the close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raygold.data.timeframes import TF_MINUTES, resample_bidask
from raygold.features import Features, build_features
from raygold.signals.events import Variant, build_events


def signal_features(f_exec: Features, tf: str) -> Features:
    if tf == "M1":
        return f_exec
    key = f"sig_{tf}"
    if key not in f_exec.cache:
        f_exec.cache[key] = build_features(resample_bidask(f_exec.to_frame(), tf))
    return f_exec.cache[key]


def map_to_exec(f_sig: Features, f_exec: Features, sig_idx: np.ndarray, tf: str) -> np.ndarray:
    """Index of the last M1 bar that closes at or before the signal bar's close."""
    close_min = f_sig["minute"][sig_idx] + TF_MINUTES[tf]
    # M1 bar j closes at minute[j] + 1; we need minute[j] + 1 <= close_min
    return np.searchsorted(f_exec["minute"], close_min, side="left") - 1


def events_for(f_exec: Features, v: Variant, tf: str = "M1", warmup: int = 1500, end_idx: int | None = None) -> pd.DataFrame:
    """Setup events for ``v`` detected on ``tf`` with ``idx`` expressed in M1 (execution) bars.

    ``end_idx`` has the same meaning as in :func:`build_events` but in M1 bars.
    """
    if tf == "M1":
        return build_events(f_exec, v, start_idx=warmup, end_idx=end_idx)
    fs = signal_features(f_exec, tf)
    ev = build_events(fs, v, start_idx=50, end_idx=len(fs))
    if len(ev) == 0:
        return ev
    ex = map_to_exec(fs, f_exec, ev["idx"].to_numpy(), tf)
    n = len(f_exec)
    end = n - 1 if end_idx is None else min(end_idx, n)
    ev = ev.assign(sig_idx=ev["idx"].to_numpy(), idx=ex)
    ev = ev[(ev["idx"] >= warmup) & (ev["idx"] < end)].reset_index(drop=True)
    return ev
