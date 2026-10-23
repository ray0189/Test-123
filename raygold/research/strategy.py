"""A concrete strategy = (entry variant, exit spec, filter spec).

:func:`run_strategy` is the single code path used for finalist analysis,
the one-time test evaluation, stress tests and (via
:mod:`raygold.signals.live`) the live scanner, so research and live use
identical signal logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from raygold.backtest.simulator import REASONS, Costs, ExitSpec, simulate
from raygold.features import Features
from raygold.research.engine import attach_scores, select_sequential
from raygold.research.grid import FilterSpec
from raygold.signals.events import Variant
from raygold.signals.multitf import events_for


@dataclass(frozen=True)
class StrategySpec:
    variant: Variant
    exit: ExitSpec
    filter: FilterSpec
    tf: str = "M1"  # timeframe the patterns are detected on (execution is always M1)

    @property
    def key(self) -> str:
        return f"{self.tf}:{self.variant.key} :: {self.exit.key} :: {self.filter.key}"

    def to_dict(self) -> dict:
        return {"variant": {"family": self.variant.family, "params": dict(self.variant.params)}, "tf": self.tf,
                "exit": self.exit.as_dict(), "filter": self.filter.as_dict(), "key": self.key}

    @staticmethod
    def from_dict(d: dict) -> "StrategySpec":
        v = Variant(d["variant"]["family"], tuple(sorted(d["variant"]["params"].items())))
        return StrategySpec(v, ExitSpec(**d["exit"]), FilterSpec(**d["filter"]), d.get("tf", "M1"))

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        payload = {"strategy": self.to_dict(), **(extra or {})}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, default=str))

    @staticmethod
    def load(path: str | Path) -> "StrategySpec":
        return StrategySpec.from_dict(json.loads(Path(path).read_text())["strategy"])


def run_strategy(f: Features, spec: StrategySpec, costs: Costs, start: str | None = None, end: str | None = None,
                 sequential: bool = True, entry_delay: int = 0, warmup: int = 1500) -> pd.DataFrame:
    """Trades of one strategy with entries in ``[start, end)``.

    ``sequential`` enforces one open position at a time.  ``entry_delay``
    adds extra bars of latency between signal and entry (stress testing).
    Trades still open at the end of the data are closed at the last bar
    (reason ``data_end``).
    """
    ev = events_for(f, spec.variant, spec.tf, warmup=warmup, end_idx=len(f) - 1 - entry_delay)
    ev = attach_scores(f, ev, costs)
    if len(ev) == 0:
        return pd.DataFrame()
    ev = ev[spec.filter.mask(ev)].reset_index(drop=True)
    sig_idx = ev["idx"].to_numpy() + entry_delay
    sim = simulate(f, sig_idx, ev["dir"].to_numpy(), ev["ref"].to_numpy(), ev["atr"].to_numpy(),
                   ev["liq_target"].to_numpy(), spec.exit, costs)
    valid = sim["valid"] & (sim["risk"] >= 2.0 * ev["spread"].to_numpy())
    entry_time = f.time.iloc[np.clip(sim["entry_idx"], 0, len(f) - 1)].reset_index(drop=True)
    in_window = np.ones(len(ev), bool)
    if start:
        in_window &= (entry_time >= pd.Timestamp(start, tz="UTC")).to_numpy()
    if end:
        in_window &= (entry_time < pd.Timestamp(end, tz="UTC")).to_numpy()
    mask = valid & in_window
    if sequential:
        take = select_sequential(sim["entry_idx"].astype(np.int64), sim["exit_idx"], mask)
    else:
        take = np.flatnonzero(mask)
    ex = sim["exit_idx"][take]
    out = pd.DataFrame({
        "signal_time": f.time.iloc[ev["idx"].to_numpy()[take]].to_numpy(),
        "entry_time": entry_time.iloc[take].to_numpy(),
        "exit_time": f.time.iloc[ex].to_numpy(),
        "dir": ev["dir"].to_numpy()[take],
        "entry": sim["entry"][take],
        "sl": sim["sl"][take],
        "tp": sim["tp"][take],
        "risk": sim["risk"][take],
        "pnl": sim["pnl"][take],
        "r": sim["r"][take],
        "mfe_r": sim["mfe_r"][take],
        "mae_r": sim["mae_r"][take],
        "reason": [REASONS[int(x)] for x in sim["reason"][take]],
        "bars": ex - sim["entry_idx"][take] + 1,
        "score": ev["score"].to_numpy()[take],
        "session": ev["session"].to_numpy()[take],
        "spread": ev["spread"].to_numpy()[take],
        "atr": ev["atr"].to_numpy()[take],
        "day": f["day_id"][sim["entry_idx"][take]],
    })
    return out
