"""The live scanner must produce the same setups as the backtest on the same bars."""

from __future__ import annotations

import numpy as np
import pytest

from raygold.backtest.simulator import Costs, ExitSpec, plan_levels
from raygold.data.synthetic import synthetic_m1
from raygold.features import build_features
from raygold.research.engine import attach_scores
from raygold.research.grid import FilterSpec
from raygold.research.strategy import StrategySpec
from raygold.signals.events import _v, build_events
from raygold.signals.live import latest_signal

pytestmark = pytest.mark.slow


def test_live_signals_match_backtest_events() -> None:
    df = synthetic_m1(30_000, seed=11, spread=0.1)
    spec = StrategySpec(_v("sweep_reversal", k=3), ExitSpec(), FilterSpec())
    costs = Costs()
    f_full = build_features(df)
    ev = attach_scores(f_full, build_events(f_full, spec.variant, 1500), costs)
    plan = plan_levels(f_full, ev["idx"].to_numpy(), ev["dir"].to_numpy(), ev["ref"].to_numpy(), ev["atr"].to_numpy(),
                       ev["liq_target"].to_numpy(), spec.exit, costs)
    ev = ev[plan["valid"] & (plan["risk"] >= 2 * ev["spread"].to_numpy())]
    lo, hi = 24_000, 24_700
    expected = {(int(i), int(d)): s for i, d, s in ev[(ev.idx >= lo) & (ev.idx < hi)][["idx", "dir", "score"]].to_numpy()}
    lookback = 8_000
    got = {}
    for end in range(lo + 1, hi + 1):  # bars [end-lookback, end) are complete; the last one is end-1
        bars = df.iloc[end - lookback:end].reset_index(drop=True)
        f = build_features(bars)
        s = latest_signal(f, spec, costs, float(bars["close"].iloc[-1]), float(bars["ask_close"].iloc[-1]))
        if s is not None:
            got[(end - 1, s.direction)] = s.confidence
    assert len(expected) > 5
    common = set(expected) & set(got)
    assert len(common) >= 0.95 * len(expected), (sorted(set(expected) - common), sorted(set(got) - common))
    assert len(got) <= len(expected) + 1
    diffs = np.array([abs(expected[k] - got[k]) for k in common])
    assert np.median(diffs) < 2.0
