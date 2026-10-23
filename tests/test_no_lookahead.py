"""Leakage tests: no feature, pattern or score at bar i may depend on bars > i.

Method: compute everything on the full series and on the series with all
bars after a cut point replaced by garbage (and, separately, removed).
Values up to the cut must be identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raygold.backtest.simulator import Costs
from raygold.data.timeframes import align_to_m1, resample
from raygold.features import build_features
from raygold.research.engine import attach_scores
from raygold.signals.events import build_events, default_variants, detect, ensure_context

CUTS = (5000, 8000, 9111)


def _perturb_after(df: pd.DataFrame, cut: int, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(df) - cut
    shock = rng.normal(0, 25, n).cumsum()
    for c in ("open", "high", "low", "close", "ask_open", "ask_high", "ask_low", "ask_close"):
        out.loc[cut:, c] = df.loc[cut:, c].to_numpy() + shock
    out.loc[cut:, "high"] = out.loc[cut:, ["open", "high", "low", "close"]].max(axis=1) + 3
    out.loc[cut:, "low"] = out.loc[cut:, ["open", "high", "low", "close"]].min(axis=1) - 3
    out.loc[cut:, "ask_high"] = out.loc[cut:, "high"] + 0.3
    out.loc[cut:, "ask_low"] = out.loc[cut:, "low"] + 0.3
    out.loc[cut:, "volume"] = rng.integers(1, 5000, n)
    return out


def _assert_equal_prefix(a: np.ndarray, b: np.ndarray, cut: int, name: str) -> None:
    a = np.asarray(a)[:cut]
    b = np.asarray(b)[:cut]
    if a.dtype.kind in "fc":
        ok = np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-9)
    else:
        ok = np.array_equal(a, b)
    assert ok, f"feature {name} changed before cut {cut} when future data changed (lookahead)"


@pytest.mark.parametrize("cut", CUTS)
def test_features_do_not_use_future_bars(synth: pd.DataFrame, cut: int) -> None:
    f_full = build_features(synth)
    f_pert = build_features(_perturb_after(synth, cut))
    f_trunc = build_features(synth.iloc[:cut].reset_index(drop=True))
    for name, arr in f_full.arr.items():
        _assert_equal_prefix(arr, f_pert.arr[name], cut, name)
        _assert_equal_prefix(arr, f_trunc.arr[name], cut, name)


@pytest.mark.parametrize("cut", CUTS)
def test_pattern_signals_do_not_use_future_bars(synth: pd.DataFrame, cut: int) -> None:
    f_full = build_features(synth)
    f_pert = build_features(_perturb_after(synth, cut))
    ensure_context(f_full)
    ensure_context(f_pert)
    for v in default_variants():
        s1, r1 = detect(f_full, v)
        s2, r2 = detect(f_pert, v)
        _assert_equal_prefix(s1, s2, cut, v.key + ":sig")
        _assert_equal_prefix(r1, r2, cut, v.key + ":ref")


def test_event_context_and_scores_do_not_use_future(synth: pd.DataFrame) -> None:
    cut = 8000
    f_full = build_features(synth)
    f_trunc = build_features(synth.iloc[:cut].reset_index(drop=True))
    costs = Costs()
    for v in default_variants()[:8]:
        e1 = build_events(f_full, v, start_idx=300)
        e2 = build_events(f_trunc, v, start_idx=300)
        e1 = e1[e1["idx"] < cut - 1].reset_index(drop=True)
        e2 = e2[e2["idx"] < cut - 1].reset_index(drop=True)
        pd.testing.assert_frame_equal(e1, e2, check_dtype=False)
        # historical-performance score uses only trades closed before entry;
        # compare events whose reference trades closed before the cut.
        s1 = attach_scores(f_full, e1, costs)
        s2 = attach_scores(f_trunc, e2, costs)
        early = e1["idx"] < cut - 400  # reference exit max_hold=120 bars
        np.testing.assert_allclose(s1.loc[early, "score"], s2.loc[early, "score"])


def test_htf_candle_only_visible_after_completion(synth: pd.DataFrame) -> None:
    m5 = resample(synth, "M5")
    m5["marker"] = np.arange(len(m5), dtype=float)
    al = align_to_m1(synth["time"], m5, ["marker"], "x_")
    t = synth["time"]
    for i in range(50, 400):
        mk = al["x_marker"].iloc[i]
        if np.isnan(mk):
            continue
        close_time = m5["close_time"].iloc[int(mk)]
        # the attached candle must have closed by the end of M1 bar i
        assert close_time <= t.iloc[i] + pd.Timedelta(minutes=1)
        # and it must be the most recent such candle
        nxt = int(mk) + 1
        if nxt < len(m5):
            assert m5["close_time"].iloc[nxt] > t.iloc[i] + pd.Timedelta(minutes=1)


def test_pivots_confirmed_k_bars_later() -> None:
    from raygold.indicators import confirmed_pivots

    h = np.array([1, 2, 3, 9, 3, 2, 1, 1, 1], float)
    l = h - 0.5
    ph, _ = confirmed_pivots(h, l, 3)
    assert np.isnan(ph[:6]).all()
    assert ph[6] == 9.0  # pivot at bar 3 known at bar 6
