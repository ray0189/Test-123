from __future__ import annotations

import numpy as np
import pytest

from raygold.backtest.simulator import (R_SL, R_TIME, R_TP, R_WEEKEND, SL_ATR, TP_NETPOS, TP_NONE, TP_R, Costs,
                                        ExitSpec, simulate)
from raygold.features import build_features
from tests.conftest import make_bars

FLAT = (100.0, 100.1, 99.9, 100.0)


def _run(rows, direction, spec, costs=Costs(commission=0.0, slippage=0.0), atr=1.0, spread=0.2, signal=0):
    df = make_bars(rows, spread=spread)
    f = build_features(df)
    idx = np.array([signal])
    out = simulate(f, idx, np.array([direction]), np.array([np.nan]), np.array([atr]), np.array([np.nan]), spec, costs)
    return {k: v[0] for k, v in out.items()}


def test_long_target_hit_pays_spread() -> None:
    # entry at ask open of bar 1 = 100 + 0.2; SL 1 ATR below; TP 1R above
    rows = [FLAT, FLAT, (100.0, 101.5, 100.0, 101.0), FLAT]
    r = _run(rows, 1, ExitSpec(SL_ATR, 1.0, TP_R, 1.0, min_stop_atr=0.0))
    assert r["entry"] == pytest.approx(100.2)
    assert r["reason"] == R_TP
    assert r["pnl"] == pytest.approx(1.0)  # TP = entry + 1.0 on the bid
    assert r["r"] == pytest.approx(1.0)


def test_ambiguous_bar_resolves_to_stop() -> None:
    rows = [FLAT, FLAT, (100.0, 102.0, 98.0, 100.0), FLAT]
    r = _run(rows, 1, ExitSpec(SL_ATR, 1.0, TP_R, 1.0, min_stop_atr=0.0))
    assert r["reason"] == R_SL
    assert r["r"] == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open() -> None:
    rows = [FLAT, FLAT, FLAT, (97.0, 97.5, 96.8, 97.2)]
    r = _run(rows, 1, ExitSpec(SL_ATR, 1.0, TP_R, 3.0, min_stop_atr=0.0))
    assert r["reason"] == R_SL
    assert r["pnl"] == pytest.approx(97.0 - 100.2)  # worse than the stop


def test_short_uses_ask_for_exits() -> None:
    # short entry at bid open 100.0; ask = bid + 0.2; TP = entry - 1 needs ask low <= 99.0 (bid low <= 98.8)
    rows = [FLAT, FLAT, (100.0, 100.0, 98.9, 99.0), (99.0, 99.0, 98.7, 98.8)]
    r = _run(rows, -1, ExitSpec(SL_ATR, 1.0, TP_R, 1.0, min_stop_atr=0.0))
    assert r["entry"] == pytest.approx(100.0)
    assert r["exit_idx"] == 3  # bar 2 ask low 99.1 did not reach 99.0
    assert r["reason"] == R_TP


def test_costs_reduce_r() -> None:
    rows = [FLAT, FLAT, (100.0, 100.2, 98.0, 99.0), FLAT]
    c = Costs(commission=0.07, slippage=0.05)
    r = _run(rows, 1, ExitSpec(SL_ATR, 1.0, TP_R, 1.0, min_stop_atr=0.0), costs=c)
    # entry 100.25, stop 99.25, fill 99.20, minus commission
    assert r["pnl"] == pytest.approx(99.20 - 100.25 - 0.07)
    assert r["r"] < -1.0


def test_breakeven_applies_from_next_bar_only() -> None:
    be = ExitSpec(SL_ATR, 1.0, TP_R, 3.0, be_trigger_r=0.5, min_stop_atr=0.0)
    # bar 2 reaches +0.6R then bar 3 falls to old stop: BE should have moved the stop
    rows = [FLAT, FLAT, (100.0, 100.85, 100.0, 100.5), (100.5, 100.5, 99.0, 99.1)]
    r = _run(rows, 1, be)
    assert r["reason"] == R_SL
    assert r["pnl"] == pytest.approx(0.0, abs=1e-9)  # scratch at break-even
    # same-bar excursion and stop: BE not yet active -> full loss
    rows2 = [FLAT, FLAT, (100.0, 100.85, 99.0, 99.1), FLAT]
    assert _run(rows2, 1, be)["r"] == pytest.approx(-1.0)


def test_time_exit_and_trailing() -> None:
    rows = [FLAT] * 2 + [(100.0 + i * 0.3, 100.4 + i * 0.3, 99.9 + i * 0.3, 100.3 + i * 0.3) for i in range(10)]
    r = _run(rows, 1, ExitSpec(SL_ATR, 2.0, TP_NONE, 0.0, max_hold=5, min_stop_atr=0.0))
    assert r["reason"] == R_TIME
    assert r["exit_idx"] == 1 + 5 - 1
    t = _run(rows + [(103.0, 103.0, 101.0, 101.2)], 1, ExitSpec(SL_ATR, 2.0, TP_NONE, 0.0, trail_atr=0.5, min_stop_atr=0.0))
    assert t["reason"] == R_SL and t["pnl"] > 0


def test_netpos_exit_is_net_positive() -> None:
    rows = [FLAT, FLAT, (100.0, 100.6, 100.0, 100.5), FLAT]
    c = Costs(commission=0.07, slippage=0.0)
    r = _run(rows, 1, ExitSpec(SL_ATR, 1.0, TP_NETPOS, 0.1, min_stop_atr=0.0), costs=c)
    assert r["reason"] == R_TP
    assert r["pnl"] == pytest.approx(0.1)


def test_weekend_flatten() -> None:
    df_rows = [FLAT] * 6
    df = make_bars(df_rows)
    df.loc[4:, "time"] = df.loc[4:, "time"] + np.timedelta64(2, "D")
    f = build_features(df)
    out = simulate(f, np.array([0]), np.array([1]), np.array([np.nan]), np.array([1.0]), np.array([np.nan]),
                   ExitSpec(SL_ATR, 1.0, TP_R, 3.0, min_stop_atr=0.0), Costs(0, 0))
    assert out["reason"][0] == R_WEEKEND
    assert out["exit_idx"][0] == 3


def test_tick_replay_first_touch() -> None:
    import pandas as pd

    from raygold.backtest.tick_replay import resolve_trade

    t = pd.date_range("2024-03-05 10:00", periods=6, freq="10s", tz="UTC")
    ticks = pd.DataFrame({"time": t, "bid": [100.0, 100.5, 101.1, 98.0, 100, 100], "ask": [100.2, 100.7, 101.3, 98.2, 100.2, 100.2]})
    when, px, why = resolve_trade(ticks, t[0], t[-1] + pd.Timedelta(seconds=1), 1, 99.0, 101.0)
    assert why == "target" and when == t[2]  # target touched before the stop on the tick stream
    when, px, why = resolve_trade(ticks, t[0], t[-1] + pd.Timedelta(seconds=1), -1, 101.5, 98.5)
    assert why == "target" and px == 98.2  # short exits on the ask


def test_tick_csv_to_m1(tmp_path) -> None:  # noqa: ANN001
    from raygold.data.loader import load_any

    p = tmp_path / "ticks.csv"
    p.write_text("time,bid,ask\n2024-03-05 10:00:01,100.0,100.2\n2024-03-05 10:00:30,100.5,100.7\n"
                 "2024-03-05 10:00:59,99.8,100.0\n2024-03-05 10:01:10,101.0,101.3\n")
    m1 = load_any(p)
    assert len(m1) == 2
    assert m1["high"].iloc[0] == 100.5 and m1["low"].iloc[0] == 99.8 and m1["ask_close"].iloc[0] == 100.0


def test_mt5_bar_csv(tmp_path) -> None:  # noqa: ANN001
    from raygold.data.loader import load_any

    p = tmp_path / "XAUUSD_M1.csv"
    p.write_text("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
                 "2024.03.05\t12:00:00\t2100.10\t2100.50\t2099.90\t2100.20\t55\t0\t25\n"
                 "2024.03.05\t12:01:00\t2100.20\t2100.30\t2100.00\t2100.10\t40\t0\t30\n")
    df = load_any(p, utc_offset_hours=2)
    assert str(df["time"].iloc[0]) == "2024-03-05 10:00:00+00:00"
    assert abs(df["ask_close"].iloc[0] - 2100.45) < 1e-9
