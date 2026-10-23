from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from raygold.config import RiskConfig
from raygold.risk.manager import NewsCalendar, OpenPosition, Proposal, RiskManager, RiskState
from raygold.risk.sizing import SymbolSpec, margin_required, max_stop_for_min_lot, size_position

SPEC = SymbolSpec()
FX = 1 / 1.27


def test_min_lot_rejected_when_over_budget() -> None:
    # 0.01 lot, 3.00 stop: loss = (3.05/0.01)*1*0.01 + 0.07 = 3.12 USD = 2.46 GBP > 1 GBP
    r = size_position(SPEC, 3.0, 1.0, 7.0, 0.05, FX)
    assert not r.ok and "min lot" in r.reason


def test_size_rounds_down_never_up() -> None:
    r = size_position(SPEC, 0.5, 10.0, 7.0, 0.05, FX)
    assert r.ok
    assert r.risk_account_ccy <= 10.0 + 1e-9
    bigger = size_position(SPEC, 0.5, 10.0 + 1e-6, 7.0, 0.05, FX)
    assert bigger.lots >= r.lots


def test_max_stop_for_min_lot_consistent() -> None:
    ms = max_stop_for_min_lot(SPEC, 1.0, 7.0, 0.05, FX)
    assert size_position(SPEC, ms - 1e-6, 1.0, 7.0, 0.05, FX).ok
    assert not size_position(SPEC, ms + 0.02, 1.0, 7.0, 0.05, FX).ok
    assert 1.0 < ms < 1.3  # ~1.15 USD/oz for GBP1 at GBPUSD 1.27


def test_margin() -> None:
    assert margin_required(SPEC, 0.01, 4000.0, 20.0, 1.0) == pytest.approx(200.0)


def _rm(tmp_path: Path, **kw) -> RiskManager:
    cfg = RiskConfig(halt_file=str(tmp_path / "HALT"), **kw)
    return RiskManager(cfg, SPEC, tmp_path / "state.json", 7.0, 0.05, 500.0)


def _prop(t: dt.datetime, d: int = 1, stop: float = 0.8, spread: float = 0.2) -> Proposal:
    return Proposal(t, d, 2000.0, 2000.0 - d * stop, 2000.0 + d * stop, spread, "s")


T0 = dt.datetime(2025, 3, 4, 14, 0, tzinfo=dt.timezone.utc)


def test_approves_normal_trade(tmp_path: Path) -> None:
    rm = _rm(tmp_path)
    rm.on_equity(100.0, T0)
    d = rm.evaluate(_prop(T0), 100.0, 100.0, [], FX, True)
    assert d.approved and d.lots == 0.01


def test_rejections(tmp_path: Path) -> None:
    rm = _rm(tmp_path, max_positions=2)
    rm.on_equity(100.0, T0)
    assert "session" in rm.evaluate(_prop(T0), 100, 100, [], FX, False).reason
    assert "spread" in rm.evaluate(_prop(T0, spread=1.0), 100, 100, [], FX, True).reason
    two = [OpenPosition(1, 0.01, 2000, 1999, 0.7)] * 2
    assert "max positions" in rm.evaluate(_prop(T0), 100, 100, two, FX, True).reason
    full = [OpenPosition(1, 0.01, 2000, 1999, 5.0)]
    assert "aggregate" in rm.evaluate(_prop(T0), 100, 100, full, FX, True).reason
    assert "margin" in rm.evaluate(_prop(T0), 100, 1.0, [], FX, True).reason
    assert "min lot" in rm.evaluate(_prop(T0, stop=5.0), 100, 100, [], FX, True).reason


def test_duplicate_signal_window(tmp_path: Path) -> None:
    rm = _rm(tmp_path)
    rm.on_equity(100.0, T0)
    p = _prop(T0)
    assert rm.evaluate(p, 100, 100, [], FX, True).approved
    rm.register_signal(p)
    assert "duplicate" in rm.evaluate(_prop(T0 + dt.timedelta(minutes=5)), 100, 100, [], FX, True).reason
    assert rm.evaluate(_prop(T0 + dt.timedelta(minutes=20)), 100, 100, [], FX, True).approved


def test_daily_loss_and_drawdown_breakers_persist(tmp_path: Path) -> None:
    rm = _rm(tmp_path, daily_loss_limit=3.0, max_drawdown_pct=0.10)
    rm.on_equity(100.0, T0)
    rm.on_trade_closed(-3.0)
    assert "daily loss" in rm.evaluate(_prop(T0), 97, 97, [], FX, True).reason
    rm.on_equity(89.0, T0)
    assert rm.state.halted
    # a restart must not clear the breaker
    rm2 = _rm(tmp_path, daily_loss_limit=3.0, max_drawdown_pct=0.10)
    assert rm2.state.halted
    assert "halted" in rm2.evaluate(_prop(T0 + dt.timedelta(days=2)), 89, 89, [], FX, True).reason


def test_halt_file_and_corrupt_state(tmp_path: Path) -> None:
    rm = _rm(tmp_path)
    rm.on_equity(100.0, T0)
    (tmp_path / "HALT").write_text("x")
    assert "emergency halt" in rm.evaluate(_prop(T0), 100, 100, [], FX, True).reason
    (tmp_path / "bad.json").write_text("{not json")
    assert RiskState.load(tmp_path / "bad.json").halted


def test_news_blackout(tmp_path: Path) -> None:
    cal = tmp_path / "news.csv"
    cal.write_text("time_utc,currency,impact,title\n2025-03-04 13:30,USD,High,CPI\n2025-03-04 18:00,EUR,High,x\n")
    rm = RiskManager(RiskConfig(halt_file=str(tmp_path / "H")), SPEC, tmp_path / "s.json", 7.0, 0.05, 500.0,
                     NewsCalendar(cal))
    rm.on_equity(100.0, T0)
    assert "news" in rm.evaluate(_prop(T0 - dt.timedelta(minutes=20)), 100, 100, [], FX, True).reason
    assert rm.evaluate(_prop(T0 + dt.timedelta(hours=4)), 100, 100, [], FX, True).approved  # EUR event ignored


def test_no_martingale_size_independent_of_losses(tmp_path: Path) -> None:
    rm = _rm(tmp_path, daily_loss_limit=50.0)
    rm.on_equity(100.0, T0)
    before = rm.evaluate(_prop(T0), 100, 100, [], FX, True).lots
    rm.on_trade_closed(-2.0)
    rm.on_equity(98.0, T0)
    after = rm.evaluate(_prop(T0 + dt.timedelta(hours=1)), 98, 98, [], FX, True).lots
    assert after <= before
