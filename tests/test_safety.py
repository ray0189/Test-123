"""Live execution must be disabled by default and gated at several layers."""

from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace

import pytest

from raygold.config import LIVE_ACK_ENV, LIVE_ACK_VALUE, BrokerConfig, ConfigError, Mode, Settings, load_settings
from raygold.mt5.connection import MT5Connection
from raygold.mt5.execution import ExecutionRefused, OrderExecutor, OrderRequest

Account = namedtuple("Account", "trade_mode server currency equity margin_free login")
Tick = namedtuple("Tick", "bid ask time")
Res = namedtuple("Res", "retcode order price comment")


class FakeMT5:
    TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    TRADE_RETCODE_DONE = 10009

    def __init__(self, trade_mode: int) -> None:
        self.trade_mode = trade_mode
        self.sent: list[dict] = []

    def initialize(self, **kw):  # noqa: ANN003
        return True

    def symbol_select(self, *a):  # noqa: ANN002
        return True

    def account_info(self):
        return Account(self.trade_mode, "srv", "GBP", 100.0, 100.0, 1)

    def symbol_info(self, s):  # noqa: ANN001
        return SimpleNamespace(filling_mode=1)

    def symbol_info_tick(self, s):  # noqa: ANN001
        return Tick(2000.0, 2000.3, 0)

    def order_check(self, req):  # noqa: ANN001
        return Res(0, 0, 0.0, "ok")

    def order_send(self, req):  # noqa: ANN001
        self.sent.append(req)
        return Res(self.TRADE_RETCODE_DONE, 42, req.get("price", 0.0), "done")

    def last_error(self):
        return (0, "")

    def shutdown(self):
        pass


def _conn(trade_mode: int) -> tuple[MT5Connection, FakeMT5]:
    fake = FakeMT5(trade_mode)
    c = MT5Connection(BrokerConfig(), fake)
    c.connect()
    return c, fake


REQ = OrderRequest(1, 0.01, 1999.0, 2001.0)


def test_default_mode_is_read_only(tmp_path) -> None:  # noqa: ANN001
    assert Settings().mode is Mode.READ_ONLY
    assert load_settings(tmp_path / "missing.yaml", environ={}).mode is Mode.READ_ONLY
    assert load_settings("config/raygold.yaml", environ={}).mode is Mode.READ_ONLY
    assert not Settings().can_send_orders


def test_live_requires_acknowledgement(tmp_path) -> None:  # noqa: ANN001
    p = tmp_path / "c.yaml"
    p.write_text("mode: LIVE\n")
    with pytest.raises(ConfigError):
        load_settings(p, environ={})
    assert load_settings(p, environ={LIVE_ACK_ENV: LIVE_ACK_VALUE}).mode is Mode.LIVE


def test_env_override(tmp_path) -> None:  # noqa: ANN001
    s = load_settings(tmp_path / "none.yaml", environ={"RAYGOLD_RISK__MAX_POSITIONS": "5", "RAYGOLD_MODE": "paper"})
    assert s.risk.max_positions == 5 and s.mode is Mode.PAPER


def test_read_only_never_sends() -> None:
    conn, fake = _conn(0)
    with pytest.raises(ExecutionRefused):
        OrderExecutor(Mode.READ_ONLY, conn).send(REQ, 2000, 2000.3)
    assert fake.sent == []


def test_demo_mode_refuses_real_account() -> None:
    conn, fake = _conn(2)  # REAL account
    with pytest.raises(ExecutionRefused, match="NOT a demo"):
        OrderExecutor(Mode.DEMO, conn).send(REQ, 2000, 2000.3)
    assert fake.sent == []


def test_demo_mode_sends_on_demo_account() -> None:
    conn, fake = _conn(0)
    res = OrderExecutor(Mode.DEMO, conn).send(REQ, 2000, 2000.3)
    assert res.ok and len(fake.sent) == 1 and fake.sent[0]["sl"] == 1999.0


def test_live_refused_without_env_and_flag(monkeypatch) -> None:  # noqa: ANN001
    conn, fake = _conn(2)
    monkeypatch.delenv(LIVE_ACK_ENV, raising=False)
    with pytest.raises(ExecutionRefused):
        OrderExecutor(Mode.LIVE, conn, allow_live=True).send(REQ, 2000, 2000.3)
    monkeypatch.setenv(LIVE_ACK_ENV, LIVE_ACK_VALUE)
    with pytest.raises(ExecutionRefused):
        OrderExecutor(Mode.LIVE, conn, allow_live=False).send(REQ, 2000, 2000.3)
    assert fake.sent == []


def test_orders_require_stop_loss() -> None:
    with pytest.raises(ExecutionRefused, match="stop-loss"):
        OrderExecutor(Mode.PAPER).send(OrderRequest(1, 0.01, 0.0, 2001.0), 2000, 2000.3)


def test_scan_cli_refuses_live(tmp_path, capsys) -> None:  # noqa: ANN001
    from raygold.cli import build_parser

    args = build_parser().parse_args(["scan", "--mode", "PAPER"])
    assert args.mode == "PAPER"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["scan", "--mode", "LIVE"])
