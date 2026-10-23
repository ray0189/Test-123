"""MetaTrader 5 terminal connection (official ``MetaTrader5`` package).

The package is Windows-only and imported lazily, so research, tests and
read-only replay work on any OS.  Credentials are read from environment
variables named in :class:`raygold.config.BrokerConfig`; they are never
stored in config files or logs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from raygold.config import BrokerConfig
from raygold.risk.sizing import SymbolSpec

log = logging.getLogger(__name__)

ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2


class MT5Error(RuntimeError):
    pass


def import_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on platform
        raise MT5Error("MetaTrader5 package not available (Windows only: pip install MetaTrader5)") from exc
    return mt5


class MT5Connection:
    def __init__(self, cfg: BrokerConfig, mt5_module: Any | None = None) -> None:
        self.cfg = cfg
        self._mt5 = mt5_module
        self.connected = False

    @property
    def mt5(self) -> Any:
        if self._mt5 is None:
            self._mt5 = import_mt5()
        return self._mt5

    def connect(self) -> None:
        kwargs: dict[str, Any] = {}
        path = os.environ.get(self.cfg.terminal_path_env)
        if path:
            kwargs["path"] = path
        login = os.environ.get(self.cfg.login_env)
        if login:
            kwargs["login"] = int(login)
            kwargs["password"] = os.environ.get(self.cfg.password_env, "")
            kwargs["server"] = os.environ.get(self.cfg.server_env, "")
        if not self.mt5.initialize(**kwargs):
            raise MT5Error(f"MT5 initialize failed: {self.mt5.last_error()}")
        if not self.mt5.symbol_select(self.cfg.symbol, True):
            raise MT5Error(f"cannot select symbol {self.cfg.symbol}: {self.mt5.last_error()}")
        self.connected = True
        info = self.account_info()
        log.info("connected to MT5", extra={"server": info.get("server"), "trade_mode": info.get("trade_mode"),
                                           "currency": info.get("currency")})

    def shutdown(self) -> None:
        if self.connected:
            self.mt5.shutdown()
            self.connected = False

    def account_info(self) -> dict[str, Any]:
        info = self.mt5.account_info()
        if info is None:
            raise MT5Error(f"account_info failed: {self.mt5.last_error()}")
        return info._asdict()

    def is_demo_account(self) -> bool:
        return int(self.account_info().get("trade_mode", ACCOUNT_TRADE_MODE_REAL)) == ACCOUNT_TRADE_MODE_DEMO

    def symbol_spec(self) -> SymbolSpec:
        si = self.mt5.symbol_info(self.cfg.symbol)
        if si is None:
            raise MT5Error(f"symbol_info failed: {self.mt5.last_error()}")
        return SymbolSpec(
            name=si.name, contract_size=si.trade_contract_size, tick_size=si.trade_tick_size,
            tick_value=si.trade_tick_value_loss or si.trade_tick_value, volume_min=si.volume_min,
            volume_step=si.volume_step, volume_max=si.volume_max, stops_level_points=si.trade_stops_level,
            point=si.point, profit_currency=si.currency_profit,
        )

    def fx_to_account(self) -> float:
        """Conversion factor from the symbol's profit currency to the account currency.

        MT5 reports ``trade_tick_value`` already in the account currency, so
        when sizing with :meth:`symbol_spec` the factor is 1.0.
        """
        return 1.0
