"""Configuration: YAML file + ``RAYGOLD_*`` environment overrides.

Safety-critical defaults live here.  The system boots in ``READ_ONLY``
mode; ``DEMO`` requires an explicit flag *and* a demo account check at
runtime; ``LIVE`` additionally requires an environment variable
acknowledgement and is never the default.
"""

from __future__ import annotations

import dataclasses
import enum
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class Mode(str, enum.Enum):
    READ_ONLY = "READ_ONLY"
    PAPER = "PAPER"
    DEMO = "DEMO"
    LIVE = "LIVE"


LIVE_ACK_ENV = "RAYGOLD_I_ACCEPT_LIVE_TRADING_RISK"
LIVE_ACK_VALUE = "YES_I_UNDERSTAND_REAL_MONEY_IS_AT_RISK"


@dataclass
class BrokerConfig:
    # Credentials are only ever read from the environment.
    login_env: str = "MT5_LOGIN"
    password_env: str = "MT5_PASSWORD"
    server_env: str = "MT5_SERVER"
    terminal_path_env: str = "MT5_TERMINAL_PATH"
    symbol: str = "XAUUSD"
    # Fallback symbol specification used in research when no terminal is
    # attached.  Live runs always query ``symbol_info`` instead.
    contract_size: float = 100.0
    tick_size: float = 0.01
    tick_value_usd: float = 1.0  # per 1.00 lot per tick
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 50.0
    stops_level_points: int = 0
    point: float = 0.01
    leverage: float = 100.0
    account_currency: str = "GBP"
    gbpusd: float = 1.27  # used only when no live conversion rate is available
    commission_usd_per_lot_round_turn: float = 7.0
    slippage_price: float = 0.05  # adverse slippage per market fill, in price units


@dataclass
class RiskConfig:
    starting_equity: float = 100.0
    risk_per_trade: float = 1.0  # account currency
    max_positions: int = 25
    max_aggregate_risk: float = 5.0  # account currency across all open positions
    daily_loss_limit: float = 3.0
    max_drawdown_pct: float = 0.15  # circuit breaker from equity peak
    max_spread_price: float = 0.60
    min_margin_level_pct: float = 300.0  # refuse new trades below this margin level
    stop_out_level_pct: float = 50.0
    duplicate_window_minutes: int = 15
    allowed_sessions: tuple[str, ...] = ("london", "newyork")
    news_blackout_minutes: int = 30
    news_calendar_path: str = "config/news_events.csv"
    halt_file: str = "state/HALT"


@dataclass
class SignalConfig:
    timeframe: str = "M1"
    min_confidence: float = 60.0
    min_rr: float = 1.0
    strategy: str = "candidate"  # name of the strategy spec in strategy file
    strategy_file: str = "reports/selected_strategy.json"
    lookback_bars: int = 3000


@dataclass
class StorageConfig:
    db_path: str = "state/raygold.sqlite"
    state_path: str = "state/risk_state.json"
    reports_dir: str = "reports"


@dataclass
class Settings:
    mode: Mode = Mode.READ_ONLY
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    poll_seconds: float = 5.0
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def can_send_orders(self) -> bool:
        return self.mode in (Mode.DEMO, Mode.LIVE)


def _merge(obj: Any, data: dict[str, Any]) -> Any:
    for key, val in data.items():
        if not hasattr(obj, key):
            raise ValueError(f"unknown config key: {key}")
        cur = getattr(obj, key)
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val)
        elif isinstance(cur, Mode):
            setattr(obj, key, Mode(str(val).upper()))
        elif isinstance(cur, tuple):
            setattr(obj, key, tuple(val))
        else:
            setattr(obj, key, type(cur)(val) if cur is not None else val)
    return obj


def _env_overrides(settings: Settings, environ: dict[str, str]) -> None:
    """``RAYGOLD_RISK__MAX_POSITIONS=5`` style overrides (double underscore = nesting)."""
    for key, raw in environ.items():
        if not key.startswith("RAYGOLD_") or key == LIVE_ACK_ENV:
            continue
        path = key[len("RAYGOLD_"):].lower().split("__")
        target: Any = settings
        for part in path[:-1]:
            if not hasattr(target, part):
                break
            target = getattr(target, part)
        else:
            leaf = path[-1]
            if hasattr(target, leaf):
                cur = getattr(target, leaf)
                if isinstance(cur, tuple):
                    raw_val: Any = [v.strip() for v in raw.split(",") if v.strip()]
                elif isinstance(cur, bool):
                    raw_val = raw.lower() in ("1", "true", "yes")
                else:
                    raw_val = raw
                _merge(target, {leaf: raw_val})


def load_settings(path: str | Path | None = None, environ: dict[str, str] | None = None) -> Settings:
    settings = Settings()
    if path is not None and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text()) or {}
        _merge(settings, data)
    _env_overrides(settings, dict(os.environ if environ is None else environ))
    validate(settings, environ)
    return settings


class ConfigError(RuntimeError):
    pass


def validate(settings: Settings, environ: dict[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    r = settings.risk
    if r.risk_per_trade <= 0 or r.risk_per_trade > r.starting_equity * 0.05:
        raise ConfigError("risk_per_trade must be positive and at most 5% of starting equity")
    if r.max_aggregate_risk < r.risk_per_trade:
        raise ConfigError("max_aggregate_risk must be >= risk_per_trade")
    if not 1 <= r.max_positions <= 25:
        raise ConfigError("max_positions must be between 1 and 25")
    if settings.mode is Mode.LIVE and env.get(LIVE_ACK_ENV) != LIVE_ACK_VALUE:
        raise ConfigError(
            f"LIVE mode requires {LIVE_ACK_ENV}={LIVE_ACK_VALUE}; live trading is disabled by default"
        )
