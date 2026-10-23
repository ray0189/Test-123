"""Pre-trade risk checks and persistent risk state.

Every proposed order passes :meth:`RiskManager.evaluate`, which returns an
approved size or a rejection reason.  State (equity peak, daily P&L,
halts, recent signals) is persisted to JSON after every change so that a
restart cannot reset the circuit breakers.  There is no code path that
increases size after a loss (no martingale): size depends only on the
fixed risk budget and the stop distance.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from raygold.config import RiskConfig
from raygold.risk.sizing import SymbolSpec, margin_required, size_position

log = logging.getLogger(__name__)


@dataclass
class RiskState:
    equity_peak: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    day_realised: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    recent_signals: dict[str, str] = field(default_factory=dict)  # key -> iso time

    @staticmethod
    def load(path: str | Path) -> "RiskState":
        p = Path(path)
        if p.exists():
            try:
                return RiskState(**json.loads(p.read_text()))
            except (ValueError, TypeError) as exc:
                log.error("corrupt risk state %s (%s); starting HALTED for safety", p, exc)
                return RiskState(halted=True, halt_reason="corrupt state file")
        return RiskState()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1))
        tmp.replace(p)


@dataclass
class OpenPosition:
    direction: int
    lots: float
    entry: float
    sl: float
    risk_ccy: float


@dataclass
class Proposal:
    time: dt.datetime
    direction: int
    entry: float
    sl: float
    tp: float
    spread: float
    strategy: str


@dataclass
class Decision:
    approved: bool
    lots: float = 0.0
    risk_ccy: float = 0.0
    reason: str = ""


class NewsCalendar:
    """High-impact events from a CSV with columns ``time_utc,currency,impact,title``."""

    def __init__(self, path: str | Path | None) -> None:
        self.times: list[pd.Timestamp] = []
        if path and Path(path).exists():
            df = pd.read_csv(path)
            if len(df):
                df = df[df["impact"].str.lower().isin(["high", "red"]) & df["currency"].str.upper().isin(["USD", "XAU"])]
                self.times = sorted(pd.to_datetime(df["time_utc"], utc=True).tolist())

    def blackout(self, t: dt.datetime, minutes: int) -> bool:
        ts = pd.Timestamp(t)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        w = pd.Timedelta(minutes=minutes)
        return any(abs(ts - e) <= w for e in self.times)


class RiskManager:
    def __init__(self, cfg: RiskConfig, spec: SymbolSpec, state_path: str | Path, commission_per_lot: float,
                 slippage: float, leverage: float, news: NewsCalendar | None = None) -> None:
        self.cfg = cfg
        self.spec = spec
        self.state_path = Path(state_path)
        self.commission = commission_per_lot
        self.slippage = slippage
        self.leverage = leverage
        self.news = news or NewsCalendar(None)
        self.state = RiskState.load(self.state_path)

    # --- state ----------------------------------------------------------------
    def _persist(self) -> None:
        self.state.save(self.state_path)

    def halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        log.critical("trading halted", extra={"reason": reason})
        self._persist()

    def resume(self) -> None:
        self.state.halted = False
        self.state.halt_reason = ""
        self._persist()

    def on_equity(self, equity: float, now: dt.datetime) -> None:
        """Update peak/daily tracking and trip circuit breakers."""
        day = now.date().isoformat()
        if self.state.day != day:
            self.state.day = day
            self.state.day_start_equity = equity
            self.state.day_realised = 0.0
        if equity > self.state.equity_peak:
            self.state.equity_peak = equity
        dd = (self.state.equity_peak - equity) / self.state.equity_peak if self.state.equity_peak > 0 else 0.0
        if dd >= self.cfg.max_drawdown_pct and not self.state.halted:
            self.halt(f"equity drawdown {dd:.1%} >= {self.cfg.max_drawdown_pct:.0%}")
        self._persist()

    def on_trade_closed(self, pnl: float) -> None:
        self.state.day_realised += pnl
        self._persist()

    # --- checks -----------------------------------------------------------------
    def emergency_halt_requested(self) -> bool:
        return Path(self.cfg.halt_file).exists()

    def evaluate(self, p: Proposal, equity: float, free_margin: float, open_positions: list[OpenPosition],
                 fx_to_account: float, session_ok: bool) -> Decision:
        c = self.cfg
        if self.emergency_halt_requested():
            return Decision(False, reason=f"emergency halt file {c.halt_file} present")
        if self.state.halted:
            return Decision(False, reason=f"halted: {self.state.halt_reason}")
        day_loss = self.state.day_start_equity - equity if self.state.day_start_equity else 0.0
        if day_loss >= c.daily_loss_limit or -self.state.day_realised >= c.daily_loss_limit:
            return Decision(False, reason=f"daily loss limit {c.daily_loss_limit} reached")
        if not session_ok:
            return Decision(False, reason="outside allowed sessions")
        if self.news.blackout(p.time, c.news_blackout_minutes):
            return Decision(False, reason="news blackout window")
        if p.spread > c.max_spread_price:
            return Decision(False, reason=f"spread {p.spread:.2f} > max {c.max_spread_price:.2f}")
        if len(open_positions) >= c.max_positions:
            return Decision(False, reason=f"max positions {c.max_positions} reached")
        key = f"{p.strategy}|{p.direction}"
        last = self.state.recent_signals.get(key)
        if last and (pd.Timestamp(p.time) - pd.Timestamp(last)) < pd.Timedelta(minutes=c.duplicate_window_minutes):
            return Decision(False, reason="duplicate signal inside dedupe window")
        stop = (p.entry - p.sl) * p.direction
        tp_dist = (p.tp - p.entry) * p.direction if p.tp == p.tp else float("inf")
        if stop <= 0 or tp_dist <= 0:
            return Decision(False, reason="invalid stop/target geometry")
        open_risk = sum(op.risk_ccy for op in open_positions)
        budget = min(c.risk_per_trade, c.max_aggregate_risk - open_risk)
        if budget <= 0:
            return Decision(False, reason=f"aggregate open risk {open_risk:.2f} at limit {c.max_aggregate_risk:.2f}")
        size = size_position(self.spec, stop, budget, self.commission, self.slippage, fx_to_account)
        if not size.ok:
            return Decision(False, reason=size.reason)
        margin = margin_required(self.spec, size.lots, p.entry, self.leverage, fx_to_account)
        if margin > free_margin:
            return Decision(False, reason=f"margin {margin:.2f} > free margin {free_margin:.2f}")
        used_after = (equity - free_margin) + margin
        level_after = equity / used_after * 100 if used_after > 0 else float("inf")
        if level_after < c.min_margin_level_pct:
            return Decision(False, reason=f"margin level after trade {level_after:.0f}% < {c.min_margin_level_pct:.0f}%")
        return Decision(True, lots=size.lots, risk_ccy=size.risk_account_ccy, reason="approved")

    def register_signal(self, p: Proposal) -> None:
        self.state.recent_signals[f"{p.strategy}|{p.direction}"] = pd.Timestamp(p.time).isoformat()
        self._persist()
