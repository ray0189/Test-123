"""The scanner loop: data -> features -> signal -> risk -> (paper/demo) execution -> dashboard."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from raygold.backtest.simulator import Costs
from raygold.config import Mode, Settings
from raygold.mt5.execution import ExecutionRefused, OrderExecutor, OrderRequest
from raygold.mt5.market_data import BarSource
from raygold.mt5.positions import ManagedPosition, update_paper_position
from raygold.patterns.sessions import session_allowed
from raygold.reporting.dashboard import DashboardState
from raygold.research.strategy import StrategySpec
from raygold.risk.manager import OpenPosition, Proposal, RiskManager
from raygold.risk.sizing import SymbolSpec
from raygold.signals.live import latest_signal, snapshot
from raygold.storage import Store

log = logging.getLogger(__name__)


@dataclass
class AccountView:
    equity: float
    free_margin: float


class Scanner:
    def __init__(self, settings: Settings, source: BarSource, strategy: StrategySpec, store: Store,
                 risk: RiskManager, executor: OrderExecutor, symbol: SymbolSpec, fx_to_account: float,
                 costs: Costs, account_fn=None) -> None:  # type: ignore[no-untyped-def]
        self.s = settings
        self.source = source
        self.strategy = strategy
        self.store = store
        self.risk = risk
        self.exec = executor
        self.symbol = symbol
        self.fx = fx_to_account
        self.costs = costs
        self.account_fn = account_fn
        self.last_bar: pd.Timestamp | None = None
        self.paper: dict[int, ManagedPosition] = {}
        self.state = DashboardState(mode=settings.mode.value, symbol=settings.broker.symbol,
                                    currency=settings.broker.account_currency, hist=self._load_hist())
        self._restore_paper_positions()

    def _load_hist(self) -> dict:
        p = Path(self.s.signal.strategy_file)
        if not p.exists():
            return {}
        data = json.loads(p.read_text())
        out = {"candidate validated": "YES" if data.get("validated") else
               "NO - failed pre-registered train/validation gates (research only)"}
        for period in ("validation", "test"):
            st = data.get(period)
            if st:
                out[f"{period} trades"] = st.get("trades")
                out[f"{period} win rate"] = f"{st.get('win_rate', float('nan')):.1%} (95% CI {st.get('win_rate_lo', 0):.1%}-{st.get('win_rate_hi', 0):.1%})"
                out[f"{period} expectancy"] = f"{st.get('expectancy_r', float('nan')):+.3f} R"
        return out

    def _restore_paper_positions(self) -> None:
        """Persistent state recovery: reload open paper trades after a restart."""
        if self.s.mode is not Mode.PAPER:
            return
        for _, t in self.store.open_trades("PAPER").iterrows():
            d = 1 if t["direction"] == "BUY" else -1
            self.paper[int(t["id"])] = ManagedPosition(
                trade_id=int(t["id"]), ticket=int(t["ticket"] or 0), direction=d, lots=float(t["lots"]),
                entry=float(t["entry"]), sl=float(t["sl"]), tp=float(t["tp"]) if t["tp"] == t["tp"] else None,
                risk=abs(float(t["entry"]) - float(t["sl"])), open_time=pd.Timestamp(t["open_utc"]), atr=0.0)

    def _paper_equity(self) -> float:
        closed = self.store.table("trades")
        closed = closed[(closed["status"] == "closed") & (closed["mode"] == "PAPER")]
        return self.s.risk.starting_equity + float(closed["pnl"].fillna(0).sum())

    def account(self) -> AccountView:
        if self.account_fn is not None:
            return self.account_fn()
        eq = self._paper_equity()
        used = sum(p.lots * self.symbol.contract_size * p.entry / self.s.broker.leverage * self.fx for p in self.paper.values())
        return AccountView(equity=eq, free_margin=eq - used)

    def _manage_paper(self, bars: pd.DataFrame) -> None:
        if not self.paper:
            return
        for tid, pos in list(self.paper.items()):
            new = bars[bars["time"] >= pos.open_time] if pos.bars == 0 else bars.tail(1)
            for _, bar in new.iterrows():
                ev = update_paper_position(pos, bar, self.strategy.exit, self.costs.commission, self.costs.slippage)
                if ev:
                    pnl = ev.pnl_per_unit * pos.lots * self.symbol.contract_size * self.fx
                    self.store.close_trade(tid, str(ev.time), ev.price, pnl, ev.pnl_per_unit / pos.risk if pos.risk else 0.0)
                    self.risk.on_trade_closed(pnl)
                    self.state.messages.append(f"{ev.time} closed paper #{tid} {ev.reason} pnl {pnl:+.2f}")
                    del self.paper[tid]
                    break
            else:
                self.store.update_stop(tid, pos.sl)

    def step(self) -> DashboardState:
        bars = self.source.latest_bars(self.s.signal.lookback_bars)
        bid, ask, now = self.source.current_quote()
        if len(bars) < 500:
            self.state.decision, self.state.decision_reason = "NO TRADE", "insufficient history"
            return self.state
        new_bar = self.last_bar is None or bars["time"].iloc[-1] > self.last_bar
        if new_bar:
            self._manage_paper(bars)
        acct = self.account()
        self.risk.on_equity(acct.equity, now.to_pydatetime())
        self.state.equity = acct.equity
        self.state.peak = self.risk.state.equity_peak
        self.state.open_positions = [{"ticket": p.ticket, "direction": "BUY" if p.direction > 0 else "SELL",
                                      "lots": p.lots, "entry": p.entry, "sl": p.sl, "tp": p.tp} for p in self.paper.values()]
        if not new_bar:
            return self.state
        self.last_bar = bars["time"].iloc[-1]
        snap, f = snapshot(bars, bid, ask)
        self.state.snapshot = snap
        sig = latest_signal(f, self.strategy, self.costs, bid, ask)
        self.state.signal = sig
        self.state.lots = 0.0
        self.state.est_net_profit = self.state.est_net_loss = None
        if sig is None:
            self.state.decision, self.state.decision_reason = "NO TRADE", "no qualifying setup on the last completed bar"
            return self.state
        if sig.confidence < self.s.signal.min_confidence:
            self.state.decision = "NO TRADE"
            self.state.decision_reason = f"confidence {sig.confidence:.1f} < minimum {self.s.signal.min_confidence:.0f}"
            self._record(sig, 0.0, "rejected", self.state.decision_reason)
            return self.state
        if sig.rr == sig.rr and sig.rr < self.s.signal.min_rr:
            self.state.decision, self.state.decision_reason = "NO TRADE", f"R:R {sig.rr:.2f} < minimum {self.s.signal.min_rr}"
            self._record(sig, 0.0, "rejected", self.state.decision_reason)
            return self.state
        proposal = Proposal(time=now.to_pydatetime(), direction=sig.direction, entry=sig.entry, sl=sig.sl, tp=sig.tp,
                            spread=sig.spread, strategy=self.strategy.key)
        open_pos = [OpenPosition(p.direction, p.lots, p.entry, p.sl, p.risk * p.lots * self.symbol.contract_size * self.fx)
                    for p in self.paper.values()]
        sess_ok = bool(session_allowed(pd.Series([snap.session]).to_numpy(), self.s.risk.allowed_sessions)[0])
        dec = self.risk.evaluate(proposal, acct.equity, acct.free_margin, open_pos, self.fx, sess_ok)
        side = "BUY" if sig.direction > 0 else "SELL"
        if dec.lots > 0:
            units = dec.lots * self.symbol.contract_size
            if sig.tp == sig.tp:
                self.state.est_net_profit = ((sig.tp - sig.entry) * sig.direction - self.costs.commission) * units * self.fx
            self.state.est_net_loss = -dec.risk_ccy
            self.state.lots = dec.lots
        if not dec.approved:
            self.state.decision, self.state.decision_reason = "NO TRADE", f"{side} setup rejected: {dec.reason}"
            self._record(sig, 0.0, "rejected", dec.reason)
            return self.state
        self.risk.register_signal(proposal)
        if self.s.mode is Mode.READ_ONLY:
            self.state.decision, self.state.decision_reason = side, "READ_ONLY: signal logged, no order sent"
            self._record(sig, dec.lots, "signal_only", "read-only mode")
            return self.state
        try:
            res = self.exec.send(OrderRequest(sig.direction, dec.lots, sig.sl, sig.tp if sig.tp == sig.tp else None,
                                              comment="raygold"), bid, ask)
        except ExecutionRefused as exc:
            self.state.decision, self.state.decision_reason = "NO TRADE", f"execution refused: {exc}"
            self._record(sig, dec.lots, "refused", str(exc))
            return self.state
        sid = self._record(sig, dec.lots, "executed" if res.ok else "failed", res.message)
        if res.ok:
            tid = self.store.open_trade({"signal_id": sid, "mode": self.s.mode.value, "symbol": self.s.broker.symbol,
                                         "direction": side, "lots": dec.lots, "open_utc": str(now), "entry": res.price,
                                         "sl": sig.sl, "tp": sig.tp if sig.tp == sig.tp else None, "ticket": res.ticket,
                                         "comment": self.strategy.key[:200]})
            if self.s.mode is Mode.PAPER:
                self.paper[tid] = ManagedPosition(tid, int(res.ticket or 0), sig.direction, dec.lots, float(res.price),
                                                  sig.sl, sig.tp if sig.tp == sig.tp else None,
                                                  abs(float(res.price) - sig.sl), pd.Timestamp(now), sig.atr)
        self.state.decision = side if res.ok else "NO TRADE"
        self.state.decision_reason = res.message
        return self.state

    def _record(self, sig, lots: float, decision: str, reason: str) -> int | None:  # type: ignore[no-untyped-def]
        return self.store.record_signal({
            "bar_time": str(sig.bar_time), "symbol": self.s.broker.symbol, "direction": "BUY" if sig.direction > 0 else "SELL",
            "strategy": self.strategy.key, "confidence": sig.confidence, "entry": sig.entry, "sl": sig.sl, "tp": sig.tp,
            "rr": sig.rr, "est_net_profit": self.state.est_net_profit, "lots": lots, "decision": decision,
            "reason": reason, "mode": self.s.mode.value})
