"""Rich terminal dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from raygold.patterns.sessions import SESSION_NAMES
from raygold.signals.live import LiveSignal, MarketSnapshot


@dataclass
class DashboardState:
    mode: str
    symbol: str
    snapshot: MarketSnapshot | None = None
    signal: LiveSignal | None = None
    decision: str = "NO TRADE"
    decision_reason: str = ""
    lots: float = 0.0
    est_net_profit: float | None = None
    est_net_loss: float | None = None
    equity: float = 0.0
    peak: float = 0.0
    open_positions: list[dict] = field(default_factory=list)
    hist: dict = field(default_factory=dict)
    currency: str = "GBP"
    messages: list[str] = field(default_factory=list)


def _trend(v: float) -> Text:
    if v > 0:
        return Text("UP", style="bold green")
    if v < 0:
        return Text("DOWN", style="bold red")
    return Text("FLAT", style="yellow")


def render(s: DashboardState) -> Group:
    hdr = Text(f" RayGold  |  {s.symbol}  |  mode: {s.mode} ", style="bold white on blue")
    if s.mode == "READ_ONLY":
        hdr.append("  (no orders will be sent)", style="italic")
    mk = Table.grid(padding=(0, 2))
    if s.snapshot:
        sn = s.snapshot
        mk.add_row("Bar (UTC)", str(sn.bar_time))
        mk.add_row("Bid / Ask", f"{sn.bid:.2f} / {sn.ask:.2f}  (spread {sn.ask - sn.bid:.2f})")
        mk.add_row("ATR(14) M1", f"{sn.atr:.2f}")
        mk.add_row("Session", SESSION_NAMES.get(sn.session, "?"))
        tr = Table.grid(padding=(0, 1))
        tr.add_row("M1", _trend(sn.trend_m1), "M5", _trend(sn.trend_m5), "M15", _trend(sn.trend_m15), "H1", _trend(sn.trend_h1))
        mk.add_row("Trend", tr)
    pats = "\n".join(s.snapshot.patterns) if s.snapshot and s.snapshot.patterns else "none detected"
    sig = Table.grid(padding=(0, 2))
    style = {"BUY": "bold green", "SELL": "bold red"}.get(s.decision, "bold yellow")
    sig.add_row("Decision", Text(s.decision, style=style))
    if s.decision_reason:
        sig.add_row("Reason", s.decision_reason)
    if s.signal:
        g = s.signal
        sig.add_row("Confidence", f"{g.confidence:.1f}/100 (heuristic score, not a probability)")
        sig.add_row("Entry", f"{g.entry:.2f}")
        sig.add_row("Stop-loss", f"{g.sl:.2f}  ({g.risk_price:.2f} away)")
        sig.add_row("Take-profit", f"{g.tp:.2f}" if g.tp == g.tp else "none (managed exit)")
        sig.add_row("R:R", f"{g.rr:.2f}" if g.rr == g.rr else "-")
        sig.add_row("Lots", f"{s.lots:.2f}")
        if s.est_net_profit is not None:
            sig.add_row("Est. net at TP", f"{s.est_net_profit:+.2f} {s.currency} (after spread, commission, slippage)")
        if s.est_net_loss is not None:
            sig.add_row("Est. net at SL", f"{s.est_net_loss:+.2f} {s.currency}")
    acct = Table.grid(padding=(0, 2))
    dd = (s.peak - s.equity) / s.peak * 100 if s.peak > 0 else 0.0
    acct.add_row("Equity", f"{s.equity:.2f} {s.currency}")
    acct.add_row("Drawdown", f"{dd:.1f}% from peak {s.peak:.2f}")
    acct.add_row("Open positions", str(len(s.open_positions)))
    pos = Table("ticket", "dir", "lots", "entry", "sl", "tp", box=None)
    for p in s.open_positions[:25]:
        pos.add_row(str(p.get("ticket")), p.get("direction", ""), f"{p.get('lots', 0):.2f}", f"{p.get('entry', 0):.2f}",
                    f"{p.get('sl', 0):.2f}", f"{p.get('tp') or 0:.2f}")
    hist = Table.grid(padding=(0, 2))
    for k, v in s.hist.items():
        hist.add_row(k, str(v))
    msgs = "\n".join(s.messages[-5:]) if s.messages else ""
    return Group(
        hdr,
        Panel(mk, title="Market"),
        Panel(pats, title="Detected patterns (last 30 bars)"),
        Panel(sig, title="Signal"),
        Panel(Group(acct, pos), title="Account"),
        Panel(hist if s.hist else Text("no validated statistics loaded"), title="Strategy statistics (out-of-sample)"),
        Panel(msgs or "-", title="Log"),
    )
