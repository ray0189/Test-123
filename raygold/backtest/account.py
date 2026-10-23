"""Account-level replay of a trade list with real sizing, margin and breakers.

Takes trades produced by :func:`raygold.research.strategy.run_strategy`
(price-level P&L per unit) and replays them chronologically on a small
account: lot sizing from the risk budget, minimum-lot rejection, margin
and leverage, maximum simultaneous positions, aggregate open risk, daily
loss limit and an equity drawdown circuit breaker.  Equity is
closed-trade equity (open P&L is not marked to market), so the reported
drawdown is a lower bound on intratrade drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from raygold.config import RiskConfig
from raygold.risk.sizing import SymbolSpec, margin_required, size_position


@dataclass
class AccountResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: dict


def replay(trades: pd.DataFrame, risk: RiskConfig, spec: SymbolSpec, leverage: float, gbpusd: float,
           commission_per_lot: float, slippage: float) -> AccountResult:
    fx = 1.0 / gbpusd
    tr = trades.sort_values("entry_time").reset_index(drop=True)
    events = []
    for i, t in tr.iterrows():
        events.append((pd.Timestamp(t["entry_time"]), 1, i))
        events.append((pd.Timestamp(t["exit_time"]), 0, i))  # exits first at equal timestamps
    events.sort(key=lambda e: (e[0], e[1]))
    equity = risk.starting_equity
    peak = equity
    halted = False
    open_pos: dict[int, tuple[float, float, float]] = {}  # i -> (lots, risk_ccy, margin)
    day = None
    day_start = equity
    status = np.array(["skipped"] * len(tr), dtype=object)
    lots_arr = np.zeros(len(tr))
    pnl_arr = np.zeros(len(tr))
    reasons = np.array([""] * len(tr), dtype=object)
    curve = []
    max_dd = 0.0
    for ts, kind, i in events:
        t = tr.iloc[i]
        d = ts.date()
        if d != day:
            day = d
            day_start = equity
        if kind == 0:
            if i in open_pos:
                lots, _, _ = open_pos.pop(i)
                # pnl per unit already includes commission per unit; scale by units held
                pnl = t["pnl"] * lots * spec.contract_size * fx
                equity += pnl
                pnl_arr[i] = pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)
                curve.append((ts, equity))
                if (peak - equity) / peak >= risk.max_drawdown_pct:
                    halted = True
            continue
        if halted:
            reasons[i] = "drawdown circuit breaker"
            continue
        if day_start - equity >= risk.daily_loss_limit:
            reasons[i] = "daily loss limit"
            continue
        if len(open_pos) >= risk.max_positions:
            reasons[i] = "max positions"
            continue
        open_risk = sum(v[1] for v in open_pos.values())
        budget = min(risk.risk_per_trade, risk.max_aggregate_risk - open_risk)
        if budget <= 0:
            reasons[i] = "aggregate risk limit"
            continue
        size = size_position(spec, float(t["risk"]), budget, commission_per_lot, slippage, fx)
        if not size.ok:
            reasons[i] = "min lot exceeds risk budget"
            continue
        margin = margin_required(spec, size.lots, float(t["entry"]), leverage, fx)
        used = sum(v[2] for v in open_pos.values())
        if used + margin > equity:
            reasons[i] = "insufficient margin"
            continue
        if equity / (used + margin) * 100 < risk.min_margin_level_pct:
            reasons[i] = "margin level below minimum"
            continue
        open_pos[i] = (size.lots, size.risk_account_ccy, margin)
        lots_arr[i] = size.lots
        status[i] = "taken"
    tr = tr.assign(account_status=status, lots=lots_arr, pnl_ccy=pnl_arr, reject_reason=reasons)
    taken = tr[tr["account_status"] == "taken"]
    rej = tr[tr["account_status"] != "taken"]["reject_reason"].value_counts().to_dict()
    summary = {
        "leverage": leverage, "signals": int(len(tr)), "taken": int(len(taken)),
        "rejected": int(len(tr) - len(taken)), "reject_reasons": rej,
        "start_equity": risk.starting_equity, "final_equity": round(equity, 2),
        "net_profit": round(equity - risk.starting_equity, 2),
        "max_drawdown_pct": round(100 * max_dd, 2),
        "win_rate": float((taken["pnl_ccy"] > 0).mean()) if len(taken) else float("nan"),
        "avg_lots": float(taken["lots"].mean()) if len(taken) else 0.0,
        "halted_by_breaker": halted,
    }
    return AccountResult(tr, pd.DataFrame(curve, columns=["time", "equity"]), summary)
