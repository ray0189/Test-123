"""Position sizing from the broker's real contract specification.

Loss per 1.00 lot for a stop distance ``D`` (price units) is
``D / tick_size * tick_value`` in the *profit currency* (USD for XAUUSD),
converted to the account currency.  Costs (commission, expected stop
slippage) are added to the per-lot loss so that a stopped-out trade loses
at most the risk budget.  Lots are rounded *down* to the volume step; if
the minimum lot already exceeds the budget the trade is rejected — the
size is never rounded up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolSpec:
    name: str = "XAUUSD"
    contract_size: float = 100.0
    tick_size: float = 0.01
    tick_value: float = 1.0  # profit currency per tick per 1.00 lot
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 50.0
    stops_level_points: int = 0
    point: float = 0.01
    profit_currency: str = "USD"

    @property
    def min_stop_distance(self) -> float:
        return self.stops_level_points * self.point


@dataclass(frozen=True)
class SizeResult:
    lots: float
    risk_account_ccy: float  # loss if stopped (incl. costs) at this size
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.lots > 0


def loss_per_lot(spec: SymbolSpec, stop_distance: float, commission_per_lot: float, slippage: float,
                 fx_to_account: float) -> float:
    """Account-currency loss of 1.00 lot stopped out at ``stop_distance`` (incl. costs)."""
    ticks = (stop_distance + slippage) / spec.tick_size
    return (ticks * spec.tick_value + commission_per_lot) * fx_to_account


def size_position(spec: SymbolSpec, stop_distance: float, risk_budget: float, commission_per_lot: float = 7.0,
                  slippage: float = 0.05, fx_to_account: float = 1 / 1.27) -> SizeResult:
    """Largest volume whose stop-out loss (incl. costs) fits ``risk_budget``.

    ``fx_to_account`` converts the profit currency into the account
    currency (USD->GBP is 1/GBPUSD).
    """
    if not (stop_distance > 0):
        return SizeResult(0.0, 0.0, "non-positive stop distance")
    if stop_distance < spec.min_stop_distance:
        return SizeResult(0.0, 0.0, f"stop {stop_distance:.2f} inside broker stops level {spec.min_stop_distance:.2f}")
    per_lot = loss_per_lot(spec, stop_distance, commission_per_lot, slippage, fx_to_account)
    raw = risk_budget / per_lot
    steps = math.floor(raw / spec.volume_step + 1e-9)
    lots = round(steps * spec.volume_step, 8)
    lots = min(lots, spec.volume_max)
    if lots < spec.volume_min - 1e-12:
        min_risk = spec.volume_min * per_lot
        return SizeResult(0.0, min_risk, f"min lot {spec.volume_min} would risk {min_risk:.2f} > budget {risk_budget:.2f}")
    return SizeResult(lots, lots * per_lot)


def margin_required(spec: SymbolSpec, lots: float, price: float, leverage: float, fx_to_account: float) -> float:
    """Approximate initial margin (account ccy) for CFD-style margin calc."""
    return lots * spec.contract_size * price / leverage * fx_to_account


def max_stop_for_min_lot(spec: SymbolSpec, risk_budget: float, commission_per_lot: float, slippage: float,
                         fx_to_account: float) -> float:
    """Widest stop (price units) that the minimum lot can use within the budget."""
    budget_profit_ccy = risk_budget / fx_to_account / spec.volume_min - commission_per_lot
    return budget_profit_ccy / spec.tick_value * spec.tick_size - slippage
