"""Order execution with layered safety gates.

=========  ======================================================
Mode       Behaviour
=========  ======================================================
READ_ONLY  Never sends or simulates orders; signals are logged only.
PAPER      Orders are simulated locally and stored in SQLite.
DEMO       Real ``order_send`` **only** if the connected account
           reports ``trade_mode == DEMO``; otherwise refused.
LIVE       Refused unless the config says LIVE *and* the
           acknowledgement environment variable is set *and*
           ``allow_live=True`` is passed by the caller.
=========  ======================================================

Every real order is pre-validated with ``order_check`` and carries a
stop-loss; orders without a stop are rejected.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from raygold.config import LIVE_ACK_ENV, LIVE_ACK_VALUE, Mode
from raygold.mt5.connection import MT5Connection

log = logging.getLogger(__name__)

MAGIC = 20_260_1023


@dataclass
class OrderRequest:
    direction: int  # +1 buy, -1 sell
    lots: float
    sl: float
    tp: float | None
    comment: str = "raygold"
    deviation_points: int = 30


@dataclass
class OrderResult:
    ok: bool
    mode: Mode
    ticket: int | None = None
    price: float | None = None
    message: str = ""


class ExecutionRefused(RuntimeError):
    pass


class OrderExecutor:
    def __init__(self, mode: Mode, conn: MT5Connection | None = None, allow_live: bool = False) -> None:
        self.mode = mode
        self.conn = conn
        self.allow_live = allow_live
        self._paper_ticket = 0

    def _gate(self) -> None:
        if self.mode is Mode.READ_ONLY:
            raise ExecutionRefused("READ_ONLY mode: order sending is disabled")
        if self.mode is Mode.PAPER:
            return
        if self.conn is None or not self.conn.connected:
            raise ExecutionRefused("no MT5 connection")
        if self.mode is Mode.DEMO:
            if not self.conn.is_demo_account():
                raise ExecutionRefused("DEMO mode but the connected account is NOT a demo account")
            return
        if self.mode is Mode.LIVE:
            if os.environ.get(LIVE_ACK_ENV) != LIVE_ACK_VALUE or not self.allow_live:
                raise ExecutionRefused("LIVE trading not authorised")
            if self.conn.is_demo_account():
                raise ExecutionRefused("LIVE mode requested but account is demo; use DEMO mode")
            return
        raise ExecutionRefused(f"unknown mode {self.mode}")

    def send(self, req: OrderRequest, quote_bid: float, quote_ask: float) -> OrderResult:
        self._gate()
        if not (req.sl > 0):
            raise ExecutionRefused("every order must carry a stop-loss")
        if req.lots <= 0:
            raise ExecutionRefused("non-positive volume")
        if self.mode is Mode.PAPER:
            self._paper_ticket += 1
            price = quote_ask if req.direction > 0 else quote_bid
            return OrderResult(True, self.mode, ticket=-self._paper_ticket, price=price, message="paper fill")
        return self._send_mt5(req)

    def _filling_mode(self, mt5: Any, symbol: str) -> int:
        info = mt5.symbol_info(symbol)
        fm = int(getattr(info, "filling_mode", 0))
        if fm & 1:
            return mt5.ORDER_FILLING_FOK
        if fm & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _send_mt5(self, req: OrderRequest) -> OrderResult:
        assert self.conn is not None
        mt5 = self.conn.mt5
        symbol = self.conn.cfg.symbol
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if req.direction > 0 else tick.bid
        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(req.lots),
            "type": mt5.ORDER_TYPE_BUY if req.direction > 0 else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": float(req.sl),
            "deviation": req.deviation_points,
            "magic": MAGIC,
            "comment": req.comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(mt5, symbol),
        }
        if req.tp:
            request["tp"] = float(req.tp)
        check = mt5.order_check(request)
        if check is None or check.retcode not in (0, mt5.TRADE_RETCODE_DONE):
            msg = f"order_check failed: {getattr(check, 'comment', mt5.last_error())}"
            log.error(msg, extra={"request": request})
            return OrderResult(False, self.mode, message=msg)
        res = mt5.order_send(request)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            msg = f"order_send failed: {getattr(res, 'retcode', None)} {getattr(res, 'comment', mt5.last_error())}"
            log.error(msg, extra={"request": request})
            return OrderResult(False, self.mode, message=msg)
        log.info("order filled", extra={"ticket": res.order, "price": res.price, "volume": req.lots, "mode": self.mode.value})
        return OrderResult(True, self.mode, ticket=int(res.order), price=float(res.price), message="filled")

    def modify_stop(self, ticket: int, sl: float, tp: float | None) -> bool:
        self._gate()
        if self.mode is Mode.PAPER:
            return True
        assert self.conn is not None
        mt5 = self.conn.mt5
        req = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "symbol": self.conn.cfg.symbol, "sl": float(sl),
               "magic": MAGIC}
        if tp:
            req["tp"] = float(tp)
        res = mt5.order_send(req)
        return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE

    def close_position(self, ticket: int, direction: int, lots: float) -> OrderResult:
        self._gate()
        if self.mode is Mode.PAPER:
            return OrderResult(True, self.mode, ticket=ticket, message="paper close")
        assert self.conn is not None
        mt5 = self.conn.mt5
        symbol = self.conn.cfg.symbol
        tick = mt5.symbol_info_tick(symbol)
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lots), "position": ticket,
            "type": mt5.ORDER_TYPE_SELL if direction > 0 else mt5.ORDER_TYPE_BUY,
            "price": tick.bid if direction > 0 else tick.ask, "deviation": 30, "magic": MAGIC,
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": self._filling_mode(mt5, symbol),
        }
        res = mt5.order_send(req)
        ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(ok, self.mode, ticket=ticket, price=getattr(res, "price", None),
                           message="closed" if ok else f"close failed {getattr(res, 'retcode', None)}")
