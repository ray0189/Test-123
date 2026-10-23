"""RayGold command-line interface.

    raygold download        fetch Dukascopy XAUUSD M1 bid/ask history
    raygold research        run the sweep + selection on train/validation only
    raygold evaluate-test   one-time evaluation of the selected candidate on the test period
    raygold backtest        run a strategy file on a date range (test period refused unless burned)
    raygold feasibility     position-sizing / margin feasibility for the configured account
    raygold scan            scanner + dashboard (READ_ONLY by default; --mode PAPER/DEMO)
    raygold export          export SQLite tables to CSV/JSON
    raygold halt|resume     emergency halt switch
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

from raygold.config import ConfigError, Mode, load_settings
from raygold.logging_setup import setup_logging

log = logging.getLogger("raygold")
DEFAULT_DATA = "data/XAUUSD_M1_bidask.parquet"


def _settings(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    s = load_settings(args.config)
    if getattr(args, "mode", None):
        s.mode = Mode(args.mode.upper())
        from raygold.config import validate

        validate(s)
    return s


def _load(path: str):  # type: ignore[no-untyped-def]
    from raygold.data.loader import load_any

    return load_any(path)


def cmd_download(args: argparse.Namespace) -> int:
    from raygold.data.dukascopy import build_m1, download

    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    cache = Path(args.cache)
    for attempt in range(3):
        failed = download(args.symbol, start, end, cache, workers=args.workers)
        if not failed:
            break
        log.warning("pass %d: %d files failed, retrying", attempt + 1, len(failed))
    df = build_m1(args.symbol, start, end, cache)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} bars {df['time'].iloc[0]} .. {df['time'].iloc[-1]} -> {args.out}")
    return 0


def _splits(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from raygold.research.protocol import Splits

    return Splits(args.train_start, args.val_start, args.test_start, args.test_end)


def cmd_research(args: argparse.Namespace) -> int:
    from raygold.research.protocol import SelectionRules
    from raygold.research.run import run_research
    from raygold.storage import Store, file_hash

    s = _settings(args)
    df = _load(args.data)
    store = Store(s.storage.db_path)
    summary = run_research(df, s, _splits(args), SelectionRules(), args.out, store, file_hash(args.data))
    print(json.dumps({k: summary[k] for k in ("configs_evaluated", "candidate", "train", "validation", "walk_forward", "null")},
                     indent=2, default=str))
    return 0


def cmd_evaluate_test(args: argparse.Namespace) -> int:
    from raygold.research.holdout import HoldoutBurned, evaluate_test
    from raygold.storage import Store, file_hash

    s = _settings(args)
    df = _load(args.data)
    store = Store(s.storage.db_path)
    try:
        res = evaluate_test(df, s, _splits(args), args.strategy, args.out, store, file_hash(args.data),
                            allow_reuse=args.i_know_the_test_set_is_burned)
    except HoldoutBurned as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: res[k] for k in ("period", "strategy", "reused_test_period", "test")}, indent=2, default=str))
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    import pandas as pd

    from raygold.data.loader import slice_period
    from raygold.features import build_features
    from raygold.research.holdout import period_key
    from raygold.research.run import costs_from_settings, detailed_stats
    from raygold.research.strategy import StrategySpec, run_strategy
    from raygold.storage import Store

    s = _settings(args)
    splits = _splits(args)
    store = Store(s.storage.db_path)
    end = pd.Timestamp(args.end or splits.test_end, tz="UTC")
    if end > pd.Timestamp(splits.test_start, tz="UTC") and not store.holdout_status(period_key(s.broker.symbol, splits)):
        print("REFUSED: range overlaps the untouched test period; use `raygold evaluate-test` (one time).", file=sys.stderr)
        return 2
    df = _load(args.data)
    sub = slice_period(df, None, str(end.date()))
    f = build_features(sub)
    spec = StrategySpec.load(args.strategy)
    trades = run_strategy(f, spec, costs_from_settings(s), args.start, args.end)
    if args.trades_out:
        trades.to_csv(args.trades_out, index=False)
    print(json.dumps(detailed_stats(trades), indent=2))
    return 0


def cmd_feasibility(args: argparse.Namespace) -> int:
    from raygold.research.run import symbol_from_settings
    from raygold.risk.sizing import margin_required, max_stop_for_min_lot

    s = _settings(args)
    sym = symbol_from_settings(s)
    fx = 1.0 / s.broker.gbpusd
    ms = max_stop_for_min_lot(sym, s.risk.risk_per_trade, s.broker.commission_usd_per_lot_round_turn,
                              s.broker.slippage_price, fx)
    print(f"Account {s.risk.starting_equity:.0f} {s.broker.account_currency}, risk/trade {s.risk.risk_per_trade:.2f}")
    print(f"Min lot {sym.volume_min} = {sym.volume_min * sym.contract_size:g} oz; widest stop at min lot: {ms:.2f} USD/oz")
    for price in (2000, 3000, 4000):
        for lev in (20, 30, 100, 500):
            m = margin_required(sym, sym.volume_min, price, lev, fx)
            print(f"  gold {price}: margin for 0.01 lot at 1:{lev} = {m:6.2f} {s.broker.account_currency}"
                  f" -> max concurrent min-lot positions by margin: {int(s.risk.starting_equity // m) if m else 0}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.live import Live

    from raygold.app import AccountView, Scanner
    from raygold.mt5.execution import OrderExecutor
    from raygold.reporting.dashboard import render
    from raygold.research.run import costs_from_settings, symbol_from_settings
    from raygold.research.strategy import StrategySpec
    from raygold.risk.manager import NewsCalendar, RiskManager
    from raygold.storage import Store

    s = _settings(args)
    if s.mode is Mode.LIVE:
        print("LIVE mode is not available from `scan`; it is disabled by default.", file=sys.stderr)
        return 2
    spec = StrategySpec.load(args.strategy or s.signal.strategy_file)
    store = Store(s.storage.db_path)
    costs = costs_from_settings(s)
    account_fn = None
    conn = None
    if args.replay:
        import pandas as pd

        from raygold.mt5.market_data import ReplayBarSource

        df = _load(args.replay)
        start = df.index[df["time"] >= pd.Timestamp(args.replay_start, tz="UTC")][0]
        source = ReplayBarSource(df, int(start))
        sym = symbol_from_settings(s)
        fx = 1.0 / s.broker.gbpusd
        if s.mode is Mode.DEMO:
            print("DEMO mode needs a live MT5 demo account, not a replay.", file=sys.stderr)
            return 2
    else:
        from raygold.mt5.connection import MT5Connection
        from raygold.mt5.market_data import MT5BarSource

        conn = MT5Connection(s.broker)
        conn.connect()
        if s.mode is Mode.DEMO and not conn.is_demo_account():
            print("REFUSED: DEMO mode requested but the connected account is not a demo account.", file=sys.stderr)
            return 2
        source = MT5BarSource(conn, args.server_utc_offset)
        sym = conn.symbol_spec()
        fx = conn.fx_to_account()
        if s.mode in (Mode.DEMO, Mode.READ_ONLY):
            def account_fn() -> AccountView:  # noqa: E306
                a = conn.account_info()
                return AccountView(equity=float(a["equity"]), free_margin=float(a["margin_free"]))
    risk = RiskManager(s.risk, sym, s.storage.state_path, s.broker.commission_usd_per_lot_round_turn,
                       s.broker.slippage_price, s.broker.leverage, NewsCalendar(s.risk.news_calendar_path))
    executor = OrderExecutor(s.mode, conn)
    scanner = Scanner(s, source, spec, store, risk, executor, sym, fx, costs, account_fn)
    console = Console()
    steps = 0
    try:
        with Live(render(scanner.state), console=console, refresh_per_second=2, screen=False) as live:
            while True:
                try:
                    state = scanner.step()
                except Exception as exc:  # keep the loop alive; log and continue
                    log.exception("scanner step failed")
                    scanner.state.messages.append(f"error: {exc}")
                    state = scanner.state
                live.update(render(state))
                steps += 1
                if args.max_steps and steps >= args.max_steps:
                    break
                if args.replay:
                    if not source.advance():  # type: ignore[union-attr]
                        break
                else:
                    time.sleep(s.poll_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        if conn:
            conn.shutdown()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from raygold.storage import Store

    s = _settings(args)
    for p in Store(s.storage.db_path).export(args.out):
        print(p)
    return 0


def cmd_halt(args: argparse.Namespace) -> int:
    s = _settings(args)
    p = Path(s.risk.halt_file)
    if args.cmd == "halt":
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"halted {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        print(f"emergency halt engaged ({p}); no new orders will be placed")
    else:
        p.unlink(missing_ok=True)
        from raygold.risk.manager import RiskState

        st = RiskState.load(s.storage.state_path)
        st.halted, st.halt_reason = False, ""
        st.save(s.storage.state_path)
        print("halt cleared")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="raygold", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/raygold.yaml")
    ap.add_argument("--log-level", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def split_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--train-start", default="2022-01-01")
        p.add_argument("--val-start", default="2024-07-01")
        p.add_argument("--test-start", default="2025-07-01")
        p.add_argument("--test-end", default="2026-10-23")

    p = sub.add_parser("download")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=dt.date.today().isoformat())
    p.add_argument("--cache", default="data/raw/dukascopy")
    p.add_argument("--out", default=DEFAULT_DATA)
    p.add_argument("--workers", type=int, default=16)
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("research")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default="reports")
    split_args(p)
    p.set_defaults(fn=cmd_research)

    p = sub.add_parser("evaluate-test")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--strategy", default="reports/selected_strategy.json")
    p.add_argument("--out", default="reports")
    p.add_argument("--i-know-the-test-set-is-burned", action="store_true")
    split_args(p)
    p.set_defaults(fn=cmd_evaluate_test)

    p = sub.add_parser("backtest")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--strategy", default="reports/selected_strategy.json")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--trades-out", default=None)
    split_args(p)
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("feasibility")
    p.set_defaults(fn=cmd_feasibility)

    p = sub.add_parser("scan")
    p.add_argument("--mode", choices=["READ_ONLY", "PAPER", "DEMO"], default=None)
    p.add_argument("--strategy", default=None)
    p.add_argument("--replay", default=None, help="replay a historical file instead of connecting to MT5")
    p.add_argument("--replay-start", default="2026-10-01")
    p.add_argument("--server-utc-offset", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=0)
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("export")
    p.add_argument("--out", default="reports/export")
    p.set_defaults(fn=cmd_export)

    for name in ("halt", "resume"):
        p = sub.add_parser(name)
        p.set_defaults(fn=cmd_halt)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        s = load_settings(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    setup_logging(args.log_level or s.log_level, s.log_json)
    try:
        return int(args.fn(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
