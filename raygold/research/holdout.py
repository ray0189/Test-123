"""One-time evaluation of the selected candidate on the untouched test period.

The first evaluation of a test period is recorded in the SQLite
``holdout_registry`` and the period is thereafter *burned*.  A second
evaluation is refused unless ``allow_reuse=True``, in which case every
output is stamped as non-independent research data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from raygold.backtest.account import replay
from raygold.config import Settings
from raygold.data.loader import slice_period
from raygold.features import build_features
from raygold.research.protocol import Splits
from raygold.research.run import costs_from_settings, detailed_stats, stress_costs, symbol_from_settings
from raygold.research.strategy import StrategySpec, run_strategy
from raygold.signals.scoring import calibration_table
from raygold.storage import Store

log = logging.getLogger(__name__)


class HoldoutBurned(RuntimeError):
    pass


def period_key(symbol: str, splits: Splits) -> str:
    return f"{symbol}:{splits.test_start}:{splits.test_end}"


def evaluate_test(df: pd.DataFrame, settings: Settings, splits: Splits, strategy_file: str | Path, out_dir: str | Path,
                  store: Store, data_hash: str, allow_reuse: bool = False) -> dict:
    key = period_key(settings.broker.symbol, splits)
    status = store.holdout_status(key)
    if status and not allow_reuse:
        raise HoldoutBurned(
            f"test period {key} was already evaluated on {status['first_used_utc']} (experiment {status['experiment_id']}). "
            "It is now research data; reserve newer data for the next independent evaluation."
        )
    payload = json.loads(Path(strategy_file).read_text())
    spec = StrategySpec.from_dict(payload["strategy"])
    # Warm-up from earlier data is allowed (indicators are causal); trades are
    # counted only if they *enter* inside the test window.
    warm_start = (pd.Timestamp(splits.test_start) - pd.Timedelta(days=30)).date().isoformat()
    sub = slice_period(df, warm_start, splits.test_end)
    f = build_features(sub)
    costs = costs_from_settings(settings)
    trades = run_strategy(f, spec, costs, splits.test_start, splits.test_end)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "candidate_trades_test.csv", index=False)
    result: dict = {"period": key, "strategy": spec.key, "validated_candidate": payload.get("validated"),
                    "reused_test_period": bool(status), "test": detailed_stats(trades)}
    if len(trades):
        cal = calibration_table(trades["score"].to_numpy(), (trades["r"] > 0).to_numpy())
        result["score_calibration_test"] = cal.astype({"bucket": str}).to_dict(orient="records")
        result["test_by_reason"] = trades["reason"].value_counts().to_dict()
        result["test_by_month_r"] = trades.groupby(pd.to_datetime(trades["entry_time"]).dt.strftime("%Y-%m"))["r"].sum().round(2).to_dict()
    stress = {}
    for name, (c, delay) in stress_costs(costs).items():
        stress[name] = detailed_stats(run_strategy(f, spec, c, splits.test_start, splits.test_end, entry_delay=delay))
    result["test_stress"] = stress
    allt = run_strategy(f, spec, costs, splits.test_start, splits.test_end, sequential=False)
    sym = symbol_from_settings(settings)
    acct = {}
    for lev in (20.0, 100.0, 500.0):
        r = replay(allt, settings.risk, sym, lev, settings.broker.gbpusd, settings.broker.commission_usd_per_lot_round_turn,
                   settings.broker.slippage_price)
        acct[f"1:{int(lev)}"] = r.summary
        if lev == 100.0:
            r.trades.to_csv(out / "account_replay_test_1to100.csv", index=False)
            r.equity_curve.to_csv(out / "equity_curve_test_1to100.csv", index=False)
    result["test_account_gbp100"] = acct
    exp_id = store.log_experiment("holdout_test", spec.key, {"splits": splits.as_dict(), "reuse": bool(status)}, result,
                                  data_hash)
    store.burn_holdout(key, exp_id, spec.key)
    result["experiment_id"] = exp_id
    payload["test"] = result["test"]
    payload["test_reused"] = bool(status)
    Path(strategy_file).write_text(json.dumps(payload, indent=2, default=str))
    (out / "test_evaluation.json").write_text(json.dumps(result, indent=2, default=str))
    return result
