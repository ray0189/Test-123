"""End-to-end research run on train + validation data only.

The data frame is cut at ``splits.test_start`` *before* any feature is
computed, so nothing in this module can observe the test period.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from raygold.backtest.account import replay
from raygold.backtest.simulator import Costs, ExitSpec
from raygold.config import Settings
from raygold.data.loader import slice_period
from raygold.features import build_features
from raygold.research import protocol as P
from raygold.research.grid import FilterSpec, default_exits, default_filters, research_variants
from raygold.research.engine import run_sweep
from raygold.research.stats import summarise
from raygold.research.strategy import StrategySpec, run_strategy
from raygold.risk.sizing import SymbolSpec
from raygold.signals.events import Variant
from raygold.signals.scoring import calibration_table
from raygold.storage import Store

log = logging.getLogger(__name__)


def costs_from_settings(s: Settings) -> Costs:
    b = s.broker
    return Costs(commission=b.commission_usd_per_lot_round_turn / b.contract_size, slippage=b.slippage_price)


def symbol_from_settings(s: Settings) -> SymbolSpec:
    b = s.broker
    return SymbolSpec(b.symbol, b.contract_size, b.tick_size, b.tick_value_usd, b.volume_min, b.volume_step,
                      b.volume_max, b.stops_level_points, b.point)


def spec_from_row(row: pd.Series, variants: list[Variant], exits: list, filters: list[FilterSpec]) -> StrategySpec:
    v = next(x for x in variants if x.key == row["variant"])
    return StrategySpec(v, exits[int(row["exit_id"])][0], filters[int(row["filter_id"])])


def stress_costs(base: Costs) -> dict[str, tuple[Costs, int]]:
    """Adverse execution scenarios: (costs, extra entry-delay bars)."""
    return {
        "baseline": (base, 0),
        "slippage_x3": (Costs(base.commission, base.slippage * 3, base.max_gap_minutes), 0),
        "commission_x2": (Costs(base.commission * 2, base.slippage, base.max_gap_minutes), 0),
        "latency_1bar": (base, 1),
        "all_adverse": (Costs(base.commission * 2, base.slippage * 3, base.max_gap_minutes), 1),
    }


def detailed_stats(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return {"trades": 0}
    return summarise(trades["r"].to_numpy(), trades["day"].to_numpy()).as_dict()


def run_research(df: pd.DataFrame, settings: Settings, splits: P.Splits, rules: P.SelectionRules, out_dir: str | Path,
                 store: Store, data_hash: str, n_random: int = 3, grid: tuple | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    research_df = slice_period(df, splits.train_start, splits.test_start)  # TEST DATA NEVER ENTERS
    assert research_df["time"].max() < pd.Timestamp(splits.test_start, tz="UTC")
    log.info("research data %s .. %s (%d bars)", research_df["time"].iloc[0], research_df["time"].iloc[-1], len(research_df))
    f = build_features(research_df)
    costs = costs_from_settings(settings)
    variants, exits, filters = grid if grid else (research_variants(n_random), default_exits(), default_filters())
    res = run_sweep(f, variants, exits, filters, costs)
    log.info("sweep: %d configurations in %.0fs", len(res.configs), res.meta["seconds"])
    m = P.period_metrics(res, splits)
    m.to_parquet(out / "all_configs.parquet", index=False)

    null = P.null_distribution(m, rules)
    chosen, stage_a, stage_b = P.select_candidate(m, rules)
    stage_a.to_csv(out / "stage_a_train_top.csv", index=False)
    stage_b.to_csv(out / "stage_b_validation.csv", index=False)
    fam = P.family_ranking(m, rules)
    fam.to_csv(out / "family_ranking.csv", index=False)
    frontier = P.winrate_frontier(m)
    frontier.to_csv(out / "winrate_frontier.csv", index=False)
    wf = P.walk_forward(res, splits, rules)
    wf.to_csv(out / "walk_forward.csv", index=False)
    wf_sum = P.walk_forward_summary(wf)

    # Highest validated win-rate configs that are positive on both periods (for the 80% question).
    hw = m[(m["family"] != "random_baseline") & (m["train_n"] >= rules.min_train_trades) & (m["val_n"] >= rules.min_val_trades)]
    hw = hw[(hw["train_exp_r"] > 0) & (hw["val_exp_r"] > 0)].sort_values("val_win_rate", ascending=False).head(25)
    hw.to_csv(out / "high_winrate_positive.csv", index=False)
    rnd = m[(m["family"] == "random_baseline") & (m["train_n"] >= rules.min_train_trades)]
    rnd_hw = rnd.sort_values("train_win_rate", ascending=False).head(10)
    rnd_hw.to_csv(out / "random_baseline_high_winrate.csv", index=False)

    summary: dict = {
        "data": {"bars": int(len(research_df)), "start": str(research_df["time"].iloc[0]),
                 "end": str(research_df["time"].iloc[-1]), "hash": data_hash},
        "splits": splits.as_dict(), "rules": rules.as_dict(), "costs": asdict(costs),
        "configs_evaluated": int(len(m)), "variants": len(variants), "exit_specs": len(exits), "filters": len(filters),
        "null": null, "walk_forward": wf_sum, "stage_a_count": int(len(stage_a)), "stage_b_count": int(len(stage_b)),
    }

    candidate_row = chosen
    fallback = False
    if candidate_row is None:
        # No config passed every gate.  For transparency we still analyse the
        # best Stage-A config by validation t (or by train t if Stage A is empty)
        # and label it clearly as NOT a validated candidate.
        fallback = True
        pool = stage_a if len(stage_a) else m[(m["family"] != "random_baseline") & (m["train_n"] >= rules.min_train_trades)]
        key = "val_t" if len(stage_a) else "train_t"
        candidate_row = pool.sort_values(key, ascending=False).iloc[0]
    spec = spec_from_row(candidate_row, variants, exits, filters)
    summary["candidate"] = {"key": spec.key, "validated": not fallback,
                            "neighbour_frac": float(candidate_row.get("neighbour_frac", float("nan")))}

    # Detailed finalist analysis on train and validation (same code path as test/live).
    tr = run_strategy(f, spec, costs, splits.train_start, splits.val_start)
    va = run_strategy(f, spec, costs, splits.val_start, splits.test_start)
    tr.to_csv(out / "candidate_trades_train.csv", index=False)
    va.to_csv(out / "candidate_trades_validation.csv", index=False)
    summary["train"] = detailed_stats(tr)
    summary["validation"] = detailed_stats(va)
    if len(va):
        cal = calibration_table(va["score"].to_numpy(), (va["r"] > 0).to_numpy())
        cal.to_csv(out / "score_calibration_validation.csv", index=False)
        summary["score_calibration_validation"] = cal.astype({"bucket": str}).to_dict(orient="records")
    stress = {}
    for name, (c, delay) in stress_costs(costs).items():
        t = run_strategy(f, spec, c, splits.val_start, splits.test_start, entry_delay=delay)
        stress[name] = detailed_stats(t)
    summary["validation_stress"] = stress
    # Feasibility on a GBP 100 account at several leverage levels (validation period).
    sym = symbol_from_settings(settings)
    va_all = run_strategy(f, spec, costs, splits.val_start, splits.test_start, sequential=False)
    acct = {}
    for lev in (20.0, 100.0, 500.0):
        r = replay(va_all, settings.risk, sym, lev, settings.broker.gbpusd, settings.broker.commission_usd_per_lot_round_turn,
                   settings.broker.slippage_price)
        acct[f"1:{int(lev)}"] = r.summary
    summary["validation_account_gbp100"] = acct
    summary["elapsed_seconds"] = round(time.time() - t0, 1)

    spec.save(out / "selected_strategy.json", {"validated": not fallback, "splits": splits.as_dict(),
                                               "train": summary["train"], "validation": summary["validation"],
                                               "selection_row": {k: (v.item() if hasattr(v, "item") else v)
                                                                 for k, v in candidate_row.to_dict().items()}})
    (out / "research_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    exp_id = store.log_experiment("research_sweep", "full_sweep", {"splits": splits.as_dict(), "rules": rules.as_dict(),
                                                                   "costs": asdict(costs)}, summary, data_hash)
    summary["experiment_id"] = exp_id
    return summary
