"""Audit the bar-level 'stop first' assumption with real Dukascopy ticks.

For a strategy's trades in a (non-test) period, every stop-out on a bar that
also touched the target is re-resolved on the tick stream.  Reports how many
flip to target-first and the expectancy with tick-resolved outcomes.

    python scripts/tick_audit.py --config-row <row id in reports/all_configs.parquet> --start 2024-07-01 --end 2025-07-01
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path

import numpy as np
import pandas as pd

from raygold.backtest.tick_replay import load_tick_hours, resolve_trade
from raygold.config import load_settings
from raygold.data.loader import load_parquet, slice_period
from raygold.features import build_features
from raygold.research.grid import default_exits, default_filters, research_variants
from raygold.research.run import costs_from_settings, spec_from_row
from raygold.research.stats import summarise
from raygold.research.strategy import run_strategy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-row", type=int, required=True)
    ap.add_argument("--start", default="2024-07-01")
    ap.add_argument("--end", default="2025-07-01")
    ap.add_argument("--data", default="data/XAUUSD_M1_bidask.parquet")
    ap.add_argument("--cache", default="data/raw/dukascopy_ticks")
    ap.add_argument("--out", default="reports/tick_audit")
    ap.add_argument("--max-trades", type=int, default=400)
    args = ap.parse_args()
    if pd.Timestamp(args.end) > pd.Timestamp("2025-07-01"):
        raise SystemExit("refusing to audit inside the test period")
    s = load_settings("config/raygold.yaml")
    m = pd.read_parquet("reports/all_configs.parquet")
    row = m.iloc[args.config_row]
    grids = {tf: (research_variants(3, tf), default_exits(tf), default_filters()) for tf in ("M1", "M5", "M15")}
    spec = spec_from_row(row, grids)
    df = slice_period(load_parquet(args.data), "2022-01-01", args.end)
    f = build_features(df)
    costs = costs_from_settings(s)
    tr = run_strategy(f, spec, costs, args.start, args.end)
    tp_dist = (tr["tp"] - tr["entry"]).abs()
    amb = tr[(tr["reason"] == "stop") & tp_dist.notna() & (tr["mfe_r"] * tr["risk"] >= tp_dist - 1e-9)].head(args.max_trades)
    cache = Path(args.cache)
    hours = set()
    for _, t in amb.iterrows():
        h = pd.Timestamp(t["entry_time"]).floor("h")
        while h <= pd.Timestamp(t["exit_time"]):
            hours.add(h)
            h += pd.Timedelta(hours=1)
    with cf.ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda h: load_tick_hours("XAUUSD", h, h + pd.Timedelta(hours=1), cache), sorted(hours)))
    resolved = []
    for i, t in amb.iterrows():
        start = pd.Timestamp(t["entry_time"])
        end = pd.Timestamp(t["exit_time"]) + pd.Timedelta(minutes=1)
        ticks = load_tick_hours("XAUUSD", start, end, cache)
        when, px, why = resolve_trade(ticks, start, end, int(t["dir"]), float(t["sl"]), float(t["tp"]))
        resolved.append((i, why))
    res = pd.DataFrame(resolved, columns=["i", "tick_reason"]).set_index("i")
    adj = tr["r"].copy()
    flips = res.index[res["tick_reason"] == "target"]
    # a flipped trade earns the target instead of the stop (commission still charged)
    adj.loc[flips] = ((tr.loc[flips, "tp"] - tr.loc[flips, "entry"]) * tr.loc[flips, "dir"] - costs.commission) / tr.loc[flips, "risk"]
    base = summarise(tr["r"].to_numpy(), tr["day"].to_numpy()).as_dict()
    tick = summarise(adj.to_numpy(), tr["day"].to_numpy()).as_dict()
    out = {"strategy": spec.key, "period": [args.start, args.end], "trades": int(len(tr)),
           "ambiguous_stop_bars": int(len(amb)), "tick_resolved": res["tick_reason"].value_counts().to_dict(),
           "bar_level": {k: base[k] for k in ("win_rate", "expectancy_r", "expectancy_r_lo", "expectancy_r_hi", "profit_factor")},
           "tick_adjusted": {k: tick[k] for k in ("win_rate", "expectancy_r", "expectancy_r_lo", "expectancy_r_hi", "profit_factor")}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"tick_audit_row{args.config_row}.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
