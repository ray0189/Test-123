"""Generate the Markdown research report from the JSON/CSV outputs (no hand-typed numbers)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


def _f(x, nd: int = 3, pct: bool = False) -> str:  # type: ignore[no-untyped-def]
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-" if x is None or math.isnan(x) else ("inf" if x > 0 else "-inf")
    if pct:
        return f"{100 * x:.1f}%"
    if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer() and abs(x) > 50):
        return f"{int(x)}"
    return f"{x:.{nd}f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def stats_row(name: str, s: dict) -> dict:
    if not s or not s.get("trades"):
        return {"period": name, "trades": 0}
    return {
        "period": name, "trades": s["trades"], "wins/losses/scratch": f"{s['wins']}/{s['losses']}/{s['scratches']}",
        "win rate (95% CI)": f"{_f(s['win_rate'], pct=True)} ({_f(s['win_rate_lo'], pct=True)}-{_f(s['win_rate_hi'], pct=True)})",
        "expectancy R (95% CI, day-clustered)": f"{s['expectancy_r']:+.3f} ({s['expectancy_r_lo']:+.3f} to {s['expectancy_r_hi']:+.3f})",
        "t (clustered)": _f(s["t_stat"], 2), "profit factor": _f(s["profit_factor"], 2),
        "avg win / loss R": f"{s['avg_win_r']:+.2f} / {s['avg_loss_r']:+.2f}",
        "breakeven WR": _f(s["breakeven_win_rate"], pct=True),
        "total R": f"{s['total_r']:+.1f}", "max DD R": _f(s["max_dd_r"], 1), "trades/day": _f(s["trades_per_day"], 2),
    }


def build_report(reports_dir: str | Path) -> str:
    d = Path(reports_dir)
    summ = json.loads((d / "research_summary.json").read_text())
    test = json.loads((d / "test_evaluation.json").read_text()) if (d / "test_evaluation.json").exists() else None
    L: list[str] = []
    L.append("# RayGold research report (generated)\n")
    L.append(f"Data: Dukascopy XAUUSD M1 bid+ask, {summ['data']['start']} .. {summ['data']['end']} used for research "
             f"({summ['data']['bars']:,} bars, file hash `{summ['data']['hash']}`).\n")
    sp = summ["splits"]
    L.append(f"Splits: train `{sp['train_start']}`..`{sp['val_start']}`, validation `{sp['val_start']}`..`{sp['test_start']}`, "
             f"test `{sp['test_start']}`..`{sp['test_end']}` (test never loaded during research).\n")
    c = summ["costs"]
    L.append(f"Costs: real bid/ask spread per bar, commission {c['commission']:.2f} USD/oz round turn "
             f"(= {c['commission'] * 100:.0f} USD/lot), slippage {c['slippage']:.2f} USD/oz per market fill, "
             "stop-first on ambiguous bars, no position held over gaps > 3h.\n")
    tfs = summ.get("timeframes", ["M1"])
    L.append(f"Configurations evaluated: **{summ['configs_evaluated']:,}** "
             f"({len(tfs)} signal timeframes {'/'.join(tfs)} x {summ['variants']} entry variants x "
             f"{summ['exit_specs']} exit specs x {summ['filters']} filter sets; execution always on M1 bid/ask).\n")

    L.append("## 1. Pattern-family comparison (train-selected best config per family, then its validation result)\n")
    fam = pd.read_csv(d / "family_ranking.csv")
    show = fam[["family", "tf", "configs", "frac_pos_train", "frac_pos_both", "median_train_exp_r", "median_val_exp_r",
                "best_train_n", "best_train_wr", "best_train_exp_r", "best_val_n", "best_val_wr", "best_val_exp_r"]].copy()
    for col in ("frac_pos_train", "frac_pos_both", "best_train_wr", "best_val_wr"):
        show[col] = show[col].map(lambda x: _f(x, pct=True))
    for col in ("median_train_exp_r", "median_val_exp_r", "best_train_exp_r", "best_val_exp_r"):
        show[col] = show[col].map(lambda x: f"{x:+.3f}" if x == x else "-")
    for col in ("best_train_n", "best_val_n", "configs"):
        show[col] = show[col].map(lambda x: f"{int(x)}" if x == x else "-")
    L.append(_md(show) + "\n")
    L.append("`frac_pos_*` = share of the family's configs (with >= minimum trades) whose expectancy after costs is "
             "positive on train / on both train and validation. The `random_baseline` family enters at random bars "
             "with random direction and is the empirical null.\n")

    nl = summ["null"]
    L.append("## 2. Multiple-testing check\n")
    L.append(f"- Configs with enough train trades: {nl['configs_tested']:,}; Bonferroni 5% threshold t = {nl['bonferroni_t_5pct']:.2f}.")
    L.append(f"- Random-entry baseline: max train t = {_f(nl['random_max_train_t'], 2)}, 99th pct = {_f(nl['random_p99_train_t'], 2)}, "
             f"max train win rate = {_f(nl['random_max_train_win_rate'], pct=True)}.")
    L.append(f"- Pattern configs: max train t = {_f(nl['real_max_train_t'], 2)}; positive on train: {_f(nl['real_frac_positive_train'], pct=True)}; "
             f"positive on validation: {_f(nl['real_frac_positive_val'], pct=True)}.\n")

    L.append("## 3. Win rate versus expectancy (validation)\n")
    fr = pd.read_csv(d / "winrate_frontier.csv")
    fr["best_val_exp_r"] = fr["best_val_exp_r"].map(lambda x: f"{x:+.3f}")
    fr["median_val_exp_r"] = fr["median_val_exp_r"].map(lambda x: f"{x:+.3f}")
    L.append(_md(fr) + "\n")
    L.append("Configs with >= 100 trades in both periods. `pos_train_and_val` counts configs whose expectancy is positive "
             "after costs in *both* periods.\n")
    hw = pd.read_csv(d / "high_winrate_positive.csv")
    if len(hw):
        L.append("Highest validation win rates among configs positive on both train and validation:\n")
        t = hw.head(10)[["variant", "exit", "session", "min_htf", "min_score", "train_n", "train_win_rate", "train_exp_r",
                         "val_n", "val_win_rate", "val_exp_r"]].copy()
        for col in ("train_win_rate", "val_win_rate"):
            t[col] = t[col].map(lambda x: _f(x, pct=True))
        for col in ("train_exp_r", "val_exp_r"):
            t[col] = t[col].map(lambda x: f"{x:+.3f}")
        L.append(_md(t) + "\n")

    L.append("## 4. Walk-forward (12-month in-sample re-selection, 3-month out-of-sample, train+validation span)\n")
    wf = summ["walk_forward"]
    L.append("```\n" + json.dumps(wf, indent=1) + "\n```\n")
    wfd = pd.read_csv(d / "walk_forward.csv")
    keep = [c for c in ("oos_start", "oos_end", "eligible", "variant", "exit", "is_t", "is_exp_r", "oos_n", "oos_wins", "oos_total_r") if c in wfd]
    L.append(_md(wfd[keep].round(3).fillna("-")) + "\n")

    L.append("## 5. Selected candidate\n")
    cand = summ["candidate"]
    L.append(f"`{cand['key']}`\n")
    L.append(f"Passed every pre-registered gate: **{cand['validated']}** (neighbour stability {_f(cand['neighbour_frac'], pct=True)}).\n")
    rows = [stats_row("train", summ["train"]), stats_row("validation", summ["validation"])]
    if test:
        rows.append(stats_row("TEST (one-time)" + (" [REUSED]" if test.get("reused_test_period") else ""), test["test"]))
    L.append(_md(pd.DataFrame(rows).fillna("")) + "\n")

    L.append("### Adverse-condition stress tests\n")
    srows = []
    for name, st in summ["validation_stress"].items():
        srows.append({"scenario": name, "period": "validation", **{k: v for k, v in stats_row("", st).items() if k != "period"}})
    if test:
        for name, st in test["test_stress"].items():
            srows.append({"scenario": name, "period": "test", **{k: v for k, v in stats_row("", st).items() if k != "period"}})
    L.append(_md(pd.DataFrame(srows).fillna("")) + "\n")

    L.append("### GBP 100 account feasibility (risk GBP 1/trade, max 25 positions, aggregate risk GBP 5, daily limit GBP 3, 15% DD breaker)\n")
    arows = []
    for period, acct in (("validation", summ["validation_account_gbp100"]), ("test", test["test_account_gbp100"] if test else {})):
        for lev, a in acct.items():
            arows.append({"period": period, "leverage": lev, "signals": a["signals"], "taken": a["taken"],
                          "final equity": a["final_equity"], "max DD %": a["max_drawdown_pct"],
                          "win rate": _f(a["win_rate"], pct=True), "top reject reasons": "; ".join(
                              f"{k}: {v}" for k, v in sorted(a["reject_reasons"].items(), key=lambda kv: -kv[1])[:3])})
    L.append(_md(pd.DataFrame(arows)) + "\n")
    if "score_calibration_validation" in summ:
        L.append("### Confidence-score calibration (validation trades of the candidate)\n")
        L.append(_md(pd.DataFrame(summ["score_calibration_validation"]).round(3)) + "\n")
    if test and "score_calibration_test" in test:
        L.append("### Confidence-score calibration (test trades)\n")
        L.append(_md(pd.DataFrame(test["score_calibration_test"]).round(3)) + "\n")
    return "\n".join(L)


def write_report(reports_dir: str | Path) -> Path:
    p = Path(reports_dir) / "RESEARCH_REPORT.md"
    p.write_text(build_report(reports_dir))
    return p
