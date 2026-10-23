"""Pre-registered research protocol.

The rules below were fixed *before* looking at any validation or test
results and are applied mechanically:

1. **Splits** (chronological): train, validation, test.  The sweep only
   ever sees data before ``test_start``.
2. **Stage A (train only)**: a config is eligible if it has at least
   ``min_train_trades`` trades, positive expectancy, profit factor >=
   ``min_pf`` and t-statistic >= ``min_train_t``.  Rank by train t; keep the
   top ``top_k``.  Random-entry baseline configs are never eligible.
3. **Stage B (validation)**: of those, keep configs with >=
   ``min_val_trades`` trades, positive expectancy, PF >= ``min_pf`` and
   t >= ``min_val_t``; rank by validation t.
4. **Stability**: walk down that ranking and pick the first config whose
   one-step neighbours (filters, stop/target, pattern parameter) are
   positive on both train and validation at least ``min_neighbour_frac``
   of the time.
5. **Test**: the chosen candidate is evaluated once on the test period.
   The run is recorded and the period is marked *burned*.

If no config survives, the protocol reports that no edge was found.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from raygold.research.engine import S_N, S_SUM, SweepResult, window_metrics
from raygold.research.grid import HTF_MIN, SCORE_MIN, SESSION_SETS, VOL_FILTER


@dataclass(frozen=True)
class Splits:
    train_start: str = "2022-01-01"
    val_start: str = "2024-07-01"
    test_start: str = "2025-07-01"
    test_end: str = "2026-10-23"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SelectionRules:
    min_train_trades: int = 300
    min_val_trades: int = 100
    min_pf: float = 1.05
    min_train_t: float = 2.0
    min_val_t: float = 1.0
    top_k: int = 300
    min_neighbour_frac: float = 0.6

    def as_dict(self) -> dict:
        return asdict(self)


def month_mask(months: list[pd.Period], start: str, end: str) -> np.ndarray:
    s = pd.Period(start, freq="M")
    e = pd.Period(end, freq="M")
    return np.array([(m >= s) and (m < e) for m in months])


def period_metrics(res: SweepResult, splits: Splits) -> pd.DataFrame:
    tr = window_metrics(res.monthly, month_mask(res.months, splits.train_start, splits.val_start))
    va = window_metrics(res.monthly, month_mask(res.months, splits.val_start, splits.test_start))
    out = res.configs.copy()
    for col in tr.columns:
        out[f"train_{col}"] = tr[col].to_numpy()
        out[f"val_{col}"] = va[col].to_numpy()
    return out


def stage_a(m: pd.DataFrame, rules: SelectionRules) -> pd.DataFrame:
    ok = (
        (m["family"] != "random_baseline")
        & (m["train_n"] >= rules.min_train_trades)
        & (m["train_exp_r"] > 0)
        & (m["train_pf"] >= rules.min_pf)
        & (m["train_t"] >= rules.min_train_t)
    )
    return m[ok].sort_values("train_t", ascending=False).head(rules.top_k)


def stage_b(a: pd.DataFrame, rules: SelectionRules) -> pd.DataFrame:
    ok = (
        (a["val_n"] >= rules.min_val_trades)
        & (a["val_exp_r"] > 0)
        & (a["val_pf"] >= rules.min_pf)
        & (a["val_t"] >= rules.min_val_t)
    )
    return a[ok].sort_values("val_t", ascending=False)


def _variant_param(variant_key: str) -> tuple[str, float | None]:
    inside = variant_key[variant_key.index("[") + 1: -1]
    if not inside:
        return "", None
    k, v = inside.split(",")[0].split("=")
    return k, float(v)


def neighbours(m: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """One-step neighbours of a config in the discrete parameter space."""
    sel = []
    same_ve = (m["variant"] == row["variant"]) & (m["exit_id"] == row["exit_id"])
    dims = {"session": SESSION_SETS, "min_htf": HTF_MIN, "min_score": SCORE_MIN, "vol_filter": VOL_FILTER}
    for dim, vals in dims.items():
        vals = list(vals)
        i = vals.index(row[dim])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                cond = same_ve.copy()
                for other in dims:
                    cond &= (m[other] == (vals[j] if other == dim else row[other]))
                sel.append(m[cond])
    same_vf = (m["variant"] == row["variant"]) & (m["filter_id"] == row["filter_id"])
    if row["tp_idx"] >= 0:
        for dt in (-1, 1):
            sel.append(m[same_vf & (m["sl_idx"] == row["sl_idx"]) & (m["tp_idx"] == row["tp_idx"] + dt)])
    for ds in (-1, 1):
        sel.append(m[same_vf & (m["sl_idx"] == row["sl_idx"] + ds) & (m["tp_idx"] == row["tp_idx"]) & (m["mgmt"] == row["mgmt"])])
    pk, pv = _variant_param(row["variant"])
    if pv is not None:
        fam = m[(m["family"] == row["family"]) & (m["exit_id"] == row["exit_id"]) & (m["filter_id"] == row["filter_id"]) & (m["variant"] != row["variant"])]
        vals = sorted({_variant_param(v)[1] for v in fam["variant"].unique()} | {pv})
        i = vals.index(pv)
        adj = {vals[j] for j in (i - 1, i + 1) if 0 <= j < len(vals)}
        sel.append(fam[[(_variant_param(v)[1] in adj) for v in fam["variant"]]])
    return pd.concat(sel) if sel else m.iloc[0:0]


def stability(m: pd.DataFrame, row: pd.Series) -> tuple[float, int]:
    nb = neighbours(m, row)
    if len(nb) == 0:
        return 0.0, 0
    good = (nb["train_exp_r"] > 0) & (nb["val_exp_r"] > 0)
    return float(good.mean()), len(nb)


def select_candidate(m: pd.DataFrame, rules: SelectionRules) -> tuple[pd.Series | None, pd.DataFrame, pd.DataFrame]:
    a = stage_a(m, rules)
    b = stage_b(a, rules)
    rows = []
    chosen = None
    for _, row in b.head(50).iterrows():
        frac, cnt = stability(m, row)
        rows.append({**row.to_dict(), "neighbour_frac": frac, "neighbours": cnt})
        if chosen is None and frac >= rules.min_neighbour_frac and cnt >= 3:
            chosen = row.copy()
            chosen["neighbour_frac"] = frac
            chosen["neighbours"] = cnt
    return chosen, a, pd.DataFrame(rows)


def null_distribution(m: pd.DataFrame, rules: SelectionRules) -> dict:
    """What the selection would find with *random entries* (empirical null)."""
    r = m[(m["family"] == "random_baseline") & (m["train_n"] >= rules.min_train_trades)]
    real = m[(m["family"] != "random_baseline") & (m["train_n"] >= rules.min_train_trades)]
    n_tests = int((m["train_n"] >= rules.min_train_trades).sum())
    bonf_t = float(sps.norm.ppf(1 - 0.05 / max(n_tests, 1)))
    return {
        "random_configs": int(len(r)),
        "random_max_train_t": float(r["train_t"].max()) if len(r) else float("nan"),
        "random_p99_train_t": float(r["train_t"].quantile(0.99)) if len(r) else float("nan"),
        "random_median_train_exp_r": float(r["train_exp_r"].median()) if len(r) else float("nan"),
        "random_max_train_win_rate": float(r["train_win_rate"].max()) if len(r) else float("nan"),
        "real_configs": int(len(real)),
        "real_max_train_t": float(real["train_t"].max()) if len(real) else float("nan"),
        "real_frac_positive_train": float((real["train_exp_r"] > 0).mean()) if len(real) else float("nan"),
        "real_frac_positive_val": float((real["val_exp_r"] > 0).mean()) if len(real) else float("nan"),
        "configs_tested": n_tests,
        "bonferroni_t_5pct": bonf_t,
    }


def walk_forward(res: SweepResult, splits: Splits, rules: SelectionRules, is_months: int = 12, oos_months: int = 3) -> pd.DataFrame:
    """Rolling re-selection on train+validation; reports concatenated out-of-sample results.

    At each fold the Stage-A rule (trade threshold scaled to window length)
    picks the top config on the in-sample window; its next ``oos_months``
    are recorded.  This estimates the performance of the *selection
    procedure*, not of a single lucky config.
    """
    months = res.months
    start = pd.Period(splits.train_start, freq="M")
    stop = pd.Period(splits.test_start, freq="M")
    idx = [i for i, mth in enumerate(months) if start <= mth < stop]
    cfg = res.configs
    is_real = (cfg["family"] != "random_baseline").to_numpy()
    folds = []
    pos = 0
    while pos + is_months + oos_months <= len(idx):
        ism = np.zeros(len(months), bool)
        ism[idx[pos: pos + is_months]] = True
        oosm = np.zeros(len(months), bool)
        oosm[idx[pos + is_months: pos + is_months + oos_months]] = True
        ism_metrics = window_metrics(res.monthly, ism)
        min_n = rules.min_train_trades * is_months / 30.0
        ok = is_real & (ism_metrics["n"] >= min_n) & (ism_metrics["exp_r"] > 0) & (ism_metrics["pf"] >= rules.min_pf) \
            & (ism_metrics["t"] >= rules.min_train_t)
        ok = ok.to_numpy()
        fold = {"is_start": str(months[idx[pos]]), "oos_start": str(months[idx[pos + is_months]]),
                "oos_end": str(months[idx[pos + is_months + oos_months - 1]]), "eligible": int(ok.sum())}
        if ok.any():
            best = int(np.argmax(np.where(ok, ism_metrics["t"].to_numpy(), -np.inf)))
            oos = res.monthly[best][oosm]
            n = float(oos[:, S_N].sum())
            fold.update({"config_row": best, "variant": cfg.iloc[best]["variant"], "exit": cfg.iloc[best]["exit"],
                         "is_t": float(ism_metrics["t"].iloc[best]), "is_exp_r": float(ism_metrics["exp_r"].iloc[best]),
                         "oos_n": n, "oos_wins": float(oos[:, 1].sum()), "oos_total_r": float(oos[:, S_SUM].sum()),
                         "oos_sumsq": float(oos[:, 3].sum())})
        folds.append(fold)
        pos += oos_months
    return pd.DataFrame(folds)


def walk_forward_summary(wf: pd.DataFrame) -> dict:
    if "oos_n" not in wf or wf["oos_n"].fillna(0).sum() == 0:
        return {"folds": int(len(wf)), "oos_trades": 0}
    n = wf["oos_n"].fillna(0).sum()
    tot = wf["oos_total_r"].fillna(0).sum()
    sq = wf["oos_sumsq"].fillna(0).sum()
    mean = tot / n
    sd = np.sqrt(max(sq / n - mean**2, 0) * n / max(n - 1, 1))
    return {"folds": int(len(wf)), "folds_with_selection": int(wf["oos_n"].notna().sum()), "oos_trades": int(n),
            "oos_win_rate": float(wf["oos_wins"].fillna(0).sum() / n), "oos_exp_r": float(mean),
            "oos_t": float(mean / (sd / np.sqrt(n))) if sd > 0 else float("nan"), "oos_total_r": float(tot),
            "folds_positive": int((wf["oos_total_r"].fillna(0) > 0).sum())}


def family_ranking(m: pd.DataFrame, rules: SelectionRules) -> pd.DataFrame:
    """Honest comparison of pattern families: best-by-train config and the family-wide picture."""
    rows = []
    for fam, g in m.groupby("family"):
        big = g[g["train_n"] >= rules.min_train_trades]
        if len(big) == 0:
            rows.append({"family": fam, "configs": len(g)})
            continue
        best = big.sort_values("train_t", ascending=False).iloc[0]
        rows.append({
            "family": fam, "configs": len(g), "configs_with_min_trades": len(big),
            "frac_pos_train": float((big["train_exp_r"] > 0).mean()),
            "frac_pos_both": float(((big["train_exp_r"] > 0) & (big["val_exp_r"] > 0)).mean()),
            "median_train_exp_r": float(big["train_exp_r"].median()),
            "median_val_exp_r": float(big["val_exp_r"].median()),
            "best_cfg": f"{best['variant']} :: {best['exit']} :: s={best['session']},htf>={best['min_htf']},score>={best['min_score']},vol={best['vol_filter']}",
            "best_train_n": int(best["train_n"]), "best_train_wr": float(best["train_win_rate"]),
            "best_train_exp_r": float(best["train_exp_r"]), "best_train_t": float(best["train_t"]),
            "best_val_n": int(best["val_n"]), "best_val_wr": float(best["val_win_rate"]),
            "best_val_exp_r": float(best["val_exp_r"]), "best_val_t": float(best["val_t"]),
        })
    return pd.DataFrame(rows).sort_values("best_val_exp_r", ascending=False, na_position="last")


def winrate_frontier(m: pd.DataFrame, min_trades: int = 100) -> pd.DataFrame:
    """For each validation win-rate band: how many configs and how many also have positive expectancy."""
    g = m[(m["family"] != "random_baseline") & (m["train_n"] >= min_trades) & (m["val_n"] >= min_trades)].copy()
    bands = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    g["band"] = pd.cut(g["val_win_rate"], bands, right=False)
    out = g.groupby("band", observed=True).agg(
        configs=("val_n", "size"),
        pos_train_and_val=("val_exp_r", lambda s: int(((s > 0) & (g.loc[s.index, "train_exp_r"] > 0)).sum())),
        best_val_exp_r=("val_exp_r", "max"),
        median_val_exp_r=("val_exp_r", "median"),
    ).reset_index()
    out["band"] = out["band"].astype(str)
    return out
