"""Turn pattern detectors into tradable *setup events*.

A setup event is ``(idx, dir, ref)``: the signal is observable at the close
of M1 bar ``idx``; the trade would be entered at the open of ``idx + 1``;
``ref`` is the structural invalidation level used by structure stops.

Each strategy *family* has a small set of parameter *variants*.  The
research optimiser treats ``(family, variant)`` as the entry dimension of
the configuration space.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from raygold.features import Features, liquidity_targets
from raygold.patterns import smc, technical


@dataclass(frozen=True)
class Variant:
    family: str
    params: tuple[tuple[str, float], ...]

    @property
    def key(self) -> str:
        p = ",".join(f"{k}={v:g}" for k, v in self.params)
        return f"{self.family}[{p}]"

    def get(self, name: str) -> float:
        return dict(self.params)[name]


def _v(family: str, **params: float) -> Variant:
    return Variant(family, tuple(sorted(params.items())))


def default_variants() -> list[Variant]:
    vs: list[Variant] = []
    for k in (3, 5, 10):
        vs.append(_v("sweep_reversal", k=k))
        vs.append(_v("bos_continuation", k=k))
        vs.append(_v("choch_reversal", k=k))
    for g in (0.15, 0.3, 0.6):
        vs.append(_v("fvg_retest", min_gap=g))
    for k in (5, 10):
        vs.append(_v("ob_retest", k=k))
    for n in (30, 60, 240):
        vs.append(_v("breakout_retest", n=n))
    vs.append(_v("ema_pullback"))
    for k in (3, 5):
        vs.append(_v("rsi_divergence", k=k))
    vs.append(_v("sr_rejection", tol=0.25))
    for k in (3, 5):
        vs.append(_v("ict_model", k=k))
    vs.append(_v("eq_sweep", tol=0.1))
    vs.append(_v("random_baseline", rate=0.004))
    return vs


def _ms_cache(f: Features, k: int) -> tuple[np.ndarray, ...]:
    key = f"_ms{k}"
    if key not in f.arr:
        bos, choch, ref, state, brk = smc.market_structure(f.h, f.l, f.c, f[f"piv_h{k}"], f[f"piv_l{k}"])
        f.arr[key] = np.vstack([bos, choch, ref, state, brk])
    m = f.arr[key]
    return m[0].astype(np.int8), m[1].astype(np.int8), m[2], m[3].astype(np.int8), m[4]


def _bars_since(sig: np.ndarray, direction: int) -> np.ndarray:
    """Bars since the last bar where ``sig == direction`` (inclusive of current), inf if never."""
    n = len(sig)
    idx = np.where(sig == direction, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, np.arange(n) - last, np.inf).astype(float)
    return out


def ensure_context(f: Features) -> None:
    """Pattern-derived context arrays shared by the scoring engine."""
    if "ctx_ready" in f.arr:
        return
    sw, sw_ref, _ = smc.liquidity_sweep(f.h, f.l, f.c, f["atr"], f["piv_h5"], f["piv_l5"], 1.5, 600)
    bos, choch, _, state, _ = _ms_cache(f, 5)
    _, fvg_rt, _ = smc.fair_value_gaps(f.h, f.l, f.c, f.o, f["atr"], 0.15, 120, 8)
    eqs, _, eqh, eql = smc.equal_levels(f["piv_h5"], f["piv_l5"], f["atr"], 0.1, f.h, f.l, f.c, 1440)
    f.arr["eqh"], f.arr["eql"] = eqh, eql
    mss = np.where(choch != 0, choch, bos)
    for d, tag in ((1, "up"), (-1, "dn")):
        f.arr[f"since_sweep_{tag}"] = np.minimum(_bars_since(sw, d), _bars_since(eqs, d))
        f.arr[f"since_mss_{tag}"] = _bars_since(mss, d)
        f.arr[f"since_choch_{tag}"] = _bars_since(choch, d)
        f.arr[f"since_fvg_{tag}"] = _bars_since(fvg_rt, d)
    f.arr["ms_state"] = state
    f.arr["rej"] = technical.rejection_candles(f.o, f.h, f.l, f.c, f["atr"])
    f.arr["ctx_ready"] = np.ones(1)


def detect(f: Features, v: Variant, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Return per-bar ``(sig, ref)`` for one variant."""
    fam = v.family
    atr = f["atr"]
    if fam == "sweep_reversal":
        k = int(v.get("k"))
        sig, ref, _ = smc.liquidity_sweep(f.h, f.l, f.c, atr, f[f"piv_h{k}"], f[f"piv_l{k}"], 1.5, 60 * k)
    elif fam in ("bos_continuation", "choch_reversal"):
        k = int(v.get("k"))
        bos, choch, ref, _, _ = _ms_cache(f, k)
        sig = bos if fam == "bos_continuation" else choch
    elif fam == "fvg_retest":
        _, sig, ref = smc.fair_value_gaps(f.h, f.l, f.c, f.o, atr, v.get("min_gap"), 120, 8)
    elif fam == "ob_retest":
        k = int(v.get("k"))
        bos, choch, _, _, _ = _ms_cache(f, k)
        brk = np.where(choch != 0, choch, bos).astype(np.int8)
        sig, ref = smc.order_blocks(f.h, f.l, f.o, f.c, brk, 10, 120)
    elif fam == "breakout_retest":
        sig, ref = technical.breakout_retest(f.h, f.l, f.c, atr, int(v.get("n")), 30, 0.2)
    elif fam == "ema_pullback":
        sig, ref = technical.ema_pullback(f.o, f.l, f.h, f.c, f["ema20"], f["ema50"], f["ema200"])
    elif fam == "rsi_divergence":
        k = int(v.get("k"))
        sig, ref = technical.rsi_divergence(f.h, f.l, f["rsi"], f[f"piv_h{k}"], f[f"piv_l{k}"], k, 60, 3.0, 35.0, 65.0)
    elif fam == "sr_rejection":
        sig, ref = technical.sr_rejection(f["rej"], f.l, f.h, f.c, atr, f["sup_levels"], f["res_levels"], v.get("tol"))
    elif fam == "ict_model":
        # Sweep, then CHoCH in the same direction within 30 bars, then the
        # first FVG retest in that direction within 60 bars of the CHoCH.
        k = int(v.get("k"))
        sw, sw_ref, _ = smc.liquidity_sweep(f.h, f.l, f.c, atr, f[f"piv_h{k}"], f[f"piv_l{k}"], 1.5, 60 * k)
        _, choch, _, _, _ = _ms_cache(f, k)
        _, fvg_rt, fvg_ref = smc.fair_value_gaps(f.h, f.l, f.c, f.o, atr, 0.1, 60, 8)
        sig = np.zeros(len(f), np.int8)
        ref = np.full(len(f), np.nan)
        for d in (1, -1):
            s_sweep = _bars_since(sw, d)
            s_choch = _bars_since(choch, d)
            # sweep must precede the choch: sweep age > choch age, gap <= 30
            ok = (fvg_rt == d) & (s_choch <= 60) & (s_sweep > s_choch) & (s_sweep - s_choch <= 30)
            sig[ok] = d
            # structural stop beyond the sweep extreme
            sweep_ext = pd.Series(np.where(sw == d, sw_ref, np.nan)).ffill().to_numpy()
            ref[ok] = sweep_ext[ok]
    elif fam == "eq_sweep":
        sig, ref, _, _ = smc.equal_levels(f["piv_h5"], f["piv_l5"], atr, v.get("tol"), f.h, f.l, f.c, 1440)
    elif fam == "random_baseline":
        rng = np.random.default_rng(int(dict(v.params).get("seed", seed)))
        hit = rng.random(len(f)) < v.get("rate")
        sig = np.where(hit, np.where(rng.random(len(f)) < 0.5, 1, -1), 0).astype(np.int8)
        ref = np.where(sig > 0, f.l, np.where(sig < 0, f.h, np.nan))
    else:
        raise ValueError(f"unknown family {fam}")
    return np.asarray(sig, np.int8), np.asarray(ref, float)


EVENT_COLUMNS = [
    "idx", "dir", "ref", "atr", "spread", "session", "htf_m5", "htf_m15", "htf_h1", "htf_align",
    "trend_m1", "since_sweep", "since_mss", "since_fvg", "vol_ratio", "atr_pct", "pd_pos", "rsi", "liq_target",
]


def build_events(f: Features, v: Variant, start_idx: int = 300, end_idx: int | None = None) -> pd.DataFrame:
    """Events for one variant with their context snapshot at the signal bar."""
    ensure_context(f)
    sig, ref = detect(f, v)
    n = len(f)
    # By default the last bar is excluded (backtests need a next bar to enter on);
    # the live scanner passes ``end_idx=n`` to include the just-completed bar.
    end_idx = n - 1 if end_idx is None else min(end_idx, n)
    idx = np.flatnonzero(sig != 0)
    idx = idx[(idx >= start_idx) & (idx < end_idx)]
    a = f.arr
    # Only bars with a valid ATR and a real next bar.
    idx = idx[np.isfinite(a["atr"][idx]) & np.isfinite(ref[idx])]
    d = sig[idx].astype(np.int8)
    ev = pd.DataFrame({"idx": idx, "dir": d, "ref": ref[idx]})
    ev["atr"] = a["atr"][idx]
    ev["spread"] = a["spread"][idx]
    ev["session"] = a["session"][idx]
    for tf in ("m5", "m15", "h1"):
        ev[f"htf_{tf}"] = a[f"{tf}_trend"][idx] * d
    ev["htf_align"] = ev[["htf_m5", "htf_m15", "htf_h1"]].fillna(0).sum(axis=1)
    ev["trend_m1"] = a["trend_m1"][idx] * d
    up = d > 0
    ev["since_sweep"] = np.where(up, a["since_sweep_up"][idx], a["since_sweep_dn"][idx])
    ev["since_mss"] = np.where(up, a["since_mss_up"][idx], a["since_mss_dn"][idx])
    ev["since_fvg"] = np.where(up, a["since_fvg_up"][idx], a["since_fvg_dn"][idx])
    ev["vol_ratio"] = a["vol_ratio"][idx]
    ev["atr_pct"] = a["atr_pct"][idx]
    pdp = a["pd_pos"][idx]
    ev["pd_pos"] = np.where(up, pdp, 1.0 - pdp)  # <0.5 = favourable side of range
    ev["rsi"] = a["rsi"][idx]
    price = np.where(up, f.ac[idx], f.c[idx])
    ev["liq_target"] = liquidity_targets(f, idx, d, price, 0.5 * a["atr"][idx])
    ev["family"] = v.family
    ev["variant"] = v.key
    return ev


def variant_grid(families: dict[str, dict[str, list[float]]]) -> list[Variant]:
    out = []
    for fam, grid in families.items():
        keys = sorted(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            out.append(Variant(fam, tuple(zip(keys, combo))))
    return out
