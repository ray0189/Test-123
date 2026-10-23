# RayGold — XAUUSD pattern research framework and MT5 scanner

RayGold detects ICT/Smart-Money and classical technical patterns on XAUUSD M1
(with M5/M15/H1 confirmation), scores them, plans stop-loss/take-profit levels,
and evaluates everything with a leakage-free, cost-aware research pipeline.
It ships with a MetaTrader 5 scanner and terminal dashboard.

> **Safety defaults.** The system starts in `READ_ONLY` mode and never sends an
> order unless you explicitly select `PAPER` (local simulation) or `DEMO` (real
> `order_send`, but *only* when the connected MT5 account reports itself as a
> demo account). `LIVE` is disabled: it requires a config change, an
> acknowledgement environment variable and a code-level flag that the CLI does
> not set. Nothing in this repository is financial advice.

**Research outcome in one paragraph:** see [Results](#results) and the generated
[`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md). Every number in
them comes from result files in `reports/`; nothing is typed by hand.

---

## Installation

Requires Python 3.11+.

```bash
git clone <this repo> && cd Test-123
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install MetaTrader5          # Windows only; needed for scan against a terminal
pytest                           # full suite (~6 min); `pytest -m "not slow"` for the fast subset
```

## Data

```bash
raygold download --start 2022-01-01 --end 2026-10-22     # Dukascopy M1 bid AND ask candles -> data/XAUUSD_M1_bidask.parquet
```

Both bid and ask series are stored, so the backtester charges the real historical
spread on every fill. Your own data works too (`--data path`):

| Input | Notes |
|---|---|
| Parquet from `raygold download` | true bid/ask |
| MT5 *Export Bars* CSV (`<DATE> <TIME> <OPEN> … <SPREAD>`) | ask = bid + spread×point; MT5 stores the bar's minimum spread, so this is optimistic. Pass the broker's UTC offset to `load_bar_csv(utc_offset_hours=…)`. |
| Generic CSV `time,open,high,low,close[,volume,spread]` | |
| Tick CSV `time,bid,ask` (or MT5 tick export) | aggregated to M1 bid/ask; also usable for exact first-touch replay (`raygold.backtest.tick_replay`) |

## Architecture

```
raygold/
  config.py            YAML + RAYGOLD_* env overrides, modes, safety validation
  data/                dukascopy downloader, CSV/tick loaders, completed-candle HTF alignment
  indicators.py        causal EMA / RSI / ATR / pivots (numba)
  patterns/smc.py      liquidity sweep, FVG, BOS/CHoCH, order block, equal highs/lows, premium/discount
  patterns/technical.py EMA trend, RSI divergence, breakout-retest, EMA pullback, S/R rejection, candles, volume
  patterns/sessions.py Asia / London / New York / overlap with DST
  features.py          all causal feature arrays + liquidity targets (PDH/PDL, Asia range, H1 swings, EQH/EQL)
  signals/events.py    strategy families -> setup events with context snapshot
  signals/scoring.py   0-100 confidence score (heuristic; calibration tables reported)
  signals/live.py      signal on the last completed bar, same code path as research
  backtest/simulator.py bid/ask M1 trade simulator (numba, parallel)
  backtest/account.py  GBP account replay: sizing, margin, positions, breakers
  backtest/tick_replay.py tick-level first-touch resolution
  research/            grid, sweep engine, protocol (selection, walk-forward, null), holdout lock, stats
  risk/sizing.py       lots from contract size / tick value / tick size / FX / volume step
  risk/manager.py      pre-trade checks and persistent circuit breakers
  mt5/                 connection, market data, execution (gated), position monitoring
  reporting/           rich dashboard, Markdown report generator
  storage.py           SQLite: signals, trades, experiments, holdout registry
  app.py, cli.py       scanner loop and CLI
```

## Pattern definitions

All detectors are deterministic, use only information available at the close of
the signal bar, and are unit-tested (`tests/test_stats_and_patterns.py`). The
exact rules are in each function's docstring; summary (bullish side, bearish is
the mirror; *A* = ATR(14); swings are fractal pivots of strength *k*, usable only
*k* bars after the pivot):

| Pattern | Definition |
|---|---|
| Liquidity sweep | low < latest unswept confirmed swing low *S*, close > *S*, depth ≤ 1.5A |
| Fair Value Gap | low[i] > high[i-2], gap ≥ g·A; signal on first retest (low ≤ top, close ≥ bottom) within 120 bars |
| BOS / CHoCH | close beyond the latest unbroken confirmed swing; CHoCH if prior structure was opposite |
| Order block | last opposite candle within 10 bars before a BOS/CHoCH; signal on first retest |
| Equal highs/lows | consecutive swings within 0.1A form a pool; signal on sweep and reclaim |
| Premium/discount | close position in the completed-H1 24-bar range |
| ICT model | sweep → CHoCH within 30 bars → FVG retest within 60 bars |
| Breakout-retest | close > N-bar high, then retest of that level within 30 bars |
| EMA pullback | EMA20>50>200, low ≤ EMA20 < close, bullish candle |
| RSI divergence | lower low at confirmed pivot with RSI higher by ≥ 3, prior RSI ≤ 35 |
| S/R rejection | pin bar or engulfing whose low is within 0.25A of a confirmed support |
| HTF trend | completed M5/M15/H1: EMA20 vs EMA50 and close vs EMA20 |

## Research protocol

* **Chronological splits** (fixed before any run): train 2022-01-01 → 2024-07-01,
  validation 2024-07-01 → 2025-07-01, test 2025-07-01 → 2026-10-23.
* **The research command never loads test data**: the frame is cut at the test
  start before any feature is computed (asserted in code and in
  `tests/test_integration.py`).
* **No lookahead**: `tests/test_no_lookahead.py` recomputes every feature, pattern
  and score after scrambling all future bars and requires identical values up to
  the cut; the test was mutation-checked (an injected one-bar leak is caught).
* **Execution realism**: entry at next bar open on the ask (longs) / bid
  (shorts) plus slippage; exits on the opposite side; stop-market with gap fills;
  limit targets never better than the limit; *stop assumed first* whenever a bar
  touches both; commission per round turn; break-even/trailing only from the next
  bar; positions flattened before gaps > 3 h (weekends).
* **Non-overlapping evidence**: each configuration holds at most one position at a
  time; confidence intervals resample whole trading days (clustered bootstrap).
* **Configuration space**: 27 entry variants (incl. 3 random-entry baselines)
  × 42 exit specs (structure/ATR stops × fixed-R, ATR, liquidity, net-positive,
  break-even, trailing, partial, time exits) × 96 filter sets (session, HTF
  alignment, min score, volatility) = **108,864** configurations.
* **Selection** (pre-registered, `raygold/research/protocol.py`): train gates
  (≥300 trades, PF ≥ 1.05, t ≥ 2) → top 300 → validation gates (≥100 trades,
  positive expectancy, PF ≥ 1.05, t ≥ 1) → neighbour stability ≥ 60 %.
* **Walk-forward**: 12-month in-sample re-selection, 3-month out-of-sample folds.
* **Null**: random-entry configurations go through the identical pipeline.
* **One-time test**: `raygold evaluate-test` records the test period in the
  SQLite `holdout_registry`; a second run is refused (or, with an explicit flag,
  stamped as non-independent).

## Commands

```bash
raygold feasibility                    # what a GBP 100 account can actually trade
raygold research                       # sweep + selection + walk-forward on train/validation (~10 min)
raygold evaluate-test                  # ONE-TIME untouched test evaluation of reports/selected_strategy.json
raygold report                         # regenerate reports/RESEARCH_REPORT.md
raygold backtest --start 2024-07-01 --end 2025-07-01   # any strategy file on non-test dates

# Scanner + dashboard
raygold scan                                            # READ_ONLY against the MT5 terminal (Windows)
raygold scan --replay data/XAUUSD_M1_bidask.parquet --replay-start 2026-10-01   # offline replay, read-only
raygold scan --mode PAPER --replay data/XAUUSD_M1_bidask.parquet                # paper trading on replay
raygold scan --mode PAPER                               # paper trading on live MT5 prices
raygold scan --mode DEMO --server-utc-offset 2          # real orders, refused unless the account is DEMO

raygold halt        # emergency stop (creates state/HALT); `raygold resume` clears it
raygold export      # signals/trades/experiments -> reports/export/*.csv|json
```

MT5 credentials come only from the environment: `MT5_LOGIN`, `MT5_PASSWORD`,
`MT5_SERVER`, optionally `MT5_TERMINAL_PATH`. Any config key can be overridden
with `RAYGOLD_<SECTION>__<KEY>` (e.g. `RAYGOLD_RISK__MAX_POSITIONS=3`).

Economic-news blackout: put high-impact events in `config/news_events.csv`
(`time_utc,currency,impact,title`); USD/XAU "High" events block new trades for
±30 minutes.

## Results

All numbers below come from `reports/` (generated by `raygold research`,
`raygold evaluate-test`, `scripts/tick_audit.py`; full tables in
[`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md)). Data: Dukascopy
XAUUSD M1 bid+ask, 2022-01-02 → 2026-10-22 (1,705,353 bars).

### Bottom line

* **No pattern configuration showed a statistically reliable, positive
  after-cost edge.** Of 326,592 configurations (M1/M5/M15 signal timeframes),
  none passed the pre-registered train gate (best train t = 1.86 vs. 2.0
  required; the multiple-testing threshold is t = 5.02). The best
  random-entry configuration reached t = 0.46.
* **An 80 % win rate with positive expectancy was not achieved.** Win rates of
  80–95 % are easy to produce (tight "net-positive" targets with wide stops):
  3,454 configurations reached ≥ 80 % on validation. None of them had positive
  expectancy on train; only 6 were positive on validation (best +0.064 R,
  t = 1.24, 102 trades) and those were negative on train. Of 19,280 configs with
  ≥ 60 % validation win rate, none was positive on both periods (best train
  expectancy −0.009 R). Random entries reach 87 % win rates with the same
  exits, so those win rates come from exit geometry, not from pattern skill.
* **The strategy is not operationally feasible on a £100 account with £1 risk.**
  A 0.01 lot of gold is 1 oz, so £1 of risk allows a stop of at most
  **$1.15/oz**. The all-in round-trip cost (median spread + $7/lot commission +
  slippage) was $0.50–0.85/oz, i.e. **43–74 % of the maximum risk on every
  trade before the market moves**. The final candidate needed a median stop
  of ~$6, so the account replay rejected **all** 515 of its validation and test
  signals (minimum lot exceeds the risk budget). Margin is a second hard limit:
  at FCA-retail 1:20 leverage £100 cannot margin even one 0.01 lot above
  ~$2,540 gold. At ~$4,000 gold, 1:100 margins at most 3 min-lot positions
  (1 while keeping a 300 % margin level) and 1:500 at most 15 (5); 25
  simultaneous £1-risk positions would also put 25 % of the account at risk
  at once (`raygold feasibility`).
* Your earlier result (30.5 % win rate over 1,587 trades with 2R targets) is
  consistent with this: its 95 % CI (28.3–32.8 %) lies below the 33.3 %
  break-even of a 2R target *before* costs.

### What the evidence shows

| Check | Result |
|---|---|
| Zero-cost diagnostic (M1) | Random entries ≈ 0 R (train +0.007, validation −0.036), as they should be. Trailing-stop exits show a small *gross* edge (medians +0.03 to +0.09 R on train, +0.02 to +0.05 R on validation); FVG-retest, sweep-reversal and ICT-model family medians are only +0.004 to +0.015 R. The cost per trade at M1 stop sizes is ~0.45 R, roughly ten times larger. Tight-target exits are negative even before costs. |
| Higher signal timeframes | Detecting patterns on M15 bars roughly halves the cost drag (family medians −0.15 to −0.30 R; M5 −0.25 to −0.42 R; M1 −0.44 to −0.51 R) but does not make them positive. |
| Walk-forward (12 m in-sample / 3 m out-of-sample, 10 folds) | A config passed the in-sample gate in only 2 folds; out-of-sample 88 trades, −0.18 R/trade, 0 positive folds. |
| Tick-level audit of the "stop-first" assumption | Only 1 of 160 and 0 of 318 stop-outs were ambiguous bars; with tick-resolved outcomes the 95 % CI of expectancy stays entirely negative (−0.17 to −0.06 R). The conservative rule is not what makes the scalps lose. |
| Confidence score | Not calibrated: validation win rate *fell* as score rose (32 % → 27 % → 11 %); on test it rose (19 % → 25 % → 33 %). It must not be read as a probability. |

### The candidate (selected on train only, failed validation gates)

Pre-registered fallback when nothing passes: highest train t-statistic.
`M15:ema_pullback :: structure stop + 0.1 ATR, no target, 900-minute time exit :: score ≥ 65, volatility filter`

| Period | Trades | Win rate (95 % CI) | Expectancy R (95 % CI, day-clustered) | PF | Max DD (R) |
|---|---|---|---|---|---|
| Train 2022-01 → 2024-06 | 303 | 29.7 % (24.8–35.1) | +0.295 (+0.006 to +0.618) | 1.42 | 28.9 |
| Validation 2024-07 → 2025-06 | 125 | 28.8 % (21.6–37.3) | +0.064 (−0.288 to +0.465) | 1.09 | 20.1 |
| **Test 2025-07 → 2026-10 (evaluated once)** | 177 | 22.6 % (17.1–29.3) | +0.113 (−0.288 to +0.555) | 1.15 | 16.3 |

The test point estimate is positive, but the interval includes zero and the
total is driven by one month (January 2026: +29 R; the other 15 months net
−9 R). Under all adverse conditions combined (2× commission, 3× slippage, one
bar of latency) test expectancy falls to +0.05 R (CI −0.31 to +0.44). **This is
not evidence of an edge, and it should not be traded with real money.** The
test period 2025-07-01 → 2026-10-23 is now recorded as burned in the
`holdout_registry`; the next independent evaluation needs data after
2026-10-23 (e.g. forward paper trading).

### Experiment history

| # | Experiment | Configs | Outcome |
|---|---|---|---|
| 1 | M1 signal timeframe, full grid | 108,864 | nothing positive on train after costs (max t 0.20); `reports/experiments/v1_m1_only/` |
| 1b | zero-cost diagnostic on the same data | 1,134 | small gross edge in some families, far below costs |
| 2 | M1 + M5 + M15 signal timeframes | 326,592 | nothing passes train gate; candidate above |
| 3 | one-time test evaluation of #2's candidate | 1 | not significant (above) |
| — | tick audits on validation | 2 | `reports/tick_audit/` |

The SQLite experiment log is exported to `reports/export/experiments.*`.

### What would have to change

Positive expectancy after costs needs a gross edge larger than the cost per
trade in R. Options, none of which this research validates: a larger account
(the candidate's ~$6 stops at 1 % risk imply roughly £500+), lower-cost
execution, longer holding periods or higher timeframes, or a different
information source than price-pattern geometry. Any new idea should go
through the same protocol on data that has not been looked at yet (forward
data after 2026-10-23); RayGold's `PAPER` mode records those signals and
trades in SQLite for exactly that purpose.

## Limitations

* Dukascopy quotes are one liquidity provider's; your broker's spread, commission and slippage will differ. Set them in `config/raygold.yaml` and re-run.
* Stops are simulated at M1 resolution (conservative on ambiguous bars; tick audit available) and equity in the account replay is closed-trade equity, so intratrade drawdown is understated.
* Dukascopy volume is tick volume, not exchange volume.
* No economic-calendar data is bundled; the blackout works only once you populate `config/news_events.csv`.
* The `MetaTrader5` package is Windows-only; it was exercised with a fake terminal in tests, not against a live broker from this environment.
