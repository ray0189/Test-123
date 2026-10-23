# RayGold research report (generated)

Data: Dukascopy XAUUSD M1 bid+ask, 2022-01-02 23:00:00+00:00 .. 2025-06-30 23:59:00+00:00 used for research (1,238,210 bars, file hash `281806b0231ead8e`).

Splits: train `2022-01-01`..`2024-07-01`, validation `2024-07-01`..`2025-07-01`, test `2025-07-01`..`2026-10-23` (test never loaded during research).

Costs: real bid/ask spread per bar, commission 0.07 USD/oz round turn (= 7 USD/lot), slippage 0.05 USD/oz per market fill, stop-first on ambiguous bars, no position held over gaps > 3h.

Configurations evaluated: **326,592** (3 signal timeframes M1/M5/M15 x 27 entry variants x 42 exit specs x 96 filter sets; execution always on M1 bid/ask).

## 1. Pattern-family comparison (train-selected best config per family, then its validation result)

| family | tf | configs | frac_pos_train | frac_pos_both | median_train_exp_r | median_val_exp_r | best_train_n | best_train_wr | best_train_exp_r | best_val_n | best_val_wr | best_val_exp_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| breakout_retest | M15 | 12096 | 0.6% | 0.5% | -0.257 | -0.176 | 305 | 22.3% | +0.174 | 140 | 25.7% | +0.215 |
| fvg_retest | M15 | 12096 | 0.2% | 0.1% | -0.230 | -0.167 | 304 | 35.9% | +0.143 | 125 | 29.6% | +0.121 |
| eq_sweep | M5 | 4032 | 0.0% | 0.0% | -0.420 | -0.202 | 320 | 13.4% | -0.468 | 98 | 23.5% | +0.118 |
| fvg_retest | M5 | 12096 | 0.0% | 0.0% | -0.310 | -0.262 | 427 | 20.1% | -0.048 | 207 | 17.4% | +0.102 |
| ema_pullback | M1 | 4032 | 0.0% | 0.0% | -0.443 | -0.362 | 327 | 9.5% | -0.230 | 198 | 13.6% | +0.097 |
| ema_pullback | M15 | 4032 | 2.6% | 1.8% | -0.153 | -0.077 | 303 | 29.7% | +0.296 | 125 | 28.8% | +0.064 |
| ema_pullback | M5 | 4032 | 0.0% | 0.0% | -0.252 | -0.193 | 422 | 22.3% | -0.026 | 169 | 24.9% | +0.023 |
| choch_reversal | M1 | 12096 | 0.0% | 0.0% | -0.485 | -0.399 | 352 | 30.1% | +0.018 | 143 | 31.5% | +0.010 |
| random_baseline | M15 | 12096 | 0.2% | 0.0% | -0.297 | -0.199 | 362 | 26.2% | +0.057 | 138 | 30.4% | -0.002 |
| ob_retest | M5 | 8064 | 0.1% | 0.0% | -0.294 | -0.275 | 350 | 21.1% | +0.039 | 154 | 20.8% | -0.047 |
| choch_reversal | M5 | 12096 | 1.0% | 0.0% | -0.266 | -0.249 | 301 | 36.2% | +0.159 | 164 | 34.8% | -0.051 |
| sr_rejection | M15 | 4032 | 0.0% | 0.0% | -0.232 | -0.162 | 594 | 11.6% | -0.020 | 252 | 10.7% | -0.071 |
| bos_continuation | M5 | 12096 | 0.0% | 0.0% | -0.290 | -0.221 | 330 | 34.2% | -0.026 | 128 | 31.2% | -0.088 |
| breakout_retest | M5 | 12096 | 0.0% | 0.0% | -0.332 | -0.290 | 409 | 13.7% | -0.019 | 185 | 17.3% | -0.113 |
| ict_model | M15 | 8064 | 0.0% | 0.0% | -0.184 | -0.136 | 380 | 30.8% | +0.018 | 152 | 28.9% | -0.134 |
| rsi_divergence | M1 | 8064 | 0.0% | 0.0% | -0.447 | -0.339 | 335 | 29.9% | -0.162 | 201 | 30.8% | -0.152 |
| ict_model | M5 | 8064 | 0.3% | 0.0% | -0.250 | -0.227 | 304 | 42.8% | +0.030 | 162 | 34.6% | -0.154 |
| bos_continuation | M1 | 12096 | 0.0% | 0.0% | -0.505 | -0.437 | 335 | 29.3% | -0.179 | 162 | 30.9% | -0.159 |
| sr_rejection | M5 | 4032 | 0.0% | 0.0% | -0.372 | -0.307 | 314 | 18.5% | -0.157 | 188 | 21.3% | -0.202 |
| sweep_reversal | M5 | 12096 | 0.0% | 0.0% | -0.397 | -0.356 | 342 | 17.5% | +0.036 | 127 | 12.6% | -0.259 |
| rsi_divergence | M5 | 8064 | 0.0% | 0.0% | -0.281 | -0.257 | 410 | 25.9% | -0.125 | 185 | 22.2% | -0.265 |
| rsi_divergence | M15 | 8064 | 0.0% | 0.0% | -0.177 | -0.174 | 311 | 27.3% | -0.015 | 149 | 22.1% | -0.298 |
| ict_model | M1 | 8064 | 0.0% | 0.0% | -0.454 | -0.371 | 347 | 27.1% | -0.128 | 148 | 21.6% | -0.317 |
| ob_retest | M15 | 8064 | 0.2% | 0.0% | -0.192 | -0.211 | 398 | 18.8% | +0.039 | 154 | 17.5% | -0.318 |
| random_baseline | M5 | 12096 | 0.0% | 0.0% | -0.411 | -0.300 | 340 | 22.1% | +0.005 | 147 | 16.3% | -0.329 |
| bos_continuation | M15 | 12096 | 0.6% | 0.2% | -0.181 | -0.132 | 386 | 26.4% | +0.083 | 181 | 19.9% | -0.338 |
| fvg_retest | M1 | 12096 | 0.0% | 0.0% | -0.481 | -0.427 | 336 | 8.3% | -0.322 | 151 | 7.3% | -0.360 |
| random_baseline | M1 | 12096 | 0.0% | 0.0% | -0.480 | -0.447 | 323 | 26.3% | -0.167 | 132 | 21.2% | -0.391 |
| sr_rejection | M1 | 4032 | 0.0% | 0.0% | -0.483 | -0.436 | 326 | 15.3% | -0.298 | 172 | 15.7% | -0.418 |
| choch_reversal | M15 | 12096 | 0.7% | 0.0% | -0.189 | -0.232 | 378 | 27.8% | +0.069 | 151 | 21.2% | -0.426 |
| eq_sweep | M1 | 4032 | 0.0% | 0.0% | -0.485 | -0.441 | 308 | 13.6% | -0.395 | 182 | 14.8% | -0.427 |
| sweep_reversal | M15 | 12096 | 0.5% | 0.1% | -0.295 | -0.239 | 327 | 20.5% | +0.097 | 114 | 12.3% | -0.446 |
| breakout_retest | M1 | 12096 | 0.0% | 0.0% | -0.505 | -0.472 | 306 | 13.4% | -0.351 | 167 | 10.8% | -0.611 |
| sweep_reversal | M1 | 12096 | 0.0% | 0.0% | -0.490 | -0.457 | 348 | 11.8% | -0.156 | 156 | 10.9% | -0.618 |
| ob_retest | M1 | 8064 | 0.0% | 0.0% | -0.465 | -0.420 | 338 | 11.2% | -0.327 | 243 | 4.9% | -0.677 |
| eq_sweep | M15 | 4032 | - | - | - | - | - | - | - | - | - | - |

`frac_pos_*` = share of the family's configs (with >= minimum trades) whose expectancy after costs is positive on train / on both train and validation. The `random_baseline` family enters at random bars with random direction and is the empirical null.

## 2. Multiple-testing check

- Configs with enough train trades: 195,157; Bonferroni 5% threshold t = 5.02.
- Random-entry baseline: max train t = 0.46, 99th pct = -0.98, max train win rate = 87.2%.
- Pattern configs: max train t = 1.86; positive on train: 0.2%; positive on validation: 1.7%.

## 3. Win rate versus expectancy (validation)

| band | configs | pos_train_and_val | best_val_exp_r | median_val_exp_r |
|---|---|---|---|---|
| [0.0, 0.3) | 100445 | 163 | +0.672 | -0.363 |
| [0.3, 0.4) | 37990 | 69 | +0.364 | -0.275 |
| [0.4, 0.5) | 23278 | 26 | +0.248 | -0.229 |
| [0.5, 0.6) | 17071 | 26 | +0.226 | -0.257 |
| [0.6, 0.7) | 9995 | 0 | +0.175 | -0.220 |
| [0.7, 0.8) | 5831 | 0 | +0.117 | -0.203 |
| [0.8, 0.9) | 3149 | 0 | +0.064 | -0.136 |
| [0.9, 1.01) | 305 | 0 | -0.033 | -0.072 |

Configs with >= 100 trades in both periods. `pos_train_and_val` counts configs whose expectancy is positive after costs in *both* periods.

Highest validation win rates among configs positive on both train and validation:

| variant | exit | session | min_htf | min_score | train_n | train_win_rate | train_exp_r | val_n | val_win_rate | val_exp_r |
|---|---|---|---|---|---|---|---|---|---|---|
| ema_pullback[] | SLstruct+0.1|TP1.5R|T1440 | all | 2 | 65 | 329.0 | 42.9% | +0.026 | 129.0 | 40.3% | +0.003 |
| ema_pullback[] | SLstruct+0.1|TP2R|T1440 | no_asia | -9 | 65 | 303.0 | 35.6% | +0.002 | 117.0 | 36.8% | +0.066 |
| ema_pullback[] | SLstruct+0.1|TP2R|T1440 | ldn_ny | -9 | 65 | 303.0 | 35.6% | +0.002 | 117.0 | 36.8% | +0.066 |
| breakout_retest[n=30] | SLatr1.5|TP2R|T1440 | overlap | 2 | 55 | 330.0 | 35.8% | +0.003 | 151.0 | 35.1% | +0.013 |
| breakout_retest[n=60] | SLatr1.5|TP2R|T1440 | overlap | -9 | 55 | 324.0 | 35.8% | +0.007 | 137.0 | 35.0% | +0.001 |
| ema_pullback[] | SLstruct+0.1|TPnone|T900 | overlap | 1 | 55 | 320.0 | 29.7% | +0.049 | 114.0 | 33.3% | +0.364 |
| ema_pullback[] | SLstruct+0.1|TPnone|T900 | overlap | -9 | 45 | 344.0 | 27.6% | +0.060 | 118.0 | 33.1% | +0.290 |
| ema_pullback[] | SLstruct+0.1|TPnone|T900 | overlap | -9 | 55 | 332.0 | 29.5% | +0.060 | 116.0 | 32.8% | +0.341 |
| ema_pullback[] | SLstruct+0.1|TPnone|T900 | overlap | -9 | 45 | 400.0 | 27.3% | +0.002 | 138.0 | 32.6% | +0.293 |
| ema_pullback[] | SLstruct+0.1|TPnone|T900 | overlap | -9 | 0 | 357.0 | 27.7% | +0.062 | 123.0 | 32.5% | +0.252 |

## 4. Walk-forward (12-month in-sample re-selection, 3-month out-of-sample, train+validation span)

```
{
 "folds": 10,
 "folds_with_selection": 2,
 "oos_trades": 88,
 "oos_win_rate": 0.25,
 "oos_exp_r": -0.17986928874796088,
 "oos_t": -1.0954380162587047,
 "oos_total_r": -15.828497409820557,
 "folds_positive": 0
}
```

| oos_start | oos_end | eligible | variant | exit | is_t | is_exp_r | oos_n | oos_wins | oos_total_r |
|---|---|---|---|---|---|---|---|---|---|
| 2023-01 | 2023-03 | 1 | M15:ema_pullback[] | SLstruct+0.1|TP3R|T1440 | 2.034 | 0.326 | 35.0 | 8.0 | -6.491 |
| 2023-04 | 2023-06 | 0 | - | - | - | - | - | - | - |
| 2023-07 | 2023-09 | 0 | - | - | - | - | - | - | - |
| 2023-10 | 2023-12 | 0 | - | - | - | - | - | - | - |
| 2024-01 | 2024-03 | 0 | - | - | - | - | - | - | - |
| 2024-04 | 2024-06 | 0 | - | - | - | - | - | - | - |
| 2024-07 | 2024-09 | 0 | - | - | - | - | - | - | - |
| 2024-10 | 2024-12 | 0 | - | - | - | - | - | - | - |
| 2025-01 | 2025-03 | 0 | - | - | - | - | - | - | - |
| 2025-04 | 2025-06 | 16 | M15:bos_continuation[k=5] | SLstruct+0.1|TPnone|T900 | 2.309 | 0.286 | 53.0 | 14.0 | -9.337 |

## 5. Selected candidate

`M15:ema_pullback[] :: SLstruct+0.1|TPnone|T900 :: sess=all|htf>=-9|score>=65|vol=1`

Passed every pre-registered gate: **False** (neighbour stability -).

| period | trades | wins/losses/scratch | win rate (95% CI) | expectancy R (95% CI, day-clustered) | t (clustered) | profit factor | avg win / loss R | breakeven WR | total R | max DD R | trades/day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 303 | 90/213/0 | 29.7% (24.8%-35.1%) | +0.295 (+0.006 to +0.618) | 1.84 | 1.42 | +3.38 / -1.01 | 23.0% | +89.5 | 28.9 | 1.24 |
| validation | 125 | 36/89/0 | 28.8% (21.6%-37.3%) | +0.064 (-0.288 to +0.465) | 0.34 | 1.09 | +2.60 / -0.96 | 27.0% | +8.0 | 20.1 | 1.16 |
| TEST (one-time) | 177 | 40/137/0 | 22.6% (17.1%-29.3%) | +0.113 (-0.288 to +0.555) | 0.54 | 1.15 | +3.87 / -0.98 | 20.3% | +20.0 | 16.3 | 1.27 |

### Adverse-condition stress tests

| scenario | period | trades | wins/losses/scratch | win rate (95% CI) | expectancy R (95% CI, day-clustered) | t (clustered) | profit factor | avg win / loss R | breakeven WR | total R | max DD R | trades/day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | validation | 125 | 36/89/0 | 28.8% (21.6%-37.3%) | +0.064 (-0.288 to +0.465) | 0.34 | 1.09 | +2.60 / -0.96 | 27.0% | +8.0 | 20.1 | 1.16 |
| slippage_x3 | validation | 119 | 36/83/0 | 30.3% (22.7%-39.0%) | +0.021 (-0.312 to +0.377) | 0.11 | 1.03 | +2.37 / -1.00 | 29.6% | +2.5 | 18.7 | 1.14 |
| commission_x2 | validation | 124 | 36/88/0 | 29.0% (21.8%-37.6%) | +0.060 (-0.294 to +0.442) | 0.32 | 1.09 | +2.59 / -0.98 | 27.4% | +7.4 | 19.5 | 1.16 |
| latency_1bar | validation | 125 | 38/87/0 | 30.4% (23.0%-38.9%) | +0.042 (-0.288 to +0.413) | 0.24 | 1.06 | +2.39 / -0.98 | 29.2% | +5.2 | 20.4 | 1.16 |
| all_adverse | validation | 119 | 35/84/0 | 29.4% (22.0%-38.1%) | -0.024 (-0.333 to +0.303) | -0.15 | 0.97 | +2.32 / -1.00 | 30.1% | -2.9 | 19.9 | 1.14 |
| favourable_no_commission | validation | 126 | 37/89/0 | 29.4% (22.1%-37.8%) | +0.068 (-0.279 to +0.452) | 0.37 | 1.10 | +2.55 / -0.96 | 27.4% | +8.6 | 19.6 | 1.16 |
| baseline | test | 177 | 40/137/0 | 22.6% (17.1%-29.3%) | +0.113 (-0.288 to +0.555) | 0.54 | 1.15 | +3.87 / -0.98 | 20.3% | +20.0 | 16.3 | 1.27 |
| slippage_x3 | test | 173 | 40/133/0 | 23.1% (17.5%-29.9%) | +0.108 (-0.281 to +0.571) | 0.51 | 1.14 | +3.79 / -1.00 | 20.9% | +18.7 | 15.8 | 1.27 |
| commission_x2 | test | 177 | 40/137/0 | 22.6% (17.1%-29.3%) | +0.106 (-0.295 to +0.549) | 0.51 | 1.14 | +3.86 / -0.99 | 20.4% | +18.8 | 16.5 | 1.27 |
| latency_1bar | test | 175 | 39/136/0 | 22.3% (16.8%-29.0%) | +0.075 (-0.269 to +0.456) | 0.39 | 1.10 | +3.74 / -0.98 | 20.7% | +13.1 | 18.2 | 1.27 |
| all_adverse | test | 171 | 38/133/0 | 22.2% (16.6%-29.0%) | +0.048 (-0.306 to +0.437) | 0.25 | 1.06 | +3.71 / -1.00 | 21.2% | +8.2 | 18.0 | 1.27 |
| favourable_no_commission | test | 177 | 40/137/0 | 22.6% (17.1%-29.3%) | +0.120 (-0.281 to +0.562) | 0.57 | 1.16 | +3.88 / -0.98 | 20.1% | +21.3 | 16.1 | 1.27 |

### GBP 100 account feasibility (risk GBP 1/trade, max 25 positions, aggregate risk GBP 5, daily limit GBP 3, 15% DD breaker)

| period | leverage | signals | taken | final equity | max DD % | win rate | top reject reasons |
|---|---|---|---|---|---|---|---|
| validation | 1:20 | 215 | 0 | 100.0 | 0.0 | - | min lot exceeds risk budget: 215 |
| validation | 1:100 | 215 | 0 | 100.0 | 0.0 | - | min lot exceeds risk budget: 215 |
| validation | 1:500 | 215 | 0 | 100.0 | 0.0 | - | min lot exceeds risk budget: 215 |
| test | 1:20 | 300 | 0 | 100.0 | 0.0 | - | min lot exceeds risk budget: 300 |
| test | 1:100 | 300 | 0 | 100.0 | 0.0 | - | min lot exceeds risk budget: 300 |
| test | 1:500 | 300 | 0 | 100.0 | 0.0 | - | min lot exceeds risk budget: 300 |

### Confidence-score calibration (validation trades of the candidate)

| bucket | trades | win_rate |
|---|---|---|
| [60, 70) | 75 | 0.32 |
| [70, 80) | 41 | 0.268 |
| [80, 101) | 9 | 0.111 |

### Confidence-score calibration (test trades)

| bucket | trades | win_rate |
|---|---|---|
| [60, 70) | 89 | 0.191 |
| [70, 80) | 76 | 0.25 |
| [80, 101) | 12 | 0.333 |
