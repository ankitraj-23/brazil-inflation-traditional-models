# Results Summary

This document summarizes the out-of-sample forecasting results for the classical econometric
benchmarks implemented in this repository. It reports the forecasting design, the verified
forecast counts, and mean squared error (MSE) by horizon. No raw data is reproduced here.

## Implemented models

| Model | Description |
|---|---|
| **RW** | Random Walk — last observed value, flat across horizons. |
| **RW_AO** | Atkeson–Ohanian moving-average random walk — mean of the last 48 observations. |
| **AR1** | AR(1) fit by OLS on the expanding window, closed-form h-step forecast. |
| **VAR1** | First-order VAR over market inflation, administered inflation, ΔM4, and ΔexchangeRate; headline reconstructed as `0.75·market + 0.25·administered`. |
| **PC_BACKWARD** | Backward-looking Phillips Curve — horizon-specific market regression on lagged inflation, imported inflation, and a recursive HP-filter output gap, plus an AR(1) for administered inflation. |

Full specifications are in [`methodology.md`](./methodology.md).

## Forecasting design

- **Expanding-window recursive out-of-sample** evaluation — at each origin, models use only
  observations up to and including that origin, and are re-fit as the window grows.
- **First forecast origin:** 2011-01.
- **Horizons:** `h = 1, …, 12`; for origin `t` and horizon `h`, the target date is `t + h`.
- **Forecast counts per model:** `h = 1` yields **95** forecasts, declining by exactly one
  per horizon down to `h = 12` with **84** forecasts. These counts are enforced by hard
  checks in the run scripts.
- **Combined forecast file:** `outputs/forecasts/traditional_forecasts_so_far.csv` contains
  **5,370** rows — five models × the per-horizon counts summed over `h = 1, …, 12`
  (`5 × (95 + 94 + … + 84) = 5 × 1074 = 5370`).

## MSE by horizon

Mean squared error of the headline IPCA month-on-month forecast, by model and horizon
(lower is better). Source: `outputs/tables/traditional_mse_so_far_by_horizon.csv`, generated
by `bash scripts/run_all.sh`.

| Horizon | n | RW | RW_AO | AR1 | VAR1 | PC_BACKWARD |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 95 | 0.0786 | 0.1029 | 0.0641 | 0.0616 | 0.0655 |
| 2 | 94 | 0.1232 | 0.1042 | 0.0875 | 0.0889 | 0.0951 |
| 3 | 93 | 0.1303 | 0.1048 | 0.0893 | 0.0927 | 0.1052 |
| 4 | 92 | 0.1601 | 0.1055 | 0.0961 | 0.0943 | 0.1114 |
| 5 | 91 | 0.1976 | 0.1072 | 0.1026 | 0.1048 | 0.1357 |
| 6 | 90 | 0.1958 | 0.1080 | 0.1014 | 0.1023 | 0.1368 |
| 7 | 89 | 0.1834 | 0.1089 | 0.1007 | 0.0993 | 0.1390 |
| 8 | 88 | 0.1694 | 0.1108 | 0.1015 | 0.0993 | 0.1223 |
| 9 | 87 | 0.1507 | 0.1126 | 0.1026 | 0.1013 | 0.1040 |
| 10 | 86 | 0.1399 | 0.1148 | 0.1042 | 0.1020 | 0.1020 |
| 11 | 85 | 0.1310 | 0.1168 | 0.1058 | 0.1050 | 0.1014 |
| 12 | 84 | 0.1353 | 0.1191 | 0.1079 | 0.1082 | 0.0985 |

## Interpretation

These results should be read cautiously and without overclaiming, as they reflect a single
sample (Brazilian IPCA, 2004–2018) and a fixed evaluation window. With that caveat:

- The **plain Random Walk (RW)** is the weakest benchmark at most horizons, and its relative
  disadvantage is largest in the medium range (`h ≈ 4–7`), consistent with mean reversion in
  monthly inflation that a flat last-value forecast ignores.
- **AR1** and **VAR1** are close competitors and tend to post the lowest MSE at short and
  intermediate horizons; the differences between them are small.
- **RW_AO** is comparatively stable across horizons, which makes it relatively more
  competitive at longer horizons even though it is not the best at short horizons.
- **PC_BACKWARD** is competitive at the shortest and longest horizons but less so in the
  medium range in this sample.

No single model dominates across all horizons, which is the expected pattern for classical
inflation benchmarks and reinforces their role as reference points rather than a settled
ranking.

## Limitations

- **PC_HYBRID not implemented.** The hybrid Phillips Curve requires Focus-survey h-step-ahead
  expected-inflation data, which was unavailable in the downloaded CEIC dataset. It is left
  out rather than approximated.
- **Raw CEIC data not redistributed.** Results are reproducible only with authorized CEIC
  access — see [`../data/README.md`](../data/README.md).
- **Administered-prices `(SP)` caveat.** The administered-prices file's raw title contains a
  `"(SP)"` token while its metadata indicates Brazil-level coverage; it is treated as national
  administered prices as a documented judgement. See [`data_audit.md` §3.2](./data_audit.md).
- **HP-filter smoothing parameter.** The PC_BACKWARD output gap uses `λ = 129600`, the
  standard Ravn–Uhlig value for monthly data.
