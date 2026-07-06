# Methodology

**Project:** Phase 2 — traditional econometric models for Brazilian inflation forecasting.
Replication of *"Machine learning methods for inflation forecasting in Brazil: new
contenders versus classical models."*

> ⚠️ **Status: BASELINE + VAR(1) + BACKWARD PHILLIPS CURVE IMPLEMENTED.** The audited clean
> dataset (§1–§4), the three baseline benchmarks — RW, RW_AO, AR(1) (§5) — the VAR(1) model
> (§6) and the backward-looking Phillips Curve (§7) are done. The hybrid Phillips Curve,
> Focus expectations, factor models and ML models remain out of scope (§8).

---

## 1. Project objective

Replicate the **classical / traditional econometric** benchmarks used in the paper for
forecasting Brazilian consumer-price inflation (IPCA), over the monthly sample
**2004-01 … 2018-12**. This phase's deliverable is a single, reproducible clean dataset
that later model code will consume. The machine-learning contenders are out of scope for
this phase.

---

## 2. Raw data sources

All series are CEIC exports in `data/raw/` (read-only; **not committed** — excluded via
`.gitignore`). Full inspection in [`docs/data_audit.md`](./data_audit.md).

| Clean variable | Raw file | Series | Source | Unit |
|---|---|---|---|---|
| `ipca_general_index` | `ipca_general_index.csv` | IPCA: General (price index level) | IBGE | Dec1993=100 |
| `ipca_non_regulated_mom` | `ipca_non_regulated_mom.csv` | IPCA MoM: Non Regulated (free prices) | BCB | % |
| `ipca_administered_mom` | `ipca_administered_prices_sp_total_mom.csv` | IPCA MoM: Administered prices (national — see audit §3.2) | BCB | % |
| `broad_money_m4` | `broad_money_m4.csv` | Broad Money Supply: M4 | BCB | BRL mn |
| `exchange_rate_brl_per_usd` | `exchange_rate_brl_per_usd_period_avg.csv` | Exchange rate, period average (BRL per USD) | CEIC / IMF / FRB | BRL per USD |
| `ibc_br_sa` | `ibc_br_seasonally_adjusted.csv` | IBC-Br economic activity index, seasonally adjusted | BCB | 2022=100 |

Sample: **180 monthly observations**, 2004-01 … 2018-12, no gaps, no missing raw values.

Two documented data caveats (details in the audit):
- The exchange-rate file's unit **label** reads `USD/BRL` but the **values are BRL per
  USD**; used as-is (correct direction for the intended variable).
- The administered-prices file's raw title contains `(SP)`; treated as **national**
  administered prices based on metadata (`Region=Brazil`, empty `Subnational`) and its
  pairing with the non-regulated series. Flagged as a judgement, not a certainty.

---

## 3. Transformations

Implemented in [`src/prepare_data.py`](../src/prepare_data.py), producing
`data/processed/traditional_models_clean_data.csv` (180 × 12).

Raw pass-through columns: `ipca_general_index`, `ipca_non_regulated_mom`,
`ipca_administered_mom`, `broad_money_m4`, `exchange_rate_brl_per_usd`, `ibc_br_sa`.

Derived columns:

| Column | Definition |
|---|---|
| `ipca_headline_mom` | `100 × (ipca_general_index / lag(ipca_general_index) − 1)` — headline IPCA month-on-month % from the index level |
| `exchange_rate_mom_pct` | `100 × (exchange_rate_brl_per_usd / lag(exchange_rate_brl_per_usd) − 1)` — monthly % change of the BRL/USD rate |
| `imported_inflation` | `exchange_rate_mom_pct + 0.165` — exchange-rate pass-through proxy with the paper's additive constant |
| `m4_diff` | `broad_money_m4 − lag(broad_money_m4)` — first difference of M4 (BRL mn) |
| `exchange_rate_diff` | `exchange_rate_brl_per_usd − lag(exchange_rate_brl_per_usd)` — first difference of the nominal BRL/USD rate (level), used as a VAR endogenous variable (distinct from `exchange_rate_mom_pct`, the percent change) |

**Dates** are stored as **month-end datetimes** (`YYYY-MM-DD`, e.g. `2004-01-31`).

**First-row NaNs (2004-01):** `ipca_headline_mom`, `exchange_rate_mom_pct`,
`imported_inflation`, `m4_diff`, and `exchange_rate_diff` are undefined in the first month
because the lag / first difference has no in-sample predecessor. This is expected.

---

## 4. Reproducibility

```bash
python src/prepare_data.py
```

Reads only `data/raw/`, writes only
`data/processed/traditional_models_clean_data.csv`. The parser locates each file's data
block by regex (robust to the varying metadata-block length) and never edits raw files.

---

## 5. Baseline forecasting models (implemented)

**Target:** `ipca_headline_mom` (headline IPCA month-on-month %).
**Code:** [`src/models/baseline_models.py`](../src/models/baseline_models.py) (models) and
[`src/run_baselines.py`](../src/run_baselines.py) (exercise + validation + outputs).

### 5.1 Expanding-window recursive out-of-sample design

Forecasts are produced by a recursive, expanding-window exercise — never using future
data:

- **First forecast origin:** 2011-01.
- **Horizons:** `h = 1..12`. For an origin `t` and horizon `h`, the **target date is
  `t + h` months**.
- **Information set:** at each origin `t`, models see only observations up to and
  including `t`. As origins advance the training window grows (expanding, not rolling).
- **Right edge:** a `(t, h)` pair is produced only if `t + h ≤ 2018-12`.
- **Resulting counts:** `h=1` → **95** forecasts, declining by exactly 1 per horizon down
  to `h=12` → **84** (last origins recede as the horizon lengthens). Verified by hard
  checks in `run_baselines.py`.

### 5.2 Models

| Model | h-step forecast from origin `t` |
|---|---|
| **RW** (Random Walk) | `ŷ(t+h) = y_t` for every `h` — the last observed value, flat across horizons. |
| **RW_AO** (Atkeson–Ohanian) | `ŷ(t+h) = mean(last 48 obs up to t)` for every `h` — a flat 48-month moving average. If fewer than 48 observations are available the code **raises** (never silently shrinks the window). |
| **AR(1)** | Fit `y_s = α + β·y_{s-1}` by OLS on training data up to `t`, then `ŷ(t+h) = β^h · y_t + α · Σ_{i=0}^{h-1} β^i`. The geometric sum uses `(1−β^h)/(1−β)`, switching to its limit `h` when `β` is within `1e-8` of 1 (safe near a unit root). |

AR(1) is fit fresh at every origin on the expanding window. No `auto_arima`, no ARMA(1,1),
no full-sample fitting, and no future data are used.

### 5.3 Outputs

- `outputs/forecasts/baseline_forecasts.csv` — one row per `(origin, horizon, model)` with
  `forecast_origin, target_date, horizon, model, forecast, actual, error, squared_error`
  (`error = actual − forecast`). 3,222 rows.
- `outputs/tables/baseline_mse_by_horizon.csv` — `horizon, model, mse, n_forecasts`
  (`mse = mean(squared_error)`).
- `outputs/figures/baseline_mse_by_horizon.png` — MSE vs horizon, one line per model.

Reproduce with `python src/run_baselines.py` (prints validation diagnostics).

---

## 6. VAR(1) model (implemented)

First-order vector autoregression, following the paper's specification.
**Code:** [`src/models/var_model.py`](../src/models/var_model.py) (model) and
[`src/run_var.py`](../src/run_var.py) (exercise + validation + outputs).

### 6.1 Endogenous variables and lag

The VAR is estimated over **four** endogenous variables (fixed order), with **one lag** and
a **constant**:

1. `ipca_non_regulated_mom` — market / non-regulated IPCA inflation
2. `ipca_administered_mom` — administered IPCA inflation
3. `m4_diff` — **first difference** of M4
4. `exchange_rate_diff` — **first difference** of the nominal BRL/USD exchange rate (level)

M4 and the exchange rate enter the VAR **first-differenced** (they are non-stationary in
level); the two IPCA components are already monthly rates and enter as-is. Fitting uses
`statsmodels.tsa.api.VAR` with `fit(1, trend="c")` — no auto lag selection, no full-sample
fitting.

> ⚠️ **Administered-prices caveat.** The administered component comes from the CEIC file
> whose raw title reads *"IPCA: MoM: Administred Prices (SP): Total"*. The `(SP)` token is
> treated as a series-label artifact and the series as **national** administered prices
> (metadata `Region=Brazil`, empty `Subnational`, and its pairing with the non-regulated
> series). This is a documented judgement, not a certainty — see
> [`docs/data_audit.md` §3.2](./data_audit.md).

### 6.2 Headline reconstruction

Headline IPCA is **not** modelled directly inside the VAR. At each origin the two forecast
inflation components are combined into the headline forecast:

```
headline_forecast(t+h) = 0.75 · market_forecast(t+h) + 0.25 · administered_forecast(t+h)
```

The evaluation target is still `ipca_headline_mom` (the actual value at `t + h`).

### 6.3 Design

Identical expanding-window recursive OOS protocol as the baselines (§5.1): first origin
2011-01, horizons `h = 1..12` with target date `t + h`, only observations up to and
including `t` used at each origin (`df.loc[:t]`), and `(t, h)` produced only if
`t + h ≤ 2018-12`. Rows with NaN among the four VAR variables are dropped before fitting
(the 2004-01 first-difference NaNs). The VAR is re-fit fresh at every origin; the model
forecasts `max(h)` steps and the h-th step is used. Counts match the baselines: `h=1` →
**95** forecasts down to `h=12` → **84** (verified by hard checks in `run_var.py`).

### 6.4 Outputs

- `outputs/forecasts/var_forecasts.csv` — one row per `(origin, horizon)` for `VAR1`, same
  schema as the baseline forecast file (1,074 rows).
- `outputs/tables/var_mse_by_horizon.csv` — `horizon, model, mse, n_forecasts`.
- `outputs/forecasts/traditional_forecasts_so_far.csv` — the baseline forecasts (RW, RW_AO,
  AR1) concatenated with `VAR1` (4,296 rows).
- `outputs/tables/traditional_mse_so_far_by_horizon.csv` — combined MSE by horizon for all
  four models.
- `outputs/figures/traditional_mse_so_far_by_horizon.png` — MSE vs horizon, one line per
  model.

Reproduce with `python src/run_var.py` (requires `python src/run_baselines.py` to have been
run first, since it merges the baseline forecast file). Prints validation diagnostics.

---

## 7. Backward-looking Phillips Curve (implemented)

Backward-looking (adaptive-expectations) Phillips Curve, following the paper.
**Code:** [`src/models/phillips_curve.py`](../src/models/phillips_curve.py) (model) and
[`src/run_phillips_backward.py`](../src/run_phillips_backward.py) (exercise + validation +
outputs). Model name: **`PC_BACKWARD`**.

### 7.1 Market / non-regulated inflation

Market inflation is modelled by a **horizon-specific ("direct") backward regression** —
one OLS fit per horizon `h`, mapping origin-dated predictors to the `h`-step-ahead market
inflation:

```
market_inflation(t+h) = a0
                      + a1 · market_inflation(t)
                      + a2 · imported_inflation(t)
                      + a3 · output_gap(t)
                      + error
```

where `market_inflation = ipca_non_regulated_mom` and `imported_inflation` is the
exchange-rate pass-through proxy (§3). Coefficients `(a0, a1, a2, a3)` are re-estimated
fresh at every origin `t` on the expanding window, separately for each horizon.

### 7.2 Recursive HP-filter output gap

At each forecast origin `t` the **output gap** is computed *recursively*: the
Hodrick–Prescott filter is applied to `ibc_br_sa` using **only observations up to and
including `t`**, and its **cycle** component is used as `output_gap`. The same
recursively-computed cycle supplies `output_gap(s)` for every training predictor date `s`
and `output_gap(t)` for the current-origin predictor, so the filter never sees post-origin
data (no look-ahead). The gap is **never** computed on the full sample.

**Smoothing parameter:** `λ = 129600`, the standard value for **monthly** data
(Ravn–Uhlig scaling, `λ = 1600 · 12⁴ = 129600`; 1600 is the classic quarterly value). This
choice is fixed across all origins.

### 7.3 Administered inflation — AR(1)

Administered inflation (`ipca_administered_mom`) is forecast **separately** with an
**AR(1)** (the paper specifies `p = 1, q = 0`), reusing the closed-form AR(1) from the
baselines (§5.2): `ŷ(t+h) = β^h · y_t + α · Σ_{i=0}^{h-1} β^i`, fit fresh on the expanding
window at every origin.

> ⚠️ **Administered-prices caveat.** The administered component comes from the CEIC file
> whose raw title reads *"IPCA: MoM: Administred Prices (SP): Total"*. The `(SP)` token is
> treated as a series-label artifact and the series as **national** administered prices
> (metadata `Region=Brazil`, empty `Subnational`, and its pairing with the non-regulated
> series). This is a documented judgement, not a certainty — see
> [`docs/data_audit.md` §3.2](./data_audit.md).

### 7.4 Headline reconstruction

Headline IPCA is **not** modelled directly. At each origin the two component forecasts are
combined with the paper's weights:

```
headline_forecast(t+h) = 0.75 · market_forecast(t+h) + 0.25 · administered_forecast(t+h)
```

The evaluation target is `ipca_headline_mom` (the actual value at `t + h`).

### 7.5 Design and no-look-ahead rules

Identical expanding-window recursive OOS protocol as the baselines (§5.1) and the VAR
(§6.3): first origin 2011-01, horizons `h = 1..12` with target date `t + h`, only
observations up to and including `t` used at each origin, and `(t, h)` produced only if
`t + h ≤ 2018-12`. Additional PC-specific guards:

- For each horizon `h`, the training pairs satisfy **`predictor_date + h ≤ t`** — the model
  is never trained on a target value observed after the origin. The run script verifies no
  training pair has `target_date > forecast_origin`.
- The predictor row at date `s = 2004-01` is dropped because `imported_inflation` is NaN
  there (first-difference edge, §3). The minimum number of training rows used across all PC
  regressions is **72** (first origin, `h = 12`).

Counts match the other models: `h=1` → **95** forecasts down to `h=12` → **84** (verified
by hard checks in `run_phillips_backward.py`).

### 7.6 Outputs

- `outputs/forecasts/pc_backward_forecasts.csv` — one row per `(origin, horizon)` for
  `PC_BACKWARD`, same schema as the other forecast files (1,074 rows).
- `outputs/tables/pc_backward_mse_by_horizon.csv` — `horizon, model, mse, n_forecasts`.
- `outputs/forecasts/traditional_forecasts_so_far.csv` — RW, RW_AO, AR1, VAR1 and
  PC_BACKWARD (5,370 rows).
- `outputs/tables/traditional_mse_so_far_by_horizon.csv` — combined MSE by horizon for all
  five models.
- `outputs/figures/traditional_mse_so_far_by_horizon.png` — MSE vs horizon, one line per
  model.

Reproduce with `python src/run_phillips_backward.py` (requires `run_baselines.py` and
`run_var.py` to have been run first, since it rebuilds the combined file from their forecast
outputs). Prints validation diagnostics.

---

## 8. Next planned models (NOT yet implemented)

Out of scope for the current deliverable:

1. **Hybrid Phillips Curve** — adds forward-looking (Focus) expectations to the backward
   specification.
2. **Focus expectations, factor models, and machine-learning contenders.**

Until each is implemented and validated, **no results for it should be reported.**
