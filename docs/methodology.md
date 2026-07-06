# Methodology

**Project:** Phase 2 — traditional econometric models for Brazilian inflation forecasting.
Replication of *"Machine learning methods for inflation forecasting in Brazil: new
contenders versus classical models."*

> ⚠️ **Status: BASELINE MODELS IMPLEMENTED.** The audited clean dataset (§1–§4) plus the
> three baseline benchmarks — RW, RW_AO, AR(1) — are done (§5). VAR(1) and the Phillips
> Curve remain out of scope for now (§6).

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
`data/processed/traditional_models_clean_data.csv` (180 × 11).

Raw pass-through columns: `ipca_general_index`, `ipca_non_regulated_mom`,
`ipca_administered_mom`, `broad_money_m4`, `exchange_rate_brl_per_usd`, `ibc_br_sa`.

Derived columns:

| Column | Definition |
|---|---|
| `ipca_headline_mom` | `100 × (ipca_general_index / lag(ipca_general_index) − 1)` — headline IPCA month-on-month % from the index level |
| `exchange_rate_mom_pct` | `100 × (exchange_rate_brl_per_usd / lag(exchange_rate_brl_per_usd) − 1)` — monthly % change of the BRL/USD rate |
| `imported_inflation` | `exchange_rate_mom_pct + 0.165` — exchange-rate pass-through proxy with the paper's additive constant |
| `m4_diff` | `broad_money_m4 − lag(broad_money_m4)` — first difference of M4 (BRL mn) |

**Dates** are stored as **month-end datetimes** (`YYYY-MM-DD`, e.g. `2004-01-31`).

**First-row NaNs (2004-01):** `ipca_headline_mom`, `exchange_rate_mom_pct`,
`imported_inflation`, and `m4_diff` are undefined in the first month because the
lag / first difference has no in-sample predecessor. This is expected.

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

## 6. Next planned models (NOT yet implemented)

Out of scope for the current deliverable:

1. **VAR(1)** — first-order vector autoregression over the core variable set.
2. **Backward-looking Phillips Curve** — inflation on its own lags plus an activity /
   slack term (IBC-Br) and imported-inflation channel.

Until these are implemented and validated, **no results for them should be reported.**
