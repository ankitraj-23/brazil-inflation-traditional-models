# Model Coverage — Phase 2 (Traditional Econometric Models)

Replication of *"Machine learning methods for inflation forecasting in Brazil: new
contenders versus classical models."* Scope: **traditional / classical benchmarks only**.

Common design for every implemented model: expanding-window recursive out-of-sample, first
origin 2011-01, horizons `h = 1..12` (target date `t + h`), evaluation target
`ipca_headline_mom`. Counts: `h=1` → 95 forecasts, declining by 1 per horizon to `h=12` → 84.

| Model | Implemented | Input variables | Output file | Notes / caveats |
|---|---|---|---|---|
| **RW** | Yes | `ipca_headline_mom` | `outputs/forecasts/baseline_forecasts.csv` | Last observed value, flat across horizons. |
| **RW_AO** | Yes | `ipca_headline_mom` | `outputs/forecasts/baseline_forecasts.csv` | Atkeson–Ohanian: mean of last 48 obs; raises if <48 available. |
| **AR1** | Yes | `ipca_headline_mom` | `outputs/forecasts/baseline_forecasts.csv` | OLS AR(1) fit fresh on expanding window; closed-form h-step forecast. |
| **VAR1** | Yes | `ipca_non_regulated_mom`, `ipca_administered_mom`, `m4_diff`, `exchange_rate_diff` | `outputs/forecasts/var_forecasts.csv` | VAR(1) + constant; headline = 0.75·market + 0.25·administered. Administered `(SP)` caveat (audit §3.2). |
| **PC_BACKWARD** | Yes | `ipca_non_regulated_mom`, `imported_inflation`, `ibc_br_sa` (HP gap), `ipca_administered_mom` | `outputs/forecasts/pc_backward_forecasts.csv` | Direct horizon-specific market regression + AR(1) administered; headline = 0.75·market + 0.25·administered. Recursive HP gap, λ=129600 (monthly). Administered `(SP)` caveat. |
| **PC_HYBRID** | **No** | *(would add Focus `h=1..12` expected inflation)* | — | **Focus-survey expected-inflation data unavailable** in the downloaded CEIC data; cannot be implemented in this phase. |
| Factor models | No | — | — | Belong to another project phase (out of Phase 2 scope). |
| ML models | No | — | — | Belong to another project phase (out of Phase 2 scope). |

**Combined outputs** (all implemented models): forecasts in
`outputs/forecasts/traditional_forecasts_so_far.csv`, MSE in
`outputs/tables/traditional_mse_so_far_by_horizon.csv`, figure in
`outputs/figures/traditional_mse_so_far_by_horizon.png`.

See [`methodology.md`](methodology.md) for full model specifications and
[`data_audit.md`](data_audit.md) for the raw-data audit and the `(SP)` administered-prices
judgement.
