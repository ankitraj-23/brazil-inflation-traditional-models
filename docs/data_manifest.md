# Data Manifest

This manifest maps each expected raw CEIC export to the processed variable it produces, its
source and unit, its role in the models, and the transformation applied. It is derived from
[`data_audit.md`](./data_audit.md) (raw-file inspection) and [`methodology.md`](./methodology.md)
(transformations and model specifications).

Raw CEIC exports are **not committed** — see [`../data/README.md`](../data/README.md) for how
to obtain them and reproduce results. No CEIC series codes or mnemonics are reproduced here
beyond those already documented in [`data_audit.md`](./data_audit.md).

## Raw pass-through series

| Expected raw filename | Processed variable | Series title (raw) | Region | Frequency | Unit | Source | Role in model | Transformation |
|---|---|---|---|---|---|---|---|---|
| `ipca_general_index.csv` | `ipca_general_index` | IPCA: General | Brazil | Monthly | Index, Dec1993=100 | IBGE | Source series for headline IPCA inflation | Pass-through (raw level) |
| `ipca_non_regulated_mom.csv` | `ipca_non_regulated_mom` | IPCA: MoM: Non Regulated | Brazil | Monthly | % (MoM) | BCB | Market / free-price inflation: VAR(1) endogenous variable and PC_BACKWARD market regression | Pass-through (already MoM %) |
| `ipca_administered_prices_sp_total_mom.csv` | `ipca_administered_mom` | IPCA: MoM: Administred Prices (SP): Total | Brazil (see `(SP)` caveat) | Monthly | % (MoM) | BCB | Administered-price inflation: VAR(1) endogenous variable and PC_BACKWARD AR(1) component | Pass-through (already MoM %) |
| `broad_money_m4.csv` | `broad_money_m4` | Broad Money Supply: M4 | Brazil | Monthly | BRL mn | BCB | Source series for `m4_diff` | Pass-through (raw level) |
| `exchange_rate_brl_per_usd_period_avg.csv` | `exchange_rate_brl_per_usd` | Exchange Rate against USD: Period Avg: Monthly: Brazil | Brazil | Monthly | BRL per USD (raw label `USD/BRL` is inverted — see audit §3.3) | CEIC Data (history extended from IMF / Federal Reserve) | Source series for `exchange_rate_mom_pct` and `exchange_rate_diff` | Pass-through (raw level, period average) |
| `ibc_br_seasonally_adjusted.csv` | `ibc_br_sa` | Economic Activity Index - IBC-Br: Seasonally Adjusted (sa) | Brazil | Monthly | Index, 2022=100 | BCB | Recursive HP-filter output gap in PC_BACKWARD (λ = 129600) | Pass-through (raw level) |

## Derived variables

| Processed variable | Series title | Region | Frequency | Unit | Derived from | Role in model | Transformation |
|---|---|---|---|---|---|---|---|
| `ipca_headline_mom` | IPCA headline month-on-month inflation | Brazil | Monthly | % (MoM) | `ipca_general_index` | Evaluation target for all models | `100 × (ipca_general_index / lag(ipca_general_index) − 1)` |
| `imported_inflation` | Imported-inflation (exchange-rate pass-through) proxy | Brazil | Monthly | % | `exchange_rate_brl_per_usd` | PC_BACKWARD market regression predictor | `exchange_rate_mom_pct + 0.165`, where `exchange_rate_mom_pct = 100 × (rate / lag(rate) − 1)` |
| `m4_diff` | First difference of M4 | Brazil | Monthly | BRL mn | `broad_money_m4` | VAR(1) endogenous variable | `broad_money_m4 − lag(broad_money_m4)` |
| `exchange_rate_diff` | First difference of BRL/USD exchange rate | Brazil | Monthly | BRL per USD | `exchange_rate_brl_per_usd` | VAR(1) endogenous variable | `exchange_rate_brl_per_usd − lag(exchange_rate_brl_per_usd)` |

## Notes

- **Sample window:** all series cover 2004-01 … 2018-12 (180 monthly observations).
- **First-row NaNs (2004-01):** the derived lag / first-difference variables
  (`ipca_headline_mom`, `imported_inflation`, `m4_diff`, `exchange_rate_diff`, and the
  intermediate `exchange_rate_mom_pct`) are undefined in the first month because no
  in-sample predecessor exists. This is expected.
- **Administered-prices `(SP)` caveat:** the administered-prices file's raw title contains a
  `"(SP)"` token, while its metadata indicates Brazil-level coverage (`Region = Brazil`,
  empty `Subnational`) and it pairs with the non-regulated series as the standard national
  free/administered decomposition. It is treated as national administered prices — a
  documented judgement, not a certainty. See [`data_audit.md` §3.2](./data_audit.md).
