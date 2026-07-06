# Traditional Econometric Models for Inflation Forecasting in Brazil

**Phase 2** of a replication project for the paper:

> *"Machine learning methods for inflation forecasting in Brazil: new contenders
> versus classical models."*

This repository covers **only the traditional / classical econometric benchmarks**
(Phase 2). The factor models and machine-learning contenders from the paper belong to
other project phases and are **not** implemented here.

---

## Scope

Replicate the classical benchmarks for forecasting Brazilian headline consumer-price
inflation (IPCA month-on-month), over the monthly sample **2004-01 … 2018-12**, using an
expanding-window recursive out-of-sample design (first forecast origin 2011-01, horizons
`h = 1..12`).

### Implemented models

| Model | Description |
|---|---|
| **RW** | Random Walk — last observed value, flat across horizons. |
| **RW_AO** | Atkeson–Ohanian — mean of the last 48 observations, flat across horizons. |
| **AR1** | AR(1), fit fresh by OLS on the expanding window; closed-form h-step forecast. |
| **VAR1** | VAR(1) over four endogenous variables; headline reconstructed 0.75·market + 0.25·administered. |
| **PC_BACKWARD** | Backward-looking Phillips Curve (direct horizon-specific market regression + AR(1) administered). |

See [`docs/methodology.md`](docs/methodology.md) for full specifications and
[`docs/model_coverage.md`](docs/model_coverage.md) for a compact coverage table.

### Not implemented

- **PC_HYBRID** (hybrid Phillips Curve) — requires Focus-survey `h = 1..12` expected-inflation
  data, which was **not available** in the downloaded CEIC data. See caveats below.
- **Factor models and machine-learning models** — these belong to **other project phases**,
  not Phase 2.

---

## Data policy

- **Raw CEIC files are not committed.** `data/raw/` is gitignored; you must supply the CSVs.
- **Processed data and outputs are gitignored** (`data/processed/`, `outputs/`) — they are
  regenerated from the raw files by the scripts.
- The user must place the CEIC CSV exports into `data/raw/` using the **exact expected
  filenames** below.

### Expected raw filenames (place in `data/raw/`)

```
ipca_general_index.csv
ipca_non_regulated_mom.csv
ipca_administered_prices_sp_total_mom.csv
broad_money_m4.csv
exchange_rate_brl_per_usd_period_avg.csv
ibc_br_seasonally_adjusted.csv
```

Each file is a standard CEIC export (a metadata block followed by `MM/YYYY,value` data
rows covering 2004-01 … 2018-12). The parser locates the data block by regex, so the
varying metadata-block length does not matter. Full inspection in
[`docs/data_audit.md`](docs/data_audit.md).

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the six raw CEIC CSVs in data/raw/ (see filenames above)
```

---

## Reproducibility

Run the full pipeline with the helper script:

```bash
bash scripts/run_all.sh
```

Or run each stage individually, **in order** (each stage depends on the previous one's
outputs):

```bash
python src/prepare_data.py          # parse raw CEIC -> clean dataset
python src/run_baselines.py         # RW, RW_AO, AR1
python src/run_var.py               # VAR1 (+ combined RW/RW_AO/AR1/VAR1)
python src/run_phillips_backward.py # PC_BACKWARD (+ full combined file)
```

Each script prints validation diagnostics and exits non-zero if any hard check fails.

---

## Outputs produced

All written under `outputs/` (gitignored):

| File | Contents |
|---|---|
| `outputs/forecasts/baseline_forecasts.csv` | Per-forecast rows for RW, RW_AO, AR1. |
| `outputs/forecasts/var_forecasts.csv` | Per-forecast rows for VAR1. |
| `outputs/forecasts/pc_backward_forecasts.csv` | Per-forecast rows for PC_BACKWARD. |
| `outputs/forecasts/traditional_forecasts_so_far.csv` | All five models combined. |
| `outputs/tables/*_mse_by_horizon.csv` | MSE tables (per model and combined) by horizon. |
| `outputs/figures/*_mse_by_horizon.png` | MSE-vs-horizon figures (per model and combined). |

---

## Validation checks

Each run enforces (and fails on) the following:

- `h = 1` produces **95** forecasts per model.
- `h = 12` produces **84** forecasts per model.
- Forecast counts decline by exactly 1 per horizon (95 → 84).
- **No target date after 2018-12.**
- **Expanding-window only** — at each origin, only observations up to and including the
  origin are used (no look-ahead; the PC additionally verifies no training pair has a
  target date after its forecast origin).
- **No raw CEIC data is committed** (enforced by `.gitignore`).

---

## Known caveats

- **Administered-prices file `(SP)` token.** `ipca_administered_prices_sp_total_mom.csv`
  has a raw title containing `"(SP)"`, but its metadata (`Region = Brazil`, empty
  `Subnational`) and its pairing with the non-regulated series indicate **national**
  Brazil-wide administered prices. It is treated as national; this is a documented
  judgement, not a certainty — see [`docs/data_audit.md` §3.2](docs/data_audit.md).
- **HP-filter smoothing parameter.** The PC output gap uses `λ = 129600`, the standard
  Ravn–Uhlig value for **monthly** data (`1600 · 12⁴`).
- **PC_HYBRID missing.** The hybrid Phillips Curve is not implemented because Focus-survey
  expected-inflation data (`h = 1..12`) was unavailable in the downloaded CEIC data.

---

## Repository layout

```
data/raw/           # (gitignored) place the six CEIC CSVs here
data/processed/     # (gitignored) generated clean dataset
outputs/            # (gitignored) forecasts, tables, figures
src/
  prepare_data.py
  run_baselines.py
  run_var.py
  run_phillips_backward.py
  models/
    baseline_models.py
    var_model.py
    phillips_curve.py
scripts/
  run_all.sh
docs/
  data_audit.md
  methodology.md
  model_coverage.md
requirements.txt
```
