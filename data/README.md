# Data — Raw CEIC exports (not committed)

The raw input series for this project are **CEIC exports**, and they are **intentionally not
committed** to this repository. CEIC data may be licensed or access-controlled, so the raw
files are not redistributed here.

Users with **authorized CEIC access** can reproduce the results by downloading the required
series and placing them in `data/raw/` under the exact filenames listed below, then running
the pipeline.

## Required sample window

All six series must cover the monthly window **2004-01 to 2018-12** (180 observations). The
CEIC exports used in development were already pre-trimmed to this window; the parser also
restricts to it. See [../docs/data_audit.md](../docs/data_audit.md) for the full raw-file
inspection.

## Required raw filenames

Place these files in `data/raw/`:

```
ipca_general_index.csv
ipca_non_regulated_mom.csv
ipca_administered_prices_sp_total_mom.csv
broad_money_m4.csv
exchange_rate_brl_per_usd_period_avg.csv
ibc_br_seasonally_adjusted.csv
```

Each is a standard CEIC CSV export: a metadata block followed by `MM/YYYY,value` data rows.
The parser locates the data block by regex, so a varying metadata-block length does not
matter. A field-by-field mapping of each file to its processed variable, source, unit, and
transformation is in [../docs/data_manifest.md](../docs/data_manifest.md).

## Reproducing results

With the six raw files in place:

```bash
bash scripts/run_all.sh
```

This parses the raw exports into a clean dataset and runs every model stage in dependency
order.

## What is ignored by git

The following are gitignored and never committed:

- `data/raw/` — raw CEIC exports (place them here locally)
- `data/processed/` — the generated clean dataset
- `outputs/` — forecasts, MSE tables, and figures
- `*.csv`, `*.xlsx`, `*.xls` — any tabular data files

Only this `README.md` inside `data/` is tracked.
