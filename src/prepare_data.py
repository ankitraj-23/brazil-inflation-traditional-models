"""Parse and standardize the raw CEIC CSV exports into a clean monthly dataset.

Phase 2 (traditional econometric models) of the replication of
"Machine learning methods for inflation forecasting in Brazil".

This module ONLY prepares data. No forecasting model is implemented here.

Each raw CEIC file has the same shape:
    - a metadata block (Region, Frequency, Unit, Source, Series ID, summary stats...)
    - a data block of `MM/YYYY,value[,value]` rows

The metadata "First/Last Obs. Date" fields describe the *full* CEIC series, but the
exported data block for every file already covers exactly 2004-01 .. 2018-12 (180 obs).
We locate the data block robustly by regex rather than by fixed row offsets, because the
metadata block length varies between files (multi-line "Series Remarks" push it down).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "traditional_models_clean_data.csv"

# Analysis window (inclusive), matching the replication paper's sample.
START = pd.Period("2004-01", freq="M")
END = pd.Period("2018-12", freq="M")

# A data row starts with a MM/YYYY token in the first column.
DATE_RE = re.compile(r"^(0[1-9]|1[0-2])/(\d{4})$")

# Raw file -> clean column name holding that file's *raw* value (before transforms).
#
# NOTE on ipca_administered_prices_sp_total_mom.csv:
#   The raw series title is "IPCA: MoM: Administred Prices (SP): Total", but its
#   metadata reports Region=Brazil with an empty Subnational field, and it pairs with
#   the "Non Regulated" series to form the standard national IPCA free-vs-administered
#   decomposition (both sourced from the Central Bank of Brazil). We therefore treat it
#   as NATIONAL administered/monitored prices and name the column ipca_administered_mom.
#   The "(SP)" token appears to be a CEIC series-label artifact, not "Sao Paulo".
#   See docs/data_audit.md for the full reasoning and warning.
RAW_FILES: dict[str, str] = {
    "ipca_general_index": "ipca_general_index.csv",
    "ipca_non_regulated_mom": "ipca_non_regulated_mom.csv",
    "ipca_administered_mom": "ipca_administered_prices_sp_total_mom.csv",
    "broad_money_m4": "broad_money_m4.csv",
    "exchange_rate_brl_per_usd": "exchange_rate_brl_per_usd_period_avg.csv",
    "ibc_br_sa": "ibc_br_seasonally_adjusted.csv",
}


def read_ceic_series(path: Path, name: str) -> pd.Series:
    """Read a single CEIC CSV and return the value series indexed by monthly Period.

    Only rows whose first cell matches MM/YYYY are kept, so the metadata block is
    skipped automatically. The first value column is used; some files (e.g. M4) carry a
    duplicated identical value column, which is ignored.
    """
    periods: list[pd.Period] = []
    values: list[float] = []

    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            key = row[0].strip()
            if not DATE_RE.match(key):
                continue
            month, year = key.split("/")
            raw = row[1].strip() if len(row) > 1 else ""
            values.append(float(raw) if raw not in ("", "-") else float("nan"))
            periods.append(pd.Period(f"{year}-{month}", freq="M"))

    series = pd.Series(values, index=pd.PeriodIndex(periods, freq="M"), name=name)
    series = series.sort_index()

    if series.index.has_duplicates:
        raise ValueError(f"{path.name}: duplicate dates found")
    return series


def load_raw() -> pd.DataFrame:
    """Load all raw series into a single monthly DataFrame, filtered to the window."""
    frame = pd.DataFrame(
        {name: read_ceic_series(RAW_DIR / fname, name) for name, fname in RAW_FILES.items()}
    )
    frame = frame.sort_index()

    # Restrict to 2004-01 .. 2018-12.
    frame = frame.loc[(frame.index >= START) & (frame.index <= END)]

    # Sanity checks: complete, gap-free, non-missing monthly coverage.
    expected = pd.period_range(START, END, freq="M")
    missing_months = expected.difference(frame.index)
    if len(missing_months):
        raise ValueError(f"Missing months in window: {list(missing_months)}")
    na_counts = frame.isna().sum()
    if na_counts.any():
        raise ValueError(f"Unexpected missing values in raw window:\n{na_counts[na_counts > 0]}")
    return frame


def build_transformations(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns required by the traditional models.

    First-row derived values (2004-01) are NaN because the lag/first-difference has no
    prior observation inside the exported window (the raw files start at 2004-01).
    """
    out = frame.copy()

    out["ipca_headline_mom"] = 100.0 * (out["ipca_general_index"] / out["ipca_general_index"].shift(1) - 1.0)
    out["exchange_rate_mom_pct"] = 100.0 * (
        out["exchange_rate_brl_per_usd"] / out["exchange_rate_brl_per_usd"].shift(1) - 1.0
    )
    out["imported_inflation"] = out["exchange_rate_mom_pct"] + 0.165
    out["m4_diff"] = out["broad_money_m4"].diff()

    # Final column order.
    columns = [
        "ipca_general_index",
        "ipca_headline_mom",
        "ipca_non_regulated_mom",
        "ipca_administered_mom",
        "broad_money_m4",
        "exchange_rate_brl_per_usd",
        "ibc_br_sa",
        "exchange_rate_mom_pct",
        "imported_inflation",
        "m4_diff",
    ]
    return out[columns]


def main() -> None:
    raw = load_raw()
    clean = build_transformations(raw)

    # Emit a month-end datetime `date` column for consistency downstream.
    clean = clean.copy()
    clean.insert(0, "date", clean.index.to_timestamp(how="end").normalize())
    clean = clean.reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT_FILE, index=False, date_format="%Y-%m-%d")

    # Report.
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(f"Wrote {OUT_FILE}")
    print(f"\nShape: {clean.shape}")
    print(f"\nColumns:\n{list(clean.columns)}")
    print(f"\nFirst 5 rows:\n{clean.head().to_string()}")
    print(f"\nLast 5 rows:\n{clean.tail().to_string()}")
    print(f"\nMissing values per column:\n{clean.isna().sum().to_string()}")


if __name__ == "__main__":
    main()
