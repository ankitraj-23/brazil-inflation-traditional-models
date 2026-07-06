"""Run the Phase 2 baseline out-of-sample forecasting exercise.

Models: RW, RW_AO, AR(1) (see src/models/baseline_models.py).
Target: ipca_headline_mom.

Design: expanding-window, recursive, out-of-sample.
    - First forecast origin: 2011-01.
    - Horizons h = 1..12; for origin t and horizon h the target date is t + h months.
    - At each origin only observations up to and including t are used (no look-ahead).
    - A (origin, horizon) pair is produced only if its target date <= 2018-12.
    - Expected counts: h=1 -> 95 forecasts, declining by 1 per horizon, h=12 -> 84.

Outputs:
    outputs/forecasts/baseline_forecasts.csv
    outputs/tables/baseline_mse_by_horizon.csv
    outputs/figures/baseline_mse_by_horizon.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make `models` importable when run as `python src/run_baselines.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.baseline_models import (  # noqa: E402
    RW_AO_WINDOW,
    ar1_fit,
    ar1_forecast,
    rw_ao_forecast,
    rw_forecast,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "traditional_models_clean_data.csv"
FORECAST_FILE = ROOT / "outputs" / "forecasts" / "baseline_forecasts.csv"
MSE_FILE = ROOT / "outputs" / "tables" / "baseline_mse_by_horizon.csv"
FIG_FILE = ROOT / "outputs" / "figures" / "baseline_mse_by_horizon.png"

TARGET = "ipca_headline_mom"
FIRST_ORIGIN = pd.Period("2011-01", freq="M")
LAST_TARGET = pd.Period("2018-12", freq="M")
HORIZONS = list(range(1, 13))
MODELS = ["RW", "RW_AO", "AR1"]

FORECAST_COLUMNS = [
    "forecast_origin",
    "target_date",
    "horizon",
    "model",
    "forecast",
    "actual",
    "error",
    "squared_error",
]


def load_target() -> pd.Series:
    """Load the target series indexed by monthly Period, dropping the leading NaN."""
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df["period"] = df["date"].dt.to_period("M")
    series = df.set_index("period")[TARGET].dropna().sort_index()
    return series


def run_forecasts(target: pd.Series) -> pd.DataFrame:
    """Execute the expanding-window recursive OOS exercise across all models."""
    origins = [
        p for p in target.index if p >= FIRST_ORIGIN and (p + 1) <= LAST_TARGET
    ]

    records: list[dict] = []
    for t in origins:
        y_train = target.loc[:t].to_numpy(dtype=float)
        y_t = float(y_train[-1])
        horizons = [h for h in HORIZONS if (t + h) <= LAST_TARGET]

        alpha, beta = ar1_fit(y_train)
        preds = {
            "RW": rw_forecast(y_train, horizons),
            "RW_AO": rw_ao_forecast(y_train, horizons, window=RW_AO_WINDOW),
            "AR1": ar1_forecast(alpha, beta, y_t, horizons),
        }

        for h in horizons:
            target_date = t + h
            actual = float(target.loc[target_date])
            for model in MODELS:
                forecast = preds[model][h]
                error = actual - forecast
                records.append(
                    {
                        "forecast_origin": str(t),
                        "target_date": str(target_date),
                        "horizon": h,
                        "model": model,
                        "forecast": forecast,
                        "actual": actual,
                        "error": error,
                        "squared_error": error**2,
                    }
                )

    return pd.DataFrame.from_records(records, columns=FORECAST_COLUMNS)


def build_mse_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MSE and forecast count by (horizon, model)."""
    grouped = (
        forecasts.groupby(["horizon", "model"])
        .agg(mse=("squared_error", "mean"), n_forecasts=("squared_error", "size"))
        .reset_index()
        .sort_values(["horizon", "model"])
        .reset_index(drop=True)
    )
    return grouped


def plot_mse(mse_table: pd.DataFrame) -> None:
    """Line plot of MSE vs horizon, one line per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        sub = mse_table[mse_table["model"] == model].sort_values("horizon")
        ax.plot(sub["horizon"], sub["mse"], marker="o", label=model)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Baseline out-of-sample MSE by horizon\n(target: ipca_headline_mom)")
    ax.set_xticks(HORIZONS)
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_FILE, dpi=150)
    plt.close(fig)


def validate(forecasts: pd.DataFrame, mse_table: pd.DataFrame) -> bool:
    """Print validation diagnostics and return True if all hard checks pass."""
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print("\n[1] Forecast counts by horizon and model:")
    counts = (
        forecasts.groupby(["horizon", "model"]).size().unstack("model").fillna(0).astype(int)
    )
    print(counts.to_string())

    print("\n[2] Missing values by output column:")
    print(forecasts.isna().sum().to_string())

    print("\n[3] First 10 forecast rows:")
    print(forecasts.head(10).to_string(index=False))

    print("\n[4] MSE table:")
    print(mse_table.to_string(index=False))

    print("\n[5] Hard checks:")
    all_ok = True
    for model in MODELS:
        n1 = int(counts.loc[1, model])
        n12 = int(counts.loc[12, model])
        ok1 = n1 == 95
        ok12 = n12 == 84
        all_ok &= ok1 and ok12
        print(f"    {model:6s} h=1 -> {n1} (expect 95) {'OK' if ok1 else 'FAIL'}"
              f" | h=12 -> {n12} (expect 84) {'OK' if ok12 else 'FAIL'}")
    # Counts should decline by exactly 1 per horizon (per model).
    decline_ok = True
    for model in MODELS:
        seq = [int(counts.loc[h, model]) for h in HORIZONS]
        if seq != list(range(95, 83, -1)):
            decline_ok = False
    all_ok &= decline_ok
    print(f"    Counts decline 95->84 by 1 per horizon: {'OK' if decline_ok else 'FAIL'}")
    print(f"\n    ALL CHECKS: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> None:
    target = load_target()
    forecasts = run_forecasts(target)
    mse_table = build_mse_table(forecasts)

    for path in (FORECAST_FILE, MSE_FILE, FIG_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)

    forecasts.to_csv(FORECAST_FILE, index=False)
    mse_table.to_csv(MSE_FILE, index=False)
    plot_mse(mse_table)

    ok = validate(forecasts, mse_table)

    print("\nWrote:")
    print(f"  {FORECAST_FILE.relative_to(ROOT)}  ({len(forecasts)} rows)")
    print(f"  {MSE_FILE.relative_to(ROOT)}  ({len(mse_table)} rows)")
    print(f"  {FIG_FILE.relative_to(ROOT)}")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
