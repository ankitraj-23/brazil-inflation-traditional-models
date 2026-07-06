"""Run the VAR(1) out-of-sample forecasting exercise.

Model: VAR(1) over four endogenous variables (see src/models/var_model.py). Headline IPCA
is reconstructed as 0.75 * market inflation + 0.25 * administered inflation; it is never
modelled directly inside the VAR.
Target for evaluation: ipca_headline_mom.

Design: expanding-window, recursive, out-of-sample — identical protocol to the baselines.
    - First forecast origin: 2011-01.
    - Horizons h = 1..12; for origin t and horizon h the target date is t + h months.
    - At each origin only observations up to and including t are used (no look-ahead).
    - A (origin, horizon) pair is produced only if its target date <= 2018-12.
    - Expected counts: h=1 -> 95 forecasts, declining by 1 per horizon, h=12 -> 84.

Outputs:
    outputs/forecasts/var_forecasts.csv
    outputs/tables/var_mse_by_horizon.csv
    outputs/forecasts/traditional_forecasts_so_far.csv        (RW, RW_AO, AR1, VAR1)
    outputs/tables/traditional_mse_so_far_by_horizon.csv
    outputs/figures/traditional_mse_so_far_by_horizon.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Make `models` importable when run as `python src/run_var.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.var_model import VAR_VARS, var1_forecast  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "traditional_models_clean_data.csv"

VAR_FORECAST_FILE = ROOT / "outputs" / "forecasts" / "var_forecasts.csv"
VAR_MSE_FILE = ROOT / "outputs" / "tables" / "var_mse_by_horizon.csv"

BASELINE_FORECAST_FILE = ROOT / "outputs" / "forecasts" / "baseline_forecasts.csv"
COMBINED_FORECAST_FILE = ROOT / "outputs" / "forecasts" / "traditional_forecasts_so_far.csv"
COMBINED_MSE_FILE = ROOT / "outputs" / "tables" / "traditional_mse_so_far_by_horizon.csv"
COMBINED_FIG_FILE = ROOT / "outputs" / "figures" / "traditional_mse_so_far_by_horizon.png"

TARGET = "ipca_headline_mom"
MODEL_NAME = "VAR1"
FIRST_ORIGIN = pd.Period("2011-01", freq="M")
LAST_TARGET = pd.Period("2018-12", freq="M")
HORIZONS = list(range(1, 13))
# Model display order for the combined outputs.
ALL_MODELS = ["RW", "RW_AO", "AR1", "VAR1"]

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


def load_data() -> pd.DataFrame:
    """Load the clean dataset indexed by monthly Period (keeps target + VAR variables)."""
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df["period"] = df["date"].dt.to_period("M")
    df = df.set_index("period").sort_index()
    return df


def run_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the expanding-window recursive OOS exercise for VAR(1)."""
    target = df[TARGET].dropna()
    origins = [p for p in target.index if p >= FIRST_ORIGIN and (p + 1) <= LAST_TARGET]

    records: list[dict] = []
    for t in origins:
        train = df.loc[:t, VAR_VARS]  # only rows up to and including the origin
        horizons = [h for h in HORIZONS if (t + h) <= LAST_TARGET]

        preds = var1_forecast(train, horizons)

        for h in horizons:
            target_date = t + h
            actual = float(target.loc[target_date])
            forecast = preds[h]["headline"]
            error = actual - forecast
            records.append(
                {
                    "forecast_origin": str(t),
                    "target_date": str(target_date),
                    "horizon": h,
                    "model": MODEL_NAME,
                    "forecast": forecast,
                    "actual": actual,
                    "error": error,
                    "squared_error": error**2,
                }
            )

    return pd.DataFrame.from_records(records, columns=FORECAST_COLUMNS)


def build_mse_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MSE and forecast count by (horizon, model)."""
    return (
        forecasts.groupby(["horizon", "model"])
        .agg(mse=("squared_error", "mean"), n_forecasts=("squared_error", "size"))
        .reset_index()
        .sort_values(["horizon", "model"])
        .reset_index(drop=True)
    )


def plot_mse(mse_table: pd.DataFrame) -> None:
    """Line plot of MSE vs horizon, one line per model (RW, RW_AO, AR1, VAR1)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ALL_MODELS:
        sub = mse_table[mse_table["model"] == model].sort_values("horizon")
        if sub.empty:
            continue
        ax.plot(sub["horizon"], sub["mse"], marker="o", label=model)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Mean squared error")
    ax.set_title(
        "Traditional models out-of-sample MSE by horizon\n(target: ipca_headline_mom)"
    )
    ax.set_xticks(HORIZONS)
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(COMBINED_FIG_FILE, dpi=150)
    plt.close(fig)


def validate(
    var_forecasts: pd.DataFrame,
    var_mse: pd.DataFrame,
    combined_forecasts: pd.DataFrame,
    combined_mse: pd.DataFrame,
) -> bool:
    """Print validation diagnostics and return True if all hard checks pass."""
    print("=" * 70)
    print("VALIDATION — VAR(1)")
    print("=" * 70)

    counts = var_forecasts.groupby("horizon").size()
    print("\n[1] VAR1 forecast counts by horizon:")
    print(counts.to_string())

    print("\n[2] Missing values in var_forecasts.csv:")
    print(var_forecasts.isna().sum().to_string())

    print("\n[3] First 10 VAR forecast rows:")
    print(var_forecasts.head(10).to_string(index=False))

    print("\n[4] VAR MSE table:")
    print(var_mse.to_string(index=False))

    print("\n[5] Combined MSE table (RW, RW_AO, AR1, VAR1):")
    combined_wide = (
        combined_mse.pivot(index="horizon", columns="model", values="mse")
        .reindex(columns=ALL_MODELS)
    )
    print(combined_wide.to_string())

    print("\n[6] Hard checks:")
    all_ok = True

    n1 = int(counts.get(1, 0))
    n12 = int(counts.get(12, 0))
    ok1 = n1 == 95
    ok12 = n12 == 84
    all_ok &= ok1 and ok12
    print(f"    VAR1 h=1 -> {n1} (expect 95) {'OK' if ok1 else 'FAIL'}"
          f" | h=12 -> {n12} (expect 84) {'OK' if ok12 else 'FAIL'}")

    seq = [int(counts.get(h, 0)) for h in HORIZONS]
    decline_ok = seq == list(range(95, 83, -1))
    all_ok &= decline_ok
    print(f"    Counts decline 95->84 by 1 per horizon: {'OK' if decline_ok else 'FAIL'}")

    no_missing = int(var_forecasts.isna().sum().sum()) == 0
    all_ok &= no_missing
    print(f"    No missing values in var_forecasts: {'OK' if no_missing else 'FAIL'}")

    first_origin = min(var_forecasts["forecast_origin"])
    ok_origin = first_origin == str(FIRST_ORIGIN)
    all_ok &= ok_origin
    print(f"    First forecast origin -> {first_origin} (expect {FIRST_ORIGIN}) "
          f"{'OK' if ok_origin else 'FAIL'}")

    max_target = max(var_forecasts["target_date"])
    ok_target = max_target == str(LAST_TARGET)
    all_ok &= ok_target
    print(f"    Max target_date -> {max_target} (expect {LAST_TARGET}) "
          f"{'OK' if ok_target else 'FAIL'}")

    # Combined file should carry all four models.
    models_present = sorted(combined_forecasts["model"].unique())
    ok_models = set(ALL_MODELS).issubset(set(models_present))
    all_ok &= ok_models
    print(f"    Combined models present -> {models_present} {'OK' if ok_models else 'FAIL'}")

    print(f"\n    ALL CHECKS: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> None:
    df = load_data()
    var_forecasts = run_forecasts(df)
    var_mse = build_mse_table(var_forecasts)

    # Combine with the existing baseline forecasts (RW, RW_AO, AR1).
    if not BASELINE_FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"{BASELINE_FORECAST_FILE} not found — run `python src/run_baselines.py` first."
        )
    baseline_forecasts = pd.read_csv(BASELINE_FORECAST_FILE)
    combined_forecasts = pd.concat(
        [baseline_forecasts[FORECAST_COLUMNS], var_forecasts[FORECAST_COLUMNS]],
        ignore_index=True,
    )
    combined_mse = build_mse_table(combined_forecasts)

    for path in (
        VAR_FORECAST_FILE,
        VAR_MSE_FILE,
        COMBINED_FORECAST_FILE,
        COMBINED_MSE_FILE,
        COMBINED_FIG_FILE,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    var_forecasts.to_csv(VAR_FORECAST_FILE, index=False)
    var_mse.to_csv(VAR_MSE_FILE, index=False)
    combined_forecasts.to_csv(COMBINED_FORECAST_FILE, index=False)
    combined_mse.to_csv(COMBINED_MSE_FILE, index=False)
    plot_mse(combined_mse)

    ok = validate(var_forecasts, var_mse, combined_forecasts, combined_mse)

    print("\nWrote:")
    print(f"  {VAR_FORECAST_FILE.relative_to(ROOT)}  ({len(var_forecasts)} rows)")
    print(f"  {VAR_MSE_FILE.relative_to(ROOT)}  ({len(var_mse)} rows)")
    print(f"  {COMBINED_FORECAST_FILE.relative_to(ROOT)}  ({len(combined_forecasts)} rows)")
    print(f"  {COMBINED_MSE_FILE.relative_to(ROOT)}  ({len(combined_mse)} rows)")
    print(f"  {COMBINED_FIG_FILE.relative_to(ROOT)}")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
