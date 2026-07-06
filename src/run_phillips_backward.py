"""Run the backward-looking Phillips Curve out-of-sample forecasting exercise.

Model: backward-looking Phillips Curve (see src/models/phillips_curve.py). Market /
non-regulated inflation is modelled by a horizon-specific ("direct") backward regression
on its own lag, imported inflation and a recursively HP-filtered output gap; administered
inflation is forecast separately with an AR(1) (p=1, q=0). Headline IPCA is reconstructed
as 0.75 * market + 0.25 * administered; it is never modelled directly.
Target for evaluation: ipca_headline_mom.

Design: expanding-window, recursive, out-of-sample — identical protocol to the baselines
and the VAR.
    - First forecast origin: 2011-01.
    - Horizons h = 1..12; for origin t and horizon h the target date is t + h months.
    - At each origin only observations up to and including t are used (no look-ahead); the
      output gap is recomputed with the HP filter on ibc_br_sa up to t only.
    - For each horizon h the Phillips-Curve training pairs satisfy predictor_date + h <= t.
    - A (origin, horizon) pair is produced only if its target date <= 2018-12.
    - Expected counts: h=1 -> 95 forecasts, declining by 1 per horizon, h=12 -> 84.

Outputs:
    outputs/forecasts/pc_backward_forecasts.csv
    outputs/tables/pc_backward_mse_by_horizon.csv
    outputs/forecasts/traditional_forecasts_so_far.csv   (RW, RW_AO, AR1, VAR1, PC_BACKWARD)
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

# Make `models` importable when run as `python src/run_phillips_backward.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.phillips_curve import (  # noqa: E402
    ACTIVITY_COL,
    ADMINISTERED_COL,
    IMPORTED_COL,
    MARKET_COL,
    compute_output_gap,
    pc_backward_forecast,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "traditional_models_clean_data.csv"

PC_FORECAST_FILE = ROOT / "outputs" / "forecasts" / "pc_backward_forecasts.csv"
PC_MSE_FILE = ROOT / "outputs" / "tables" / "pc_backward_mse_by_horizon.csv"

BASELINE_FORECAST_FILE = ROOT / "outputs" / "forecasts" / "baseline_forecasts.csv"
VAR_FORECAST_FILE = ROOT / "outputs" / "forecasts" / "var_forecasts.csv"
COMBINED_FORECAST_FILE = ROOT / "outputs" / "forecasts" / "traditional_forecasts_so_far.csv"
COMBINED_MSE_FILE = ROOT / "outputs" / "tables" / "traditional_mse_so_far_by_horizon.csv"
COMBINED_FIG_FILE = ROOT / "outputs" / "figures" / "traditional_mse_so_far_by_horizon.png"

TARGET = "ipca_headline_mom"
MODEL_NAME = "PC_BACKWARD"
FIRST_ORIGIN = pd.Period("2011-01", freq="M")
LAST_TARGET = pd.Period("2018-12", freq="M")
HORIZONS = list(range(1, 13))
# Model display order for the combined outputs.
ALL_MODELS = ["RW", "RW_AO", "AR1", "VAR1", "PC_BACKWARD"]

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
    """Load the clean dataset indexed by monthly Period."""
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df["period"] = df["date"].dt.to_period("M")
    return df.set_index("period").sort_index()


def run_forecasts(df: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.Period]]:
    """Execute the expanding-window recursive OOS exercise for the backward PC.

    Returns the forecast records and a list of every training target date used across all
    PC market regressions violating the no-look-ahead rule (should be empty).
    """
    target = df[TARGET].dropna()
    market = df[MARKET_COL]
    imported = df[IMPORTED_COL]
    administered = df[ADMINISTERED_COL]
    activity = df[ACTIVITY_COL]

    origins = [p for p in target.index if p >= FIRST_ORIGIN and (p + 1) <= LAST_TARGET]

    records: list[dict] = []
    min_train_rows = None
    lookahead_violations: list[pd.Period] = []

    for t in origins:
        horizons = [h for h in HORIZONS if (t + h) <= LAST_TARGET]

        # Recursive output gap: HP cycle on ibc_br_sa using only data up to and including t.
        output_gap = compute_output_gap(activity.loc[:t])

        preds = pc_backward_forecast(
            market=market.loc[:t],
            imported=imported.loc[:t],
            output_gap=output_gap,
            administered=administered.loc[:t],
            t=t,
            horizons=horizons,
        )

        for h in horizons:
            info = preds[h]
            # No-look-ahead audit: latest training target must not exceed the origin.
            if info["max_train_target"] is not None and info["max_train_target"] > t:
                lookahead_violations.append(info["max_train_target"])
            if min_train_rows is None or info["n_train"] < min_train_rows:
                min_train_rows = info["n_train"]

            target_date = t + h
            actual = float(target.loc[target_date])
            forecast = info["headline"]
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

    run_forecasts.min_train_rows = min_train_rows  # type: ignore[attr-defined]
    return pd.DataFrame.from_records(records, columns=FORECAST_COLUMNS), lookahead_violations


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
    """Line plot of MSE vs horizon, one line per model (all five)."""
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
    pc_forecasts: pd.DataFrame,
    pc_mse: pd.DataFrame,
    combined_forecasts: pd.DataFrame,
    combined_mse: pd.DataFrame,
    min_train_rows: int | None,
    lookahead_violations: list[pd.Period],
) -> bool:
    """Print validation diagnostics and return True if all hard checks pass."""
    print("=" * 70)
    print("VALIDATION — PC_BACKWARD (backward-looking Phillips Curve)")
    print("=" * 70)

    counts = pc_forecasts.groupby("horizon").size()
    print("\n[1] PC_BACKWARD forecast counts by horizon:")
    print(counts.to_string())

    print("\n[2] Missing values in pc_backward_forecasts.csv:")
    print(pc_forecasts.isna().sum().to_string())

    print("\n[3] First 10 PC_BACKWARD forecast rows:")
    print(pc_forecasts.head(10).to_string(index=False))

    print("\n[4] PC_BACKWARD MSE table:")
    print(pc_mse.to_string(index=False))

    print("\n[5] Combined MSE table (RW, RW_AO, AR1, VAR1, PC_BACKWARD):")
    combined_wide = combined_mse.pivot(
        index="horizon", columns="model", values="mse"
    ).reindex(columns=ALL_MODELS)
    print(combined_wide.to_string())

    print(f"\n[6] Minimum training rows across all PC market regressions: {min_train_rows}")

    print("\n[7] Hard checks:")
    all_ok = True

    n1 = int(counts.get(1, 0))
    n12 = int(counts.get(12, 0))
    ok1 = n1 == 95
    ok12 = n12 == 84
    all_ok &= ok1 and ok12
    print(f"    PC_BACKWARD h=1 -> {n1} (expect 95) {'OK' if ok1 else 'FAIL'}"
          f" | h=12 -> {n12} (expect 84) {'OK' if ok12 else 'FAIL'}")

    seq = [int(counts.get(h, 0)) for h in HORIZONS]
    decline_ok = seq == list(range(95, 83, -1))
    all_ok &= decline_ok
    print(f"    Counts decline 95->84 by 1 per horizon: {'OK' if decline_ok else 'FAIL'}")

    no_missing = int(pc_forecasts.isna().sum().sum()) == 0
    all_ok &= no_missing
    print(f"    No missing values in pc_backward_forecasts: {'OK' if no_missing else 'FAIL'}")

    first_origin = min(pc_forecasts["forecast_origin"])
    ok_origin = first_origin == str(FIRST_ORIGIN)
    all_ok &= ok_origin
    print(f"    First forecast origin -> {first_origin} (expect {FIRST_ORIGIN}) "
          f"{'OK' if ok_origin else 'FAIL'}")

    max_target = max(pc_forecasts["target_date"])
    ok_target = max_target == str(LAST_TARGET)
    all_ok &= ok_target
    print(f"    Max target_date -> {max_target} (expect {LAST_TARGET}) "
          f"{'OK' if ok_target else 'FAIL'}")

    # No PC training pair may have a target date after its forecast origin.
    no_lookahead = len(lookahead_violations) == 0
    all_ok &= no_lookahead
    print(f"    No PC training pair with target_date > forecast_origin: "
          f"{'OK' if no_lookahead else f'FAIL ({len(lookahead_violations)} violations)'}")

    # Combined file should carry all five models and the expected total row count.
    models_present = sorted(combined_forecasts["model"].unique())
    ok_models = set(ALL_MODELS).issubset(set(models_present))
    all_ok &= ok_models
    print(f"    Combined models present -> {models_present} {'OK' if ok_models else 'FAIL'}")

    n_combined = len(combined_forecasts)
    ok_count = n_combined == 5370
    all_ok &= ok_count
    print(f"    Combined forecast row count -> {n_combined} (expect 5370) "
          f"{'OK' if ok_count else 'FAIL'}")

    print(f"\n    ALL CHECKS: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> None:
    df = load_data()
    pc_forecasts, lookahead_violations = run_forecasts(df)
    min_train_rows = getattr(run_forecasts, "min_train_rows", None)
    pc_mse = build_mse_table(pc_forecasts)

    # Rebuild the combined file from the base per-model forecast files (idempotent: safe to
    # re-run without double-appending PC_BACKWARD).
    for base in (BASELINE_FORECAST_FILE, VAR_FORECAST_FILE):
        if not base.exists():
            raise FileNotFoundError(
                f"{base} not found — run `python src/run_baselines.py` and "
                f"`python src/run_var.py` first."
            )
    baseline_forecasts = pd.read_csv(BASELINE_FORECAST_FILE)
    var_forecasts = pd.read_csv(VAR_FORECAST_FILE)
    combined_forecasts = pd.concat(
        [
            baseline_forecasts[FORECAST_COLUMNS],
            var_forecasts[FORECAST_COLUMNS],
            pc_forecasts[FORECAST_COLUMNS],
        ],
        ignore_index=True,
    )
    combined_mse = build_mse_table(combined_forecasts)

    for path in (
        PC_FORECAST_FILE,
        PC_MSE_FILE,
        COMBINED_FORECAST_FILE,
        COMBINED_MSE_FILE,
        COMBINED_FIG_FILE,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    pc_forecasts.to_csv(PC_FORECAST_FILE, index=False)
    pc_mse.to_csv(PC_MSE_FILE, index=False)
    combined_forecasts.to_csv(COMBINED_FORECAST_FILE, index=False)
    combined_mse.to_csv(COMBINED_MSE_FILE, index=False)
    plot_mse(combined_mse)

    ok = validate(
        pc_forecasts,
        pc_mse,
        combined_forecasts,
        combined_mse,
        min_train_rows,
        lookahead_violations,
    )

    print("\nWrote:")
    print(f"  {PC_FORECAST_FILE.relative_to(ROOT)}  ({len(pc_forecasts)} rows)")
    print(f"  {PC_MSE_FILE.relative_to(ROOT)}  ({len(pc_mse)} rows)")
    print(f"  {COMBINED_FORECAST_FILE.relative_to(ROOT)}  ({len(combined_forecasts)} rows)")
    print(f"  {COMBINED_MSE_FILE.relative_to(ROOT)}  ({len(combined_mse)} rows)")
    print(f"  {COMBINED_FIG_FILE.relative_to(ROOT)}")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
