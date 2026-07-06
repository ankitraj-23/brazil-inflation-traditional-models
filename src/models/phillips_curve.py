"""Backward-looking Phillips Curve traditional benchmark.

Scope: the *backward-looking* Phillips Curve only. The hybrid Phillips Curve, Focus
expectations, factor models and ML models are intentionally NOT implemented here. The
random-walk / AR(1) baselines live in `baseline_models.py` and the VAR in `var_model.py`;
neither is touched.

Paper specification
-------------------
Market / non-regulated inflation is modelled by a horizon-specific ("direct") backward
regression:

    market_inflation(t+h) = a0
                          + a1 * market_inflation(t)
                          + a2 * imported_inflation(t)
                          + a3 * output_gap(t)
                          + error

Administered inflation is forecast separately with an AR(1) (paper: p=1, q=0), reusing the
closed-form AR(1) from `baseline_models`. Headline IPCA is reconstructed as

    headline = 0.75 * market_forecast + 0.25 * administered_forecast

Output gap
----------
At each forecast origin `t` the output gap is computed *recursively*: the Hodrick-Prescott
filter is applied to `ibc_br_sa` using only observations up to and including `t`, with
`lamb = 129600` (the standard monthly smoothing constant). The resulting HP *cycle* is used
as `output_gap` for every training predictor date `s` and for the current-origin predictor
`t`. The filter never sees data after `t`, so the exercise stays free of look-ahead.

Every function here is a pure function of training data up to and including the forecast
origin `t`; nothing looks past the origin, so the module is safe for a recursive,
expanding-window out-of-sample exercise.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from models.baseline_models import ar1_fit, ar1_forecast

# Column roles.
MARKET_COL = "ipca_non_regulated_mom"
ADMINISTERED_COL = "ipca_administered_mom"
IMPORTED_COL = "imported_inflation"
ACTIVITY_COL = "ibc_br_sa"

# HP-filter smoothing parameter for monthly data (Ravn-Uhlig / standard monthly value).
HP_LAMBDA = 129600

# Headline reconstruction weights (paper): 0.75 market + 0.25 administered.
MARKET_WEIGHT = 0.75
ADMINISTERED_WEIGHT = 0.25


def compute_output_gap(ibc: pd.Series) -> pd.Series:
    """HP-filter cycle of the activity index `ibc`, used as the output gap.

    `ibc` must contain only observations up to and including the forecast origin (the
    recursion is the caller's responsibility). Returns the cycle component (same index as
    `ibc`), using `lamb = HP_LAMBDA`.
    """
    ibc = ibc.dropna().astype(float)
    if ibc.size < 4:
        raise ValueError(
            f"compute_output_gap: need at least 4 observations, got {ibc.size}"
        )
    cycle, _trend = hpfilter(ibc, lamb=HP_LAMBDA)
    return cycle


def pc_market_forecast(
    market: pd.Series,
    imported: pd.Series,
    output_gap: pd.Series,
    t: pd.Period,
    horizons: Iterable[int],
) -> dict[int, dict[str, float]]:
    """Direct h-step backward Phillips-Curve forecasts of market inflation.

    Parameters
    ----------
    market, imported, output_gap:
        Series indexed by monthly ``Period``, all restricted to observations up to and
        including the forecast origin ``t`` (``output_gap`` is the recursively computed HP
        cycle for this origin).
    t:
        Forecast origin period.
    horizons:
        Horizons to forecast.

    For each horizon ``h`` the regression is fit on all predictor dates ``s`` whose target
    date ``s + h`` satisfies ``s + h <= t`` (no training on post-origin targets), then
    applied to the current-origin predictor row at ``t``.

    Returns a dict mapping each ``h`` to ``{"forecast", "n_train", "max_train_target"}``
    where ``max_train_target`` is the latest training target date used (for no-look-ahead
    auditing; ``None`` if the regression had no usable pairs).
    """
    horizons = [int(h) for h in horizons]
    if not horizons:
        return {}

    # Predictor rows available at all (need every regressor present); drops the 2004-01
    # NaN in imported_inflation.
    preds = (
        pd.DataFrame({"market": market, "imported": imported, "gap": output_gap})
        .dropna()
        .sort_index()
    )

    # Current-origin predictor row (must be fully available at t).
    x_t = np.array(
        [1.0, float(market.loc[t]), float(imported.loc[t]), float(output_gap.loc[t])],
        dtype=float,
    )

    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        rows: list[list[float]] = []
        targets: list[float] = []
        target_dates: list[pd.Period] = []
        for s in preds.index:
            target_date = s + h
            # No-look-ahead: only train on pairs whose target date is at or before t.
            if target_date > t:
                continue
            if target_date not in market.index:
                continue
            y_val = market.loc[target_date]
            if pd.isna(y_val):
                continue
            row = preds.loc[s]
            rows.append([1.0, float(row["market"]), float(row["imported"]), float(row["gap"])])
            targets.append(float(y_val))
            target_dates.append(target_date)

        if len(targets) < 4:
            raise ValueError(
                f"pc_market_forecast: too few training pairs ({len(targets)}) for "
                f"horizon {h} at origin {t}"
            )

        X = np.asarray(rows, dtype=float)
        y = np.asarray(targets, dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        forecast = float(x_t @ beta)

        out[h] = {
            "forecast": forecast,
            "n_train": len(targets),
            "max_train_target": max(target_dates),
        }
    return out


def pc_administered_forecast(
    administered: pd.Series, horizons: Iterable[int]
) -> dict[int, float]:
    """AR(1) forecast of administered inflation (paper: p=1, q=0).

    `administered` is the series up to and including the origin (chronological). Reuses the
    closed-form AR(1) from `baseline_models` (fit fresh on the expanding window).
    """
    y_train = administered.dropna().to_numpy(dtype=float)
    alpha, beta = ar1_fit(y_train)
    return ar1_forecast(alpha, beta, float(y_train[-1]), horizons)


def pc_backward_forecast(
    market: pd.Series,
    imported: pd.Series,
    output_gap: pd.Series,
    administered: pd.Series,
    t: pd.Period,
    horizons: Iterable[int],
) -> dict[int, dict[str, float]]:
    """Full backward Phillips-Curve headline forecast at origin `t`.

    Combines the direct market regression and the administered AR(1) into the headline via
    the 0.75 / 0.25 weights. Returns a dict mapping each ``h`` to
    ``{"market", "administered", "headline", "n_train", "max_train_target"}``.
    """
    horizons = [int(h) for h in horizons]
    market_out = pc_market_forecast(market, imported, output_gap, t, horizons)
    admin_out = pc_administered_forecast(administered, horizons)

    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        m = market_out[h]
        market_f = m["forecast"]
        admin_f = admin_out[h]
        headline = MARKET_WEIGHT * market_f + ADMINISTERED_WEIGHT * admin_f
        out[h] = {
            "market": market_f,
            "administered": admin_f,
            "headline": headline,
            "n_train": m["n_train"],
            "max_train_target": m["max_train_target"],
        }
    return out
