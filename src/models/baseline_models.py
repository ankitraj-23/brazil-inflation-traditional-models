"""Baseline forecasting models for Phase 2 (traditional benchmarks).

Scope: Random Walk (RW), Random Walk Atkeson-Ohanian (RW_AO), and AR(1) only.
VAR and the Phillips Curve are intentionally NOT implemented here.

Every model is a pure function of a *training* array `y_train` — the target series in
chronological order up to and including the forecast origin `t` (so `y_train[-1] == y_t`).
No function ever looks past the origin, so these are safe for a recursive, expanding-window
out-of-sample exercise. Multi-step forecasts are computed analytically (no simulation).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# Below this distance from 1.0 the AR(1) geometric sum sum_{i=0}^{h-1} beta^i is treated
# as the removable-singularity limit (= h) instead of (1 - beta**h) / (1 - beta).
_BETA_UNIT_TOL = 1e-8

RW_AO_WINDOW = 48


def rw_forecast(y_train: np.ndarray, horizons: Iterable[int]) -> dict[int, float]:
    """Random Walk: the forecast for every horizon is the last observed value y_t."""
    y = np.asarray(y_train, dtype=float)
    if y.size == 0:
        raise ValueError("rw_forecast: empty training array")
    y_t = float(y[-1])
    return {int(h): y_t for h in horizons}


def rw_ao_forecast(
    y_train: np.ndarray, horizons: Iterable[int], window: int = RW_AO_WINDOW
) -> dict[int, float]:
    """Atkeson-Ohanian variant: forecast = mean of the last `window` observations.

    The same flat forecast is used for all horizons. Raises if fewer than `window`
    observations are available (never silently shrinks the window).
    """
    y = np.asarray(y_train, dtype=float)
    if y.size < window:
        raise ValueError(
            f"rw_ao_forecast: need at least {window} observations, got {y.size}"
        )
    mean_val = float(np.mean(y[-window:]))
    return {int(h): mean_val for h in horizons}


def ar1_fit(y_train: np.ndarray) -> tuple[float, float]:
    """OLS fit of y_t = alpha + beta * y_{t-1} on the training data only.

    Returns (alpha, beta). Uses consecutive-observation pairs; assumes `y_train` is a
    contiguous monthly series (true for the cleaned IPCA sample).
    """
    y = np.asarray(y_train, dtype=float)
    if y.size < 3:
        raise ValueError(f"ar1_fit: need at least 3 observations, got {y.size}")

    response = y[1:]
    predictor = y[:-1]
    x_bar = predictor.mean()
    y_bar = response.mean()
    var_x = np.sum((predictor - x_bar) ** 2)
    if var_x == 0.0:
        raise ValueError("ar1_fit: zero variance in predictor; cannot fit AR(1)")
    beta = float(np.sum((predictor - x_bar) * (response - y_bar)) / var_x)
    alpha = float(y_bar - beta * x_bar)
    return alpha, beta


def ar1_forecast(
    alpha: float, beta: float, y_t: float, horizons: Iterable[int]
) -> dict[int, float]:
    """Recursive h-step AR(1) forecast in closed form.

        yhat(t+h) = beta**h * y_t + alpha * sum_{i=0}^{h-1} beta**i

    The geometric sum is evaluated as (1 - beta**h) / (1 - beta), except when beta is
    within `_BETA_UNIT_TOL` of 1, where the limit is exactly h (avoids a 0/0 blow-up).
    """
    out: dict[int, float] = {}
    for h in horizons:
        h = int(h)
        if abs(1.0 - beta) < _BETA_UNIT_TOL:
            geom = float(h)
        else:
            geom = (1.0 - beta**h) / (1.0 - beta)
        out[h] = float(beta**h * y_t + alpha * geom)
    return out
