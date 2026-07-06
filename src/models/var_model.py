"""VAR(1) traditional benchmark for Phase 2 (Brazilian inflation forecasting).

Scope: first-order vector autoregression only. The random-walk / AR(1) baselines live in
`baseline_models.py` and are not touched here.

Paper specification
-------------------
The VAR is estimated over four endogenous variables (fixed order below):

    1. market / non-regulated IPCA inflation   (`ipca_non_regulated_mom`)
    2. administered IPCA inflation              (`ipca_administered_mom`)
    3. M4 first difference                      (`m4_diff`)
    4. nominal exchange-rate first difference   (`exchange_rate_diff`)

One lag, with a constant. Headline IPCA is **not** modelled inside the VAR: it is
reconstructed from the two inflation components as

    headline = 0.75 * market_inflation + 0.25 * administered_inflation

The function is a pure function of a *training* DataFrame containing only observations up
to and including the forecast origin `t`, so it never looks past the origin and is safe for
a recursive, expanding-window out-of-sample exercise.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from statsmodels.tsa.api import VAR

# Endogenous variables in their fixed estimation order.
MARKET_COL = "ipca_non_regulated_mom"
ADMINISTERED_COL = "ipca_administered_mom"
VAR_VARS: list[str] = [
    MARKET_COL,
    ADMINISTERED_COL,
    "m4_diff",
    "exchange_rate_diff",
]

# Headline reconstruction weights (paper): 0.75 market + 0.25 administered.
MARKET_WEIGHT = 0.75
ADMINISTERED_WEIGHT = 0.25


def var1_forecast(
    train: pd.DataFrame, horizons: Iterable[int]
) -> dict[int, dict[str, float]]:
    """Fit VAR(1) with a constant on `train` and forecast the requested horizons.

    Parameters
    ----------
    train:
        DataFrame whose rows are the training observations up to and including the forecast
        origin. Must contain the four `VAR_VARS` columns; rows with any NaN among those
        columns are dropped before fitting (never looks past the origin).
    horizons:
        Horizons to return. The model forecasts out to ``max(horizons)`` steps and the
        h-th step is picked for each requested ``h``.

    Returns
    -------
    dict mapping each horizon ``h`` to a dict with keys ``market``, ``administered`` and
    ``headline`` (the 0.75/0.25 combination of the first two).
    """
    horizons = [int(h) for h in horizons]
    if not horizons:
        return {}

    endog = train.loc[:, VAR_VARS].dropna()
    if endog.shape[0] < len(VAR_VARS) + 2:
        raise ValueError(
            f"var1_forecast: too few clean rows ({endog.shape[0]}) to fit VAR(1) "
            f"on {len(VAR_VARS)} variables"
        )

    model = VAR(endog.to_numpy(dtype=float))
    results = model.fit(1, trend="c")  # VAR(1), constant; no auto lag selection.

    steps = max(horizons)
    # forecast() needs the last k_ar (=1) observations as the initial condition.
    fc = results.forecast(endog.to_numpy(dtype=float)[-results.k_ar :], steps=steps)

    market_idx = VAR_VARS.index(MARKET_COL)
    admin_idx = VAR_VARS.index(ADMINISTERED_COL)

    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        market = float(fc[h - 1, market_idx])
        administered = float(fc[h - 1, admin_idx])
        headline = MARKET_WEIGHT * market + ADMINISTERED_WEIGHT * administered
        out[h] = {
            "market": market,
            "administered": administered,
            "headline": headline,
        }
    return out
