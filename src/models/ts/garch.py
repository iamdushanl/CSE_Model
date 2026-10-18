"""Chamathka, task C4: AR(1)-GJR-GARCH(1,1) with Student-t errors.

TODO:
  * Use the `arch` package: arch_model(r, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t").
  * Scale: the feature table already holds returns in PERCENT (good for the optimiser).
  * fit() on training returns of ONE entity (the market proxy); predict() filters through
    history + test with the fitted parameters and returns y_pred (conditional mean) and
    p_up = P(r > 0) under the fitted Student-t distribution.
  * Log convergence status and parameters at every yearly refit (needed for C5 parameter drift).
Register the class in src/models/registry.py.
"""
from src.models.base import Forecaster


class GarchForecaster(Forecaster):
    name = "ar_gjr_garch_t"

    def fit(self, train, horizon):
        raise NotImplementedError("Chamathka: implement C4")

    def predict(self, history, test):
        raise NotImplementedError
