"""Chamathka, tasks C2 and C3: AR(p) and ARMA(p,q) forecasters.

TODO (see handbook section 6):
  * C2  choose p and q by AIC and BIC on TRAINING data only (ACF/PACF first), p, q <= 5.
  * C3  walk-forward AR(p) and ARMA(p,q); h=1 and h=5. For h=5 iterate the model or
        cumulate the multi-step forecasts; do NOT repeat the 1-day forecast five times.
  * Output y_pred (percent) and p_up = P(h-day return > 0) from the forecast and the
    residual standard deviation (normal CDF) so direction metrics work.
  * ARIMA(p,1,q) on log price equals ARMA(p,q) on returns. ADF says returns are
    stationary, so do not difference them again.

Use statsmodels (statsmodels.tsa.arima.model.ARIMA). Warm up with `history["r"]` and
filter through `test["r"]` day by day with the FITTED parameters (use `.apply()` or
`.append(refit=False)`), so each forecast uses only returns up to that day.
Register your classes in src/models/registry.py.
"""
from src.models.base import Forecaster


class ARForecaster(Forecaster):
    name = "ar"

    def fit(self, train, horizon):
        raise NotImplementedError("Chamathka: implement C3 (AR)")

    def predict(self, history, test):
        raise NotImplementedError


class ARMAForecaster(Forecaster):
    name = "arma"

    def fit(self, train, horizon):
        raise NotImplementedError("Chamathka: implement C3 (ARMA)")

    def predict(self, history, test):
        raise NotImplementedError
