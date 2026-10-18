"""Teddy, task T4: Logistic Regression (linear ML benchmark) for direction.

TODO: standardise features (scaler fit on TRAIN only), fit on rows with a direction label
(y_dir_h not NaN), return p_up. Declare `features = ["mkt", "zero_ret"]` for the pooled panel.
Tune C on the validation window only. Register in src/models/registry.py.
"""
from src.models.base import Forecaster


class LogisticForecaster(Forecaster):
    name = "logistic"

    def fit(self, train, horizon):
        raise NotImplementedError("Teddy: implement T4 (logistic)")

    def predict(self, history, test):
        raise NotImplementedError
