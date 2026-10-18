"""Teddy, task T4: XGBoost (regression and classification).

TODO: shallow trees (max_depth 2 to 4), small learning rate, early stopping on the validation
period only; fixed seed. Keep the fitted booster per refit year for SHAP (T6) and for the
stability audit (T7).
"""
from src.models.base import Forecaster


class XGBoostForecaster(Forecaster):
    name = "xgboost"

    def fit(self, train, horizon):
        raise NotImplementedError("Teddy: implement T4 (xgboost)")

    def predict(self, history, test):
        raise NotImplementedError
