"""Teddy, task T4: Random Forest (regression for y_pred, classification for p_up).

TODO: min_samples_leaf large (30 to 100) because daily returns are mostly noise; fixed seed
(src.eval.config.SEED); tune on validation only; export feature importances for T6.
"""
from src.models.base import Forecaster


class RandomForestForecaster(Forecaster):
    name = "random_forest"

    def fit(self, train, horizon):
        raise NotImplementedError("Teddy: implement T4 (random forest)")

    def predict(self, history, test):
        raise NotImplementedError
