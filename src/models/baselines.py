"""Baselines.  Owner: Chamathka (task C1).

Every real model must beat these. They are deliberately simple.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster


class ZeroForecaster(Forecaster):
    """Random walk: the h-day return is forecast to be 0. Direction is undefined."""
    name = "zero"

    def predict(self, history, test):
        return self.frame(test.index, y_pred=np.zeros(len(test)))


class MeanForecaster(Forecaster):
    """Historical mean of the training returns; direction follows the training up-share."""
    name = "hist_mean"

    def fit(self, train, horizon):
        self.mean_ = float(train[f"y_ret_{horizon}"].mean())
        d = train[f"y_dir_{horizon}"].dropna()
        self.p_up_ = float(d.mean()) if len(d) else 0.5
        return self

    def predict(self, history, test):
        n = len(test)
        return self.frame(test.index, y_pred=np.full(n, self.mean_), p_up=np.full(n, self.p_up_))


class PersistenceForecaster(Forecaster):
    """Tomorrow repeats today: h=1 uses r(t); h>1 uses the recent 5-day mean times h."""
    name = "persistence"

    def fit(self, train, horizon):
        self.h_ = horizon
        return self

    def predict(self, history, test):
        base = test["r"] if self.h_ == 1 else test["m5"] * self.h_
        return self.frame(test.index, y_pred=base.to_numpy())
