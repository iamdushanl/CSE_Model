"""TEMPLATE: copy this file to start a model. It is a working AR-style regression.

It shows the full contract: fit() reads train[y_ret_h], predict() returns y_pred.
Replace it with the real models; delete it from the registry when you do.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.base import Forecaster

LAGS = ["r", "l1", "l2", "l3", "l4"]


class TemplateAR5(Forecaster):
    """Linear regression of the h-day return on today's and 4 lagged returns."""
    name = "template_ar5"

    def fit(self, train, horizon):
        self.m_ = LinearRegression().fit(train[LAGS], train[f"y_ret_{horizon}"])
        return self

    def predict(self, history, test):
        return self.frame(test.index, y_pred=self.m_.predict(test[LAGS]))
