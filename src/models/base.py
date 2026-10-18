"""The model interface every forecaster implements (handbook section 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Forecaster:
    """Base class. Subclass it, set `name`, implement `fit` and `predict`.

    fit(train, horizon)       learn from `train` (rows up to the end of the training
                              period; contains features AND target columns y_ret_h,
                              y_dir_h, y_rv_5). Return self.
    predict(history, test)    return a DataFrame with the SAME length and order as
                              `test` and columns:
                                y_pred  forecast of the h-day return in percent
                                        (NaN if the model only classifies)
                                p_up    probability that the h-day return is positive
                                        (NaN if the model only forecasts a number)
    `test` never contains target columns. `history` is the training frame, so models
    that need a warm-up (ARMA, GARCH) can filter through it.
    """

    name = "base"
    # Extra columns (beyond the base features) the model reads, e.g. ["mkt", "zero_ret"].
    # The evaluator drops training and test rows where any declared column is missing.
    features: list[str] = []

    def fit(self, train: pd.DataFrame, horizon: int) -> "Forecaster":
        return self

    def predict(self, history: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def frame(index, y_pred=None, p_up=None) -> pd.DataFrame:
        n = len(index)
        return pd.DataFrame({
            "y_pred": np.full(n, np.nan) if y_pred is None else np.asarray(y_pred, float),
            "p_up": np.full(n, np.nan) if p_up is None else np.asarray(p_up, float),
        }, index=index)
