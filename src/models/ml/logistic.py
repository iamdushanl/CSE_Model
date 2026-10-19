"""Teddy, task T4: Logistic Regression (linear ML benchmark) for direction.

Standardises features (scaler fit on TRAIN only), fits on rows with a direction label
(y_dir_h not NaN), and returns p_up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eval.config import SEED
from src.models.base import Forecaster

BASE_COLS = ["r", "l1", "l2", "l3", "l4", "m5", "m22", "v5", "v22", "abs_r"]


class LogisticForecaster(Forecaster):
    name = "logistic"
    features = ["zero_ret", "dow"]

    def __init__(self, C: float = 0.1, penalty: str = "l2"):
        self.C = C
        self.penalty = penalty
        self.feature_cols_: list[str] = []
        self.scaler_: StandardScaler | None = None
        self.model_: LogisticRegression | None = None
        self.default_p_up_: float = 0.5
        self.classes_: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, horizon: int) -> "LogisticForecaster":
        y_col = f"y_dir_{horizon}"
        valid = train.dropna(subset=[y_col]).copy()

        # Dynamic features: use BASE_COLS + EXTRA_FEATURES + PANEL_FEATURES (if present and valid)
        cols = [c for c in BASE_COLS + ["zero_ret", "dow"] if c in valid.columns]
        if "mkt" in valid.columns and valid["mkt"].notna().sum() > 0.5 * len(valid):
            cols.append("mkt")
        self.feature_cols_ = cols

        valid = valid.dropna(subset=self.feature_cols_)
        y = valid[y_col].to_numpy()

        if len(y) == 0 or len(np.unique(y)) < 2:
            self.default_p_up_ = float(np.mean(y)) if len(y) else 0.5
            self.model_ = None
            return self

        self.default_p_up_ = float(np.mean(y))
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(valid[self.feature_cols_])

        self.model_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
        )
        self.model_.fit(X_scaled, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, history: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None or self.scaler_ is None:
            return self.frame(test.index, p_up=np.full(len(test), self.default_p_up_))

        X = test[self.feature_cols_].copy()
        # Impute missing values with training feature means if any occur
        for i, col in enumerate(self.feature_cols_):
            if X[col].isna().any():
                X[col] = X[col].fillna(self.scaler_.mean_[i])

        X_scaled = self.scaler_.transform(X)
        proba = self.model_.predict_proba(X_scaled)

        if 1.0 in self.classes_:
            up_idx = int(np.where(self.classes_ == 1.0)[0][0])
            p_up = proba[:, up_idx]
        else:
            p_up = np.full(len(test), self.default_p_up_)

        return self.frame(test.index, p_up=p_up)
