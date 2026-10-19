"""Teddy, task T4: Random Forest (regression for y_pred, classification for p_up).

min_samples_leaf is kept relatively large (30 to 100) to regularize against noisy daily returns.
Feature importances are exported for task T6 / interpretability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.eval.config import SEED
from src.models.base import Forecaster

BASE_COLS = ["r", "l1", "l2", "l3", "l4", "m5", "m22", "v5", "v22", "abs_r"]


class RandomForestForecaster(Forecaster):
    name = "random_forest"
    features = ["zero_ret", "dow"]

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        min_samples_leaf: int = 50,
        random_state: int = SEED,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.feature_cols_: list[str] = []
        self.train_medians_: pd.Series | None = None
        self.regressor_: RandomForestRegressor | None = None
        self.classifier_: RandomForestClassifier | None = None
        self.default_ret_: float = 0.0
        self.default_p_up_: float = 0.5
        self.classes_: np.ndarray | None = None
        self.feature_importances_: pd.Series | None = None

    def fit(self, train: pd.DataFrame, horizon: int) -> "RandomForestForecaster":
        ret_col = f"y_ret_{horizon}"
        dir_col = f"y_dir_{horizon}"

        cols = [c for c in BASE_COLS + ["zero_ret", "dow"] if c in train.columns]
        if "mkt" in train.columns and train["mkt"].notna().sum() > 0.5 * len(train):
            cols.append("mkt")
        self.feature_cols_ = cols

        # Training data for regression
        reg_data = train.dropna(subset=self.feature_cols_ + [ret_col])
        if len(reg_data) >= self.min_samples_leaf:
            X_reg = reg_data[self.feature_cols_]
            y_reg = reg_data[ret_col].to_numpy()
            self.train_medians_ = X_reg.median()
            self.default_ret_ = float(np.mean(y_reg))

            self.regressor_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.regressor_.fit(X_reg, y_reg)
            self.feature_importances_ = pd.Series(
                self.regressor_.feature_importances_, index=self.feature_cols_
            )
        else:
            self.regressor_ = None
            self.default_ret_ = float(train[ret_col].mean()) if train[ret_col].notna().any() else 0.0

        # Training data for classification
        clf_data = train.dropna(subset=self.feature_cols_ + [dir_col])
        y_clf = clf_data[dir_col].to_numpy()
        if len(clf_data) >= self.min_samples_leaf and len(np.unique(y_clf)) >= 2:
            X_clf = clf_data[self.feature_cols_]
            self.default_p_up_ = float(np.mean(y_clf))

            self.classifier_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.classifier_.fit(X_clf, y_clf)
            self.classes_ = self.classifier_.classes_
        else:
            self.classifier_ = None
            self.default_p_up_ = float(np.mean(y_clf)) if len(y_clf) else 0.5

        return self

    def predict(self, history: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        n = len(test)
        if n == 0:
            return self.frame(test.index)

        X = test[self.feature_cols_].copy()
        if self.train_medians_ is not None:
            X = X.fillna(self.train_medians_)

        # Predict returns
        if self.regressor_ is not None:
            y_pred = self.regressor_.predict(X)
        else:
            y_pred = np.full(n, self.default_ret_)

        # Predict direction probability
        if self.classifier_ is not None:
            proba = self.classifier_.predict_proba(X)
            if 1.0 in self.classes_:
                up_idx = int(np.where(self.classes_ == 1.0)[0][0])
                p_up = proba[:, up_idx]
            else:
                p_up = np.full(n, self.default_p_up_)
        else:
            p_up = np.full(n, self.default_p_up_)

        return self.frame(test.index, y_pred=y_pred, p_up=p_up)
