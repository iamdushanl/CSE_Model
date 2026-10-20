"""Leakage-aware AR and ARMA forecasters for percent daily log returns.

Orders are selected outside test windows; defaults are intentionally fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from src.models.base import Forecaster


def select_orders(train, max_p=5, max_q=5):
    """Training-only AIC/BIC grid; return ranked table, never use test targets."""
    if max_p > 5 or max_q > 5:
        raise ValueError('Candidate orders must not exceed five')
    x = train['r'].dropna().to_numpy(dtype=float)
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                fit = ARIMA(x, order=(p, 0, q), trend='c').fit()
                rows.append({'p': p, 'q': q, 'aic': fit.aic, 'bic': fit.bic})
            except (ValueError, np.linalg.LinAlgError):
                continue
    return pd.DataFrame(rows).sort_values(['bic', 'aic']).reset_index(drop=True)


class _ARIMAForecaster(Forecaster):
    order = (1, 0, 0)
    def __init__(self, p=None, q=None):
        self.p = self.order[0] if p is None else int(p)
        self.q = self.order[2] if q is None else int(q)
        if not (0 <= self.p <= 5 and 0 <= self.q <= 5):
            raise ValueError('p and q must be in [0,5]')

    def fit(self, train, horizon):
        self.horizon = int(horizon)
        if self.horizon not in (1, 5):
            raise ValueError('Only h=1 and h=5 are supported')
        if train['entity'].nunique() != 1:
            raise ValueError('AR/ARMA require a single entity per fit')
        self.entity_ = train['entity'].iloc[0]
        self.segment_ = train['seg'].iloc[-1]
        x = train.loc[train['seg'] == self.segment_, 'r'].dropna().to_numpy(dtype=float)
        if len(x) < max(30, 5 * (self.p + self.q + 1)):
            raise ValueError('Insufficient contiguous training history')
        self.fitted_ = ARIMA(x, order=(self.p, 0, self.q), trend='c').fit()
        self.sigma_ = max(float(np.sqrt(self.fitted_.params[self.fitted_.param_names.index('sigma2')])), 1e-8)
        self.last_date_ = train.index[-1]
        return self

    def predict(self, history, test):
        if (test['entity'] != self.entity_).any():
            raise ValueError('Mixed entities in single-entity forecaster')
        state = self.fitted_
        # Purging removes recent training targets, not observed returns: filter the
        # available history up to the first prediction date without re-estimating.
        recent = history[(history.index > self.last_date_) & (history.index < test.index[0])]
        if len(recent):
            state = state.append(recent['r'].to_numpy(dtype=float), refit=False)
        means, probs = [], []
        prev_seg = self.segment_
        for _, row in test.iterrows():
            if row['seg'] != prev_seg:
                # A segment break invalidates the old filtered state; do not bridge it.
                state = None
                prev_seg = row['seg']
            if state is None:
                means.append(np.nan); probs.append(np.nan)
                continue
            # At date t the observed r(t) is known, so append before forecasting t+1.
            state = state.append([float(row['r'])], refit=False)
            path = np.asarray(state.forecast(steps=self.horizon), float)
            mean = float(path.sum())
            # Approximate cumulative-return variance using forecast-error covariance
            # under the fitted ARMA model (not h times the one-day point forecast).
            from statsmodels.tsa.arima_process import arma2ma
            ar = np.r_[1., -np.asarray([state.params[state.param_names.index(f'ar.L{i}')] for i in range(1, self.p + 1)])]
            ma = np.r_[1., np.asarray([state.params[state.param_names.index(f'ma.L{i}')] for i in range(1, self.q + 1)])]
            psi = arma2ma(ar, ma, lags=self.horizon)
            weights = np.cumsum(psi[:self.horizon])
            sd = self.sigma_ * np.sqrt(np.sum(weights ** 2))
            means.append(mean)
            probs.append(float(norm.cdf(mean / max(sd, 1e-8))))
        return self.frame(test.index, means, probs)


class ARForecaster(_ARIMAForecaster):
    name = 'ar'
    order = (1, 0, 0)


class ARMAForecaster(_ARIMAForecaster):
    name = 'arma'
    order = (1, 0, 1)
