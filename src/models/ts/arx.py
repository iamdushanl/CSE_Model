"""Stock-specific ARX(1) with same-day market return, estimated on past targets."""
import numpy as np
from scipy.stats import norm
from src.models.base import Forecaster


class StockARXForecaster(Forecaster):
    name = 'stock_arx'
    features = ['mkt']

    def fit(self, train, horizon):
        self.horizon = horizon
        self.params_ = {}
        for entity, g in train.groupby('entity'):
            if entity == 'MARKET' or len(g) < 30:
                continue
            x = np.column_stack([np.ones(len(g)), g['r'], g['mkt']])
            y = g[f'y_ret_{horizon}'].to_numpy(dtype=float)
            valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
            if valid.sum() < 30:
                continue
            coef = np.linalg.lstsq(x[valid], y[valid], rcond=None)[0]
            residual = y[valid] - x[valid] @ coef
            sd = max(float(np.std(residual, ddof=3)), 1e-8)
            self.params_[entity] = (coef, sd)
        return self

    def predict(self, history, test):
        means, probs = [], []
        for _, row in test.iterrows():
            entry = self.params_.get(row['entity'])
            if entry is None:
                means.append(np.nan); probs.append(np.nan)
                continue
            coef, sd = entry
            mean = float(np.dot([1, row['r'], row['mkt']], coef))
            means.append(mean)
            probs.append(float(norm.cdf(mean / sd)))
        return self.frame(test.index, means, probs)
