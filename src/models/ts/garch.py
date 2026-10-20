"""Market AR(1)-GJR-GARCH(1,1), standardized Student-t innovations.

A fixed yearly fit is filtered with observed origin-date returns. Forecasts
never incorporate future realized returns. Five-day mean is recursively summed.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import t as student_t
from src.models.base import Forecaster


class GarchForecaster(Forecaster):
    name = 'ar_gjr_garch_t'
    features = []

    def fit(self, train, horizon):
        from arch import arch_model
        if horizon not in (1, 5):
            raise ValueError('horizon must be 1 or 5')
        if train['entity'].nunique() != 1 or train['entity'].iloc[0] != 'MARKET':
            raise ValueError('market-only forecaster')
        self.horizon = horizon
        self.segment_ = train['seg'].iloc[-1]
        self.train_end_ = train.index.max()
        x = train.loc[train.seg == self.segment_, 'r'].dropna().to_numpy(dtype=float)
        if len(x) < 100:
            raise ValueError('insufficient within-segment training data')
        self.train_returns_ = x
        self.spec_ = dict(mean='AR', lags=1, vol='GARCH', p=1, o=1, q=1, dist='t', rescale=False)
        self.result_ = arch_model(x, **self.spec_).fit(disp='off')
        self.params_ = self.result_.params.copy()
        self.convergence_ = int(self.result_.convergence_flag)
        self.fit_record_ = dict(horizon=horizon, fit_year=int(self.train_end_.year)+1,
                                n_train=len(x), convergence=self.convergence_,
                                **{str(k): float(v) for k,v in self.params_.items()})
        return self

    def predict(self, history, test):
        from arch import arch_model
        if (test['entity'] != 'MARKET').any():
            raise ValueError('market-only forecaster')
        # The evaluator purges h labels, not observations. Reconstruct the
        # intervening observed returns from history when they are supplied.
        seen = list(self.train_returns_)
        bridge = history.loc[(history.index > self.train_end_) &
                             (history.index < test.index.min()) &
                             (history.seg == self.segment_), 'r'].dropna()
        seen.extend(bridge.to_numpy(dtype=float))
        means, probs = [], []
        nu = float(self.params_['nu'])
        if nu <= 2:
            raise ValueError('Student-t df must exceed 2')
        for _, row in test.iterrows():
            if row['seg'] != self.segment_ or not np.isfinite(row['r']):
                means.append(np.nan); probs.append(np.nan)
                continue
            seen.append(float(row['r']))  # r_t is known at forecast origin t
            fixed = arch_model(np.asarray(seen), **self.spec_).fix(self.params_)
            forecast = fixed.forecast(horizon=self.horizon, method='analytic', reindex=False)
            mu = np.asarray(forecast.mean.iloc[-1], dtype=float)
            variances = np.asarray(forecast.variance.iloc[-1], dtype=float)
            total_mean = float(mu.sum())
            # Sum of marginal variances ignores cross-horizon covariance.
            # Use this as an explicitly approximate directional probability.
            approx_var = max(float(variances.sum()), 1e-12)
            scale = np.sqrt(approx_var * (nu-2)/nu)
            means.append(total_mean)
            probs.append(float(student_t.cdf(total_mean/scale, df=nu)))
        return self.frame(test.index, means, probs)
