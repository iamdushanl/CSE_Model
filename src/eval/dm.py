"""Diebold-Mariano test.  Owner: Teddy."""
from __future__ import annotations

import numpy as np
from scipy import stats


def dm_test(loss_a, loss_b, h: int = 1) -> tuple[float, float]:
    """Two-sided DM test of equal predictive accuracy (Harvey-Leybourne-Newbold form).

    loss_a, loss_b: per-period losses of model A and model B (same length, same dates).
    h: forecast horizon. Overlapping h-day targets get a Bartlett long-run variance
       with h-1 lags.
    Returns (statistic, p_value). A NEGATIVE statistic means model A has the LOWER loss.
    For a panel, average the losses across entities per date before calling this.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return float("nan"), float("nan")
    dbar = d.mean()
    lrv = np.mean((d - dbar) ** 2)
    for k in range(1, h):
        w = 1.0 - k / h
        lrv += 2.0 * w * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    lrv = max(lrv, 1e-18)
    stat = dbar / np.sqrt(lrv / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)       # small-sample correction
    p = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)
