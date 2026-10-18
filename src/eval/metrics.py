"""Forecast metrics.  Owner: Teddy."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score


def rmse(y, f) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(f)) ** 2)))


def mae(y, f) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f))))


def oos_r2(y, f) -> float:
    """Out-of-sample R-squared against the zero (random-walk) forecast.

    0 = no better than predicting no change; negative = worse.
    """
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(1.0 - np.sum((y - f) ** 2) / np.sum(y ** 2))


def qlike(y_var, f_var) -> float:
    """QLIKE loss for variance forecasts (used for the optional volatility target)."""
    r = np.asarray(y_var, float) / np.asarray(f_var, float)
    return float(np.mean(r - np.log(r) - 1.0))


def direction_prediction(df: pd.DataFrame) -> pd.Series:
    """Predicted direction (1 up, 0 down, NaN undefined) from p_up or the sign of y_pred."""
    from_p = (df["p_up"] > 0.5).astype(float).where(df["p_up"].notna())
    s = np.sign(df["y_pred"])
    from_y = (s > 0).astype(float).where(df["y_pred"].notna() & (s != 0))
    return from_p.fillna(from_y)


def summarise(pred: pd.DataFrame) -> dict:
    """Metrics for ONE model / horizon / window / entity set prediction frame."""
    out = {"n": int(len(pred))}
    reg = pred.dropna(subset=["y_ret", "y_pred"])
    if len(reg):
        out.update(rmse=rmse(reg.y_ret, reg.y_pred), mae=mae(reg.y_ret, reg.y_pred),
                   oos_r2=oos_r2(reg.y_ret, reg.y_pred))
    d = pred.dropna(subset=["y_dir"]).copy()
    d["dir_pred"] = direction_prediction(d)
    d = d.dropna(subset=["dir_pred"])
    if len(d):
        out["n_dir"] = int(len(d))
        out["accuracy"] = float((d.dir_pred == d.y_dir).mean())
        out["base_rate_up"] = float(d.y_dir.mean())
        out["balanced_acc"] = float(balanced_accuracy_score(d.y_dir, d.dir_pred))
        p = d.dropna(subset=["p_up"])
        if len(p) and p.y_dir.nunique() == 2:
            out["auc"] = float(roc_auc_score(p.y_dir, p.p_up))
            out["logloss"] = float(log_loss(p.y_dir, p.p_up.clip(1e-4, 1 - 1e-4)))
    return out
