import numpy as np
import pandas as pd

from src.eval.metrics import summarise
from src.eval.walkforward import walk_forward
from src.features.build import build_entity_features
from tests.test_eval import WIN, _table
from src.models.ml.logistic import LogisticForecaster
from src.models.ml.random_forest import RandomForestForecaster
from src.models.ml.xgboost_model import XGBoostForecaster


def _ml_leaks(make) -> bool:
    """True if predictions up to cutoff change when rows AFTER cutoff are modified."""
    t = _table()
    cutoff = pd.Timestamp("2014-06-30")
    a = walk_forward(make, t, WIN, horizon=1)
    t2 = t.copy()
    future = t2.index > cutoff
    for c in t2.columns:
        if c not in ("entity", "seg") and t2[c].dtype.kind == "f":
            t2.loc[future, c] = t2.loc[future, c] * 3 + 1.0
    b = walk_forward(make, t2, WIN, horizon=1)
    a, b = a[a.date <= cutoff].reset_index(drop=True), b[b.date <= cutoff].reset_index(drop=True)
    if a.y_pred.notna().any():
        if not np.allclose(a.y_pred.to_numpy(), b.y_pred.to_numpy(), equal_nan=True):
            return True
    if a.p_up.notna().any():
        if not np.allclose(a.p_up.to_numpy(), b.p_up.to_numpy(), equal_nan=True):
            return True
    return False


def test_logistic_forecaster():
    t = _table()
    p = walk_forward(LogisticForecaster, t, WIN, horizon=1)
    assert len(p) > 0
    assert not _ml_leaks(LogisticForecaster)
    p_up = p["p_up"].dropna()
    assert len(p_up) > 0
    assert ((p_up >= 0.0) & (p_up <= 1.0)).all()
    stats = summarise(p)
    assert "accuracy" in stats or "n_dir" in stats


def test_random_forest_forecaster():
    t = _table()
    p = walk_forward(RandomForestForecaster, t, WIN, horizon=1)
    assert len(p) > 0
    assert not _ml_leaks(RandomForestForecaster)
    p_up = p["p_up"].dropna()
    assert len(p_up) > 0
    assert ((p_up >= 0.0) & (p_up <= 1.0)).all()
    stats = summarise(p)
    assert "rmse" in stats


def test_xgboost_forecaster():
    t = _table()
    p = walk_forward(XGBoostForecaster, t, WIN, horizon=1)
    assert len(p) > 0
    assert not _ml_leaks(XGBoostForecaster)
    p_up = p["p_up"].dropna()
    assert len(p_up) > 0
    assert ((p_up >= 0.0) & (p_up <= 1.0)).all()
    stats = summarise(p)
    assert "rmse" in stats
