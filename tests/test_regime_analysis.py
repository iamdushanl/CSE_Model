import numpy as np
import pandas as pd

from src.eda.regime_analysis import error_table


def _preds(n=240, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-06", periods=n)
    y = rng.normal(size=n)
    base = {"date": dates, "entity": "MARKET", "y_ret": y, "y_dir": (y > 0).astype(float), "p_up": np.nan,
            "fit_year": 2020, "horizon": 1, "window": "test1", "entity_set": "MARKET",
            "vol": np.where(np.arange(n) % 2 == 0, "calm", "turbulent"), "crisis": "other"}
    zero = pd.DataFrame({**base, "y_pred": 0.0, "model": "zero"})
    good = pd.DataFrame({**base, "y_pred": 0.5 * y, "model": "m_good"})      # deliberately informative
    return pd.concat([zero, good], ignore_index=True)


def test_error_table_scores_every_regime_and_dm_prefers_the_better_model():
    t = error_table(_preds())
    vol = t[t["regime_type"] == "volatility"]
    assert set(vol["regime"]) == {"calm", "turbulent"}
    good = vol[vol["model"] == "m_good"]
    assert (good["oos_r2"] > 0).all()
    assert (good["dm_stat"] < 0).all() and (good["dm_p"] < 0.05).all()
    assert (vol[vol["model"] == "zero"]["oos_r2"] == 0).all()
