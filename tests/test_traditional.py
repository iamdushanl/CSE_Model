import numpy as np
import pandas as pd
import pytest
from src.models.baselines import ZeroForecaster, MeanForecaster, PersistenceForecaster
from src.models.ts.arma import ARForecaster, ARMAForecaster, select_orders


def sample(n=90):
    rng = np.random.default_rng(42)
    r = rng.normal(size=n)
    idx = pd.bdate_range('2017-01-02', periods=n)
    return pd.DataFrame({'entity': 'MARKET', 'seg': 0, 'r': r,
                         'm5': pd.Series(r).rolling(5).mean().to_numpy(),
                         'y_ret_1': np.roll(r, -1), 'y_dir_1': (np.roll(r, -1) > 0).astype(float),
                         'y_ret_5': np.roll(r, -5)}, index=idx)


def test_baselines():
    train, test = sample().iloc[:70], sample().iloc[70:]
    z = ZeroForecaster().fit(train, 1).predict(train, test)
    assert np.all(z.y_pred == 0) and z.p_up.isna().all()
    m = MeanForecaster().fit(train, 1).predict(train, test)
    assert np.allclose(m.y_pred, train.y_ret_1.mean())
    assert np.all((m.p_up >= 0) & (m.p_up <= 1))
    p = PersistenceForecaster().fit(train, 5).predict(train, test)
    assert np.allclose(p.y_pred, test.m5 * 5)


@pytest.mark.parametrize('model', [ARForecaster, ARMAForecaster])
def test_multistep_and_probabilities(model):
    df = sample()
    fitted = model().fit(df.iloc[:70], 5)
    result = fitted.predict(df.iloc[:70], df.iloc[70:75])
    assert len(result) == 5
    assert np.isfinite(result.y_pred).all()
    assert result.p_up.between(0, 1).all()
    assert not np.allclose(result.y_pred, 5 * result.y_pred.iloc[0])


def test_order_limits():
    with pytest.raises(ValueError):
        select_orders(sample(), max_p=6)
    with pytest.raises(ValueError):
        ARForecaster(p=6)
