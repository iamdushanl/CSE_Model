import numpy as np
import pandas as pd
from src.models.ts.arx import StockARXForecaster


def test_stock_specific_arx():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2017-01-01', periods=60)
    frames = []
    for stock, b in [('AAA', 1.), ('BBB', -1.)]:
        r = rng.normal(size=60)
        mkt = rng.normal(size=60)
        frames.append(pd.DataFrame({'entity': stock, 'r': r, 'mkt': mkt,
                                    'y_ret_1': b * mkt + 0.2 * r}, index=dates))
    train = pd.concat(frames)
    model = StockARXForecaster().fit(train, 1)
    out = model.predict(train, train.iloc[:5].drop(columns=['y_ret_1']))
    assert len(model.params_) == 2
    assert np.isfinite(out.y_pred).all()
    assert out.p_up.between(0, 1).all()
