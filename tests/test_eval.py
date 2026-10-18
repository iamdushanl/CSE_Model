import numpy as np
import pandas as pd

from src.eval.dm import dm_test
from src.eval.metrics import oos_r2, summarise
from src.eval.walkforward import metrics_table, walk_forward
from src.features.build import build_entity_features
from src.models.baselines import MeanForecaster, PersistenceForecaster, ZeroForecaster
from src.models.template_linear import TemplateAR5

WIN = {"test_start": "2014-01-01", "test_end": "2014-12-31"}


def _table(seed=1, n=1800):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=n)
    r = pd.Series(rng.normal(0.0002, 0.01, n), index=idx)
    r = r + 0.2 * r.shift(1).fillna(0)                    # build in some autocorrelation
    return build_entity_features(r, "MARKET")


def test_zero_forecast_has_zero_oos_r2():
    y = np.array([1.0, -2.0, 0.5])
    assert oos_r2(y, np.zeros(3)) == 0.0


def test_dm_identical_models_gives_nan_and_better_model_negative_stat():
    e = np.random.default_rng(0).normal(size=300)
    assert np.isnan(dm_test(e ** 2, e ** 2)[1])
    stat, p = dm_test((0.5 * e) ** 2, e ** 2)
    assert stat < 0 and p < 0.05


def test_walk_forward_output_shape_and_columns():
    t = _table()
    p = walk_forward(ZeroForecaster, t, WIN, horizon=1)
    assert {"date", "entity", "y_ret", "y_dir", "y_pred", "p_up", "model", "horizon"} <= set(p.columns)
    assert p.date.min() >= pd.Timestamp("2014-01-01") and p.date.max() <= pd.Timestamp("2014-12-31")
    assert (p.fit_year == 2014).all()


def _leaks(make) -> bool:
    """True if predictions up to a cutoff change when rows AFTER the cutoff are rewritten."""
    t = _table()
    cutoff = pd.Timestamp("2014-06-30")
    a = walk_forward(make, t, WIN, horizon=1)
    t2 = t.copy()
    future = t2.index > cutoff
    for c in t2.columns:
        if c not in ("entity", "seg") and t2[c].dtype.kind == "f":
            t2.loc[future, c] = t2.loc[future, c] * 3 + 1.0          # rewrite everything after the cutoff
    b = walk_forward(make, t2, WIN, horizon=1)
    a, b = a[a.date <= cutoff].reset_index(drop=True), b[b.date <= cutoff].reset_index(drop=True)
    return not np.allclose(a.y_pred.to_numpy(), b.y_pred.to_numpy())


def test_no_leakage_future_rows_do_not_change_earlier_predictions():
    for make in (ZeroForecaster, MeanForecaster, PersistenceForecaster, TemplateAR5):
        assert not _leaks(make), make.__name__


def test_leakage_check_catches_a_cheating_model():
    class Cheater(ZeroForecaster):
        name = "cheater"

        def predict(self, history, test):                   # uses the whole test year: future info
            return self.frame(test.index, y_pred=(test["r"] - test["r"].mean()).to_numpy())

    assert _leaks(Cheater)


def test_training_never_includes_the_test_year():
    seen = {}

    class Spy(ZeroForecaster):
        name = "spy"

        def fit(self, train, horizon):
            seen["last_train_date"] = train.index.max()
            return self

    walk_forward(Spy, _table(), WIN, horizon=5)
    assert seen["last_train_date"] < pd.Timestamp("2014-01-01")


def test_ar_template_beats_zero_when_returns_are_autocorrelated():
    t = _table()
    pz = walk_forward(ZeroForecaster, t, WIN, 1)
    pa = walk_forward(TemplateAR5, t, WIN, 1)
    assert summarise(pa)["oos_r2"] > 0
    mt = metrics_table(pd.concat([pz, pa, walk_forward(MeanForecaster, t, WIN, 1)]))
    assert {"dm_p", "best_baseline"} <= set(mt.columns)
