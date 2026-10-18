import numpy as np
import pandas as pd

from src.features.build import BASE_FEATURES, TARGETS, build_entity_features


def _series(n=120, seed=0, gap_at=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    if gap_at is not None:
        idx = idx[:gap_at].append(idx[gap_at:] + pd.Timedelta(days=60))
    return pd.Series(rng.normal(0, 0.01, n), index=idx)


def test_features_never_use_the_future():
    s = _series()
    base = build_entity_features(s, "X")
    cut = s.index[80]
    changed = s.copy()
    changed[changed.index > cut] = changed[changed.index > cut] + 0.5   # rewrite the future
    new = build_entity_features(changed, "X")
    feats = BASE_FEATURES + ["zero_ret", "dow"]
    pd.testing.assert_frame_equal(base.loc[:cut, feats], new.loc[:cut, feats])


def test_targets_look_forward_and_match_definition():
    s = _series()
    f = build_entity_features(s, "X")
    r = s * 100
    i = 50
    assert abs(f["y_ret_1"].iloc[i] - r.iloc[i + 1]) < 1e-9
    assert abs(f["y_ret_5"].iloc[i] - r.iloc[i + 1:i + 6].sum()) < 1e-9
    assert abs(f["y_rv_5"].iloc[i] - (r.iloc[i + 1:i + 6] ** 2).sum()) < 1e-9


def test_windows_do_not_cross_a_gap():
    s = _series(gap_at=60)
    f = build_entity_features(s, "X")
    assert f["seg"].nunique() == 2
    first_new = f.index[60]
    after = f.loc[first_new:].iloc[:22]
    assert after["m22"].iloc[:21].isna().all()            # needs 22 obs inside the new segment
    assert f["y_ret_1"].iloc[59] != f["y_ret_1"].iloc[59] # last day before the gap has no target
    assert f["l1"].iloc[60] != f["l1"].iloc[60]           # no lag across the gap


def test_missing_return_breaks_rolling_windows():
    s = _series()
    s.iloc[40] = np.nan                                   # a split day removed by cleaning
    f = build_entity_features(s, "X")
    assert f["m5"].iloc[40:45].isna().all()
    assert set(TARGETS) <= set(f.columns)
