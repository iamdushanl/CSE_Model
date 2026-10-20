import numpy as np
import pandas as pd

from src.eda.stat_tests import adf_test, arch_lm_test, kpss_test, stat_test_table


def _noise(n=600, seed=42):
    return pd.Series(np.random.default_rng(seed).normal(size=n),
                     index=pd.bdate_range("2010-01-04", periods=n))


def test_adf_kpss_tell_random_walk_from_noise():
    noise = _noise()
    walk = noise.cumsum()
    assert adf_test(noise)[1] < 0.01          # noise is stationary
    assert adf_test(walk)[1] > 0.05           # random walk has a unit root
    assert kpss_test(walk)[1] < 0.05          # KPSS flags the walk as non-stationary


def test_arch_lm_detects_volatility_clustering():
    rng = np.random.default_rng(42)
    sigma = np.where(np.arange(800) % 200 < 100, 0.3, 3.0)      # calm / wild blocks
    x = pd.Series(rng.normal(size=800) * sigma, index=pd.bdate_range("2010-01-04", periods=800))
    assert arch_lm_test(x)[1] < 0.01
    assert arch_lm_test(_noise(800))[1] > 0.01


def test_table_skips_short_segments():
    r = _noise(600)
    feats = pd.concat([
        pd.DataFrame({"entity": "A", "seg": 0, "r": r}),
        pd.DataFrame({"entity": "A", "seg": 1, "r": r.iloc[:50]}),
    ])
    out = stat_test_table(feats)
    assert set(out.seg) == {0}
