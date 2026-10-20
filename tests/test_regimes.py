import numpy as np
import pandas as pd

from src.eda.regimes import label, tercile_cutoffs


def test_regime_cutoffs_use_training_rows_only():
    idx = pd.bdate_range("2010-01-04", periods=800)
    vol = pd.Series(np.r_[np.linspace(1, 2, 400), np.full(400, 100.0)], index=idx)
    lo, hi = tercile_cutoffs(vol, train_end=str(idx[399].date()))
    assert hi <= 2.0                          # the huge later values must not leak in
    assert (label(vol, lo, hi).iloc[-1]) == "turbulent"
