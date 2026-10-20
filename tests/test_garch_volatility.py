import numpy as np
import pytest
from scripts.volatility_analysis import qlike
from src.models.ts.garch import GarchForecaster

def test_qlike():
    assert np.isfinite(qlike([0.,2.],[1.,2.]))
    with pytest.raises(ValueError): qlike([1.],[0.])

def test_garch_registration():
    from src.models.registry import MODELS
    assert MODELS['ar_gjr_garch_t'] is GarchForecaster
