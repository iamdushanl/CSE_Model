import numpy as np
import pandas as pd

from src.data.clean import (add_returns, core_tickers, coverage_table, market_proxy,
                            ordinary_shares, returns_panel)


def _raw():
    d = pd.bdate_range("2012-01-02", periods=30)
    rows = []
    for i, day in enumerate(d):
        rows.append(("AAA", day, "N", 100 + i))
        rows.append(("AAA", day, "X", 60 + i))          # other share class: must be ignored
        rows.append(("BBB", day, "N", 50 + 0.5 * i))
    return pd.DataFrame(rows, columns=["CompanyCode", "Date", "MainType", "Close"])


def test_only_ordinary_shares_kept():
    n = ordinary_shares(_raw())
    assert set(n.MainType) == {"N"}
    assert len(n) == 60


def test_split_and_gap_are_removed_from_clean_returns():
    raw = _raw()
    # a 10-for-1 split on AAA day 10, then a 60-day gap before the last 5 rows
    raw.loc[(raw.CompanyCode == "AAA") & (raw.MainType == "N") & (raw.Date >= raw.Date.iloc[30]), "Close"] /= 10
    n = add_returns(ordinary_shares(raw))
    a = n[n.CompanyCode == "AAA"]
    assert a.flag_split.sum() >= 1
    assert a.loc[a.flag_split, "lr_clean"].isna().all()
    assert a.lr_clean.abs().max() < 0.35


def test_gap_breaks_series():
    raw = _raw()
    late = raw.copy()
    late["Date"] = late["Date"] + pd.Timedelta(days=90)
    n = add_returns(ordinary_shares(pd.concat([raw, late])))
    a = n[n.CompanyCode == "AAA"]
    assert a.flag_gap.sum() == 1
    assert a.loc[a.flag_gap, "lr_clean"].isna().all()


def test_coverage_and_core_universe():
    raw = _raw()
    drop = raw[(raw.CompanyCode == "BBB")].index[::2]      # BBB misses half of its days
    n = add_returns(ordinary_shares(raw.drop(drop)))
    cov = coverage_table(raw, n)
    assert cov.set_index("CompanyCode").loc["AAA", "coverage"] == 1.0
    assert core_tickers(cov, 0.93) == ["AAA"]


def test_market_proxy_needs_enough_stocks():
    idx = pd.bdate_range("2020-01-01", periods=3)
    panel = pd.DataFrame({"A": [0.01, np.nan, 0.02], "B": [0.03, np.nan, np.nan],
                          "C": [0.02, 0.01, np.nan]}, index=idx)
    p = market_proxy(panel, min_stocks=2)
    assert list(p.index) == [idx[0]]
    assert abs(p.iloc[0] - 0.02) < 1e-12
    assert returns_panel(add_returns(ordinary_shares(_raw()))).shape[1] == 2
