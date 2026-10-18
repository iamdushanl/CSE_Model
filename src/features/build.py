"""Leakage-safe feature table.  Owner: Member 3 (feature list agreed with Teddy).

Rules (tests/test_features.py enforces them):
  * every feature at day t uses data up to and including day t only;
  * every target looks forward from day t;
  * rolling windows and shifts never cross a segment break (a gap above 10 days).

Returns are in PERCENT inside the feature table (r = 100 * log return).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_DAYS = 10
BASE_FEATURES = ["r", "l1", "l2", "l3", "l4", "m5", "m22", "v5", "v22", "abs_r"]
EXTRA_FEATURES = ["zero_ret", "dow"]          # available for every entity
PANEL_FEATURES = ["mkt"]                      # same-day market return, stocks only
TARGETS = ["y_ret_1", "y_dir_1", "y_ret_5", "y_dir_5", "y_rv_5"]
KEYS = ["entity", "seg"]


def _forward_sum(x: pd.Series, h: int) -> pd.Series:
    """Sum of x[t+1 .. t+h] at position t (NaN if any term is missing)."""
    return x[::-1].rolling(h).sum()[::-1].shift(-1)


def build_entity_features(r_dec: pd.Series, entity: str, mkt: pd.Series | None = None) -> pd.DataFrame:
    """Features and targets for ONE entity.

    r_dec: cleaned daily log returns (decimal) indexed by the entity's own trading
           dates. Missing returns (split or gap days) must stay as NaN rows so that
           they break rolling windows instead of being skipped.
    mkt:   optional market-proxy return (decimal) used for the 'mkt' feature.
    """
    s = (r_dec.astype(float) * 100.0).sort_index()
    seg = (s.index.to_series().diff().dt.days > GAP_DAYS).cumsum().to_numpy()
    df = pd.DataFrame({"r": s.to_numpy()}, index=s.index)
    df.index.name = "date"
    df["entity"], df["seg"] = entity, seg
    g = df.groupby("seg")["r"]
    for k in range(1, 5):
        df[f"l{k}"] = g.shift(k)
    df["m5"] = g.transform(lambda x: x.rolling(5).mean())
    df["m22"] = g.transform(lambda x: x.rolling(22).mean())
    df["v5"] = g.transform(lambda x: x.rolling(5).std())
    df["v22"] = g.transform(lambda x: x.rolling(22).std())
    df["abs_r"] = df["r"].abs()
    df["zero_ret"] = (df["r"] == 0).astype(float).where(df["r"].notna())
    df["dow"] = df.index.dayofweek.astype(float)
    df["mkt"] = (mkt.reindex(df.index) * 100.0).to_numpy() if mkt is not None else np.nan
    # ---- targets (look forward) ----
    df["y_ret_1"] = g.shift(-1)
    df["y_ret_5"] = g.transform(lambda x: _forward_sum(x, 5))
    r2 = (df["r"] ** 2).groupby(df["seg"])
    df["y_rv_5"] = r2.transform(lambda x: _forward_sum(x, 5))
    for h in (1, 5):
        y = df[f"y_ret_{h}"]
        df[f"y_dir_{h}"] = (y > 0).astype(float).where(y.notna() & (y != 0))
    cols = KEYS + BASE_FEATURES + EXTRA_FEATURES + PANEL_FEATURES + TARGETS
    return df[cols]


def build_feature_table(n: pd.DataFrame, tickers: list[str], proxy: pd.Series) -> pd.DataFrame:
    """MARKET entity plus one entity per core ticker, stacked, indexed by date."""
    parts = [build_entity_features(proxy, "MARKET")]
    for t in tickers:
        s = n.loc[n["CompanyCode"] == t].set_index("Date")["lr_clean"]
        parts.append(build_entity_features(s, t, mkt=proxy))
    return pd.concat(parts).sort_index(kind="stable")


def load_features(path: str = "data/processed/features.parquet") -> pd.DataFrame:
    """Read the feature table written by `python run.py data`."""
    df = pd.read_parquet(path)
    return df.set_index("date").sort_index(kind="stable")
