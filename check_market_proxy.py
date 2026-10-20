
import pandas as pd

proxy = pd.read_parquet("data/processed/market_proxy.parquet")
features = pd.read_parquet("data/processed/features.parquet")

proxy.index = pd.to_datetime(proxy.index)
features["date"] = pd.to_datetime(features["date"])

proxy_dates = proxy.index
feature_dates = pd.Index(features["date"].unique())

print("MARKET PROXY")
print("Shape:", proxy.shape)
print("Index name:", proxy.index.name)
print("First date:", proxy.index.min())
print("Last date:", proxy.index.max())
print("Missing proxy values:", int(proxy.isna().sum().sum()))

print("\nDATE ALIGNMENT")
print("Unique feature dates:", len(feature_dates))
print("Dates in both datasets:",
      len(feature_dates.intersection(proxy_dates)))
print("Feature dates absent from proxy:",
      len(feature_dates.difference(proxy_dates)))

missing = features[features["mkt"].isna()].copy()
missing["date_absent_from_proxy"] = ~missing["date"].isin(proxy_dates)

print("\nMISSING MARKET VALUES BY CAUSE")
print(missing["date_absent_from_proxy"].value_counts(dropna=False))

print("\nMISSING MARKET VALUES BY ENTITY")
print(missing.groupby("entity").size().sort_values(ascending=False).to_string())

print("\nMISSING MARKET VALUES BY ENTITY AND SEGMENT")
print(
    missing.groupby(["entity", "seg"]).size()
    .sort_values(ascending=False)
    .head(20).to_string()
)

print("\nNo files modified.")


import pandas as pd
from src.data.clean import (
    load_raw,
    ordinary_shares,
    add_returns,
    core_tickers,
    coverage_table,
    returns_panel,
    MIN_PROXY_STOCKS,
)

raw = load_raw()
ordinary = add_returns(ordinary_shares(raw))
coverage = coverage_table(raw, ordinary)
tickers = core_tickers(coverage)

panel = returns_panel(ordinary, tickers)
valid_counts = panel.notna().sum(axis=1)

proxy = pd.read_parquet("data/processed/market_proxy.parquet")
proxy_dates = pd.DatetimeIndex(pd.to_datetime(proxy.index))
valid_counts.index = pd.to_datetime(valid_counts.index)

missing_dates = valid_counts.index.difference(proxy_dates)
counts = valid_counts.loc[missing_dates]

print("Minimum stocks required:", MIN_PROXY_STOCKS)
print("Dates missing from proxy:", len(missing_dates))
print("\nValid stock counts on missing dates:")
print(counts.value_counts().sort_index().to_string())

print("\nSummary:")
print("Minimum:", int(counts.min()) if len(counts) else "N/A")
print("Maximum:", int(counts.max()) if len(counts) else "N/A")
print("Missing dates with fewer than 8 stocks:",
      int((counts < MIN_PROXY_STOCKS).sum()))
print("Missing dates with 8 or more stocks:",
      int((counts >= MIN_PROXY_STOCKS).sum()))

print("\nNo project files modified.")
