"""Cleaning rules for the S&P SL20 CSE file.  Owner: Member 3.

Why this module exists (see handbook section 3):
  1. The sheet mixes several securities under one company code (ordinary N,
     non-voting X, warrants W, rights R, debentures D, preference P).
     The sheet's own DailyReturn_Pct / MA / Volatility columns are therefore wrong.
  2. There are large time gaps (2021-04..2021-12 and 2022-04..2023-12 are absent,
     and 2020-03..05 is the real COVID closure).
  3. Corporate actions (splits, rights, bonus issues) are not adjusted.

Rules applied here:
  * keep MainType == 'N' (ordinary voting shares) only;
  * a gap above GAP_DAYS calendar days breaks the series: the return across it is
    set to missing and features never cross it;
  * any daily |log return| above SPLIT_THRESHOLD is treated as a corporate action
    and set to missing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILE = "data/raw/SP20_CSE_Dataset_2001_2025_Adjusted.xlsx"
SHEET = "Full_Dataset_2001_2025"
GAP_DAYS = 10            # calendar days; a longer gap starts a new segment
SPLIT_THRESHOLD = 0.35   # |daily log return| above this = corporate action
MIN_COVERAGE = 0.93      # share of market days with an ordinary-share row
MIN_PROXY_STOCKS = 8     # minimum stocks present for a market-proxy day
USECOLS = ["Date", "CompanyCode", "MainType", "Close", "High", "Low",
           "TradeVolume", "ShareVolume", "Turnover"]


def load_raw(path: str | Path = RAW_FILE) -> pd.DataFrame:
    """Read the Full_Dataset sheet (about 87,000 rows, 10 s)."""
    df = pd.read_excel(path, sheet_name=SHEET, usecols=USECOLS)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def ordinary_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ordinary voting shares only, with a usable close price."""
    n = df[(df["MainType"] == "N") & (df["Close"] > 0)].copy()
    return n.sort_values(["CompanyCode", "Date"]).reset_index(drop=True)


def add_returns(n: pd.DataFrame, gap_days: int = GAP_DAYS,
                split_threshold: float = SPLIT_THRESHOLD) -> pd.DataFrame:
    """Add log returns and the cleaning flags.

    lr        raw daily log return within a company
    flag_gap  the previous row is more than gap_days calendar days earlier
    flag_split |lr| above split_threshold (corporate action or bad print)
    lr_clean  lr with gap and split rows set to NaN
    """
    n = n.sort_values(["CompanyCode", "Date"]).copy()
    by = n.groupby("CompanyCode", sort=False)
    n["gap_days"] = by["Date"].diff().dt.days
    n["lr"] = np.log(n["Close"]).groupby(n["CompanyCode"], sort=False).diff()
    n["flag_gap"] = n["gap_days"] > gap_days
    n["flag_split"] = n["lr"].abs() > split_threshold
    n["lr_clean"] = n["lr"].where(~(n["flag_gap"] | n["flag_split"]))
    return n


def coverage_table(df_all: pd.DataFrame, n: pd.DataFrame) -> pd.DataFrame:
    """Per company: how complete is the ordinary-share series?"""
    market_days = pd.Series(df_all["Date"].drop_duplicates().sort_values().values)
    rows = []
    for code, g in n.groupby("CompanyCode"):
        first, last = g["Date"].min(), g["Date"].max()
        window = ((market_days >= first) & (market_days <= last)).sum()
        valid = g["lr"].dropna()
        rows.append({
            "CompanyCode": code, "first_date": first.date(), "last_date": last.date(),
            "n_ordinary_rows": len(g), "market_days_in_window": int(window),
            "coverage": len(g) / window if window else np.nan,
            "zero_return_share": float((valid == 0).mean()) if len(valid) else np.nan,
            "n_gap_flags": int(g["flag_gap"].sum()),
            "n_split_flags": int((g["flag_split"] & ~g["flag_gap"]).sum()),
        })
    return pd.DataFrame(rows).sort_values("CompanyCode").reset_index(drop=True)


def core_tickers(cov: pd.DataFrame, min_coverage: float = MIN_COVERAGE) -> list[str]:
    """Stocks with a complete enough ordinary-share series."""
    return sorted(cov.loc[cov["coverage"] >= min_coverage, "CompanyCode"])


def returns_panel(n: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    """Date x ticker matrix of cleaned log returns (decimal, not percent)."""
    sub = n if tickers is None else n[n["CompanyCode"].isin(tickers)]
    return sub.pivot(index="Date", columns="CompanyCode", values="lr_clean").sort_index()


def market_proxy(panel: pd.DataFrame, min_stocks: int = MIN_PROXY_STOCKS) -> pd.Series:
    """Equal-weighted mean return of the stocks present that day (our own index)."""
    ok = panel.notna().sum(axis=1) >= min_stocks
    proxy = panel.mean(axis=1, skipna=True)[ok]
    proxy.name = "MARKET"
    return proxy


def audit_tables(df_all: pd.DataFrame, n: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables for reports/tables/ (Member 3 extends these)."""
    share_types = (df_all.groupby(["CompanyCode", "MainType"]).size()
                   .unstack(fill_value=0).reset_index())
    cov = coverage_table(df_all, n)
    flagged = (n[n["flag_split"] | n["flag_gap"]]
               [["CompanyCode", "Date", "Close", "lr", "gap_days", "flag_gap", "flag_split"]]
               .copy())
    prev = n.groupby("CompanyCode")["Close"].shift(1)
    flagged["prev_close"] = prev.loc[flagged.index]
    all_dates = pd.Series(df_all["Date"].drop_duplicates().sort_values().values)
    gap = all_dates.diff().dt.days
    gaps = pd.DataFrame({"gap_ends_on": all_dates[gap > GAP_DAYS].values,
                         "gap_days": gap[gap > GAP_DAYS].values})
    return {"data_audit_share_types": share_types, "data_audit_coverage": cov,
            "corporate_actions_flagged": flagged, "data_gaps": gaps}


def build_processed(raw_path: str | Path = RAW_FILE, out_dir: str | Path = "data/processed",
                    audit_dir: str | Path = "reports/tables") -> dict:
    """Run the whole cleaning pipeline and write parquet and audit files."""
    from src.features.build import build_feature_table  # local import avoids a cycle

    out_dir, audit_dir = Path(out_dir), Path(audit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_path)
    n = add_returns(ordinary_shares(df))
    tables = audit_tables(df, n)
    for name, t in tables.items():
        t.to_csv(audit_dir / f"{name}.csv", index=False)
    core = core_tickers(tables["data_audit_coverage"])
    panel = returns_panel(n, core)
    proxy = market_proxy(panel)

    n[["CompanyCode", "Date", "Close", "lr", "lr_clean", "flag_gap", "flag_split"]] \
        .to_parquet(out_dir / "returns_clean.parquet", index=False)
    proxy.to_frame().to_parquet(out_dir / "market_proxy.parquet")
    feats = build_feature_table(n, core, proxy)
    feats.reset_index().to_parquet(out_dir / "features.parquet", index=False)
    return {"core_tickers": core, "n_rows_ordinary": len(n),
            "n_returns_removed": int((n["lr"].notna() & n["lr_clean"].isna()).sum()),
            "n_feature_rows": len(feats)}
