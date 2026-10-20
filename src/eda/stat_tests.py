"""Statistical tests for the EDA.  Owner: Member 4 (tasks E1 and E3).

Every test is run INSIDE one unbroken stretch of trading days (a 'segment'),
never across the 2020 closure or the 2021 to 2023 holes. Only segments with at
least MIN_OBS observations are tested.

Run:  python -m src.eda.stat_tests      (writes reports/tables/stat_tests.csv)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

# statsmodels announces a future change of return type; harmless here
warnings.filterwarnings("ignore", message=".*currently returns a plain tuple.*")

MIN_OBS = 200
LAGS = 10
ALPHA = 0.05


def adf_test(x: pd.Series) -> tuple[float, float]:
    """H0: unit root (non-stationary). Small p-value means stationary."""
    stat, p = adfuller(x.dropna(), autolag="AIC")[:2]
    return float(stat), float(p)


def kpss_test(x: pd.Series) -> tuple[float, float]:
    """H0: stationary. Small p-value means NON-stationary (opposite of ADF)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # p-value is clipped to 0.01..0.10
        stat, p = kpss(x.dropna(), regression="c", nlags="auto")[:2]
    return float(stat), float(p)


def ljung_box_test(x: pd.Series, lags: int = LAGS) -> tuple[float, float]:
    """H0: no autocorrelation up to `lags`. Small p-value means autocorrelated."""
    out = acorr_ljungbox(x.dropna(), lags=[lags], return_df=True)
    return float(out["lb_stat"].iloc[0]), float(out["lb_pvalue"].iloc[0])


def arch_lm_test(x: pd.Series, lags: int = LAGS) -> tuple[float, float]:
    """H0: no ARCH effects (constant variance). Small p-value means volatility clusters."""
    x = x.dropna()
    stat, p = het_arch(x - x.mean(), nlags=lags)[:2]
    return float(stat), float(p)


def jarque_bera_test(x: pd.Series) -> tuple[float, float]:
    """H0: normal. Small p-value means non-normal (skewed or fat-tailed)."""
    res = stats.jarque_bera(x.dropna())
    return float(res.statistic), float(res.pvalue)


def _verdict(test: str, p: float) -> str:
    sig = p < ALPHA
    return {
        "ADF": "stationary" if sig else "unit root not rejected",
        "KPSS": "NON-stationary" if sig else "stationary not rejected",
        "Ljung-Box": "autocorrelated" if sig else "no autocorrelation found",
        "Ljung-Box (squared)": "volatility clustering" if sig else "no clustering found",
        "ARCH-LM": "ARCH effects" if sig else "no ARCH effects",
        "Jarque-Bera": "non-normal" if sig else "normality not rejected",
    }[test]


def run_segment_tests(x: pd.Series, kind: str) -> list[dict]:
    """All tests for one series inside one segment. kind: 'return' or 'log_price'."""
    jobs = [("ADF", adf_test, x), ("KPSS", kpss_test, x)]
    if kind == "return":
        jobs += [("Ljung-Box", ljung_box_test, x),
                 ("Ljung-Box (squared)", ljung_box_test, x ** 2),
                 ("ARCH-LM", arch_lm_test, x),
                 ("Jarque-Bera", jarque_bera_test, x)]
    rows = []
    for name, fn, series in jobs:
        stat, p = fn(series)
        rows.append({"test": name, "statistic": round(stat, 4), "p_value": round(p, 4),
                     "conclusion": _verdict(name, p)})
    return rows


def stat_test_table(feats: pd.DataFrame, min_obs: int = MIN_OBS) -> pd.DataFrame:
    """One row per entity, segment, series kind and test.

    feats: output of load_features() (columns entity, seg, r; r in percent).
    The log-price proxy is the running sum of returns inside a segment.
    """
    rows = []
    for (entity, seg), g in feats.groupby(["entity", "seg"]):
        r = g["r"].dropna()
        if len(r) < min_obs:
            continue
        base = {"entity": entity, "seg": int(seg), "start": r.index.min().date(),
                "end": r.index.max().date(), "n": len(r)}
        for kind, series in (("log_price", r.cumsum()), ("return", r)):
            for row in run_segment_tests(series, kind):
                rows.append({**base, "series": kind, **row})
    return pd.DataFrame(rows)


def descriptives(feats: pd.DataFrame, min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Mean, std, skew and excess kurtosis of returns per entity and segment."""
    rows = []
    for (entity, seg), g in feats.groupby(["entity", "seg"]):
        r = g["r"].dropna()
        if len(r) < min_obs:
            continue
        rows.append({"entity": entity, "seg": int(seg), "n": len(r), "mean": r.mean(),
                     "std": r.std(), "skew": stats.skew(r), "excess_kurtosis": stats.kurtosis(r)})
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    from pathlib import Path

    from src.features.build import load_features

    f = load_features()
    out = stat_test_table(f)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv("reports/tables/stat_tests.csv", index=False)
    descriptives(f).to_csv("reports/tables/return_descriptives.csv", index=False)
    print(f"wrote {len(out)} test rows")
    m = out[(out.entity == "MARKET") & (out.seg == 1)]
    print(m[["series", "test", "statistic", "p_value", "conclusion"]].to_string(index=False))
