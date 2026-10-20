"""Figures and tables for the EDA notebook.  Owner: Member 4 (task E1).

Every function takes the feature table from load_features() (columns entity, seg, r;
r is a daily log return in PERCENT) and returns a figure or a table. Nothing here
runs a statistical test across a gap in the data: segments are kept apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.viz.style import PALETTE

GAP_DAYS = 10


def market_frame(feats: pd.DataFrame) -> pd.DataFrame:
    """Rows of the MARKET entity, in date order."""
    return feats[feats["entity"] == "MARKET"].sort_index()


def longest_segment_returns(feats: pd.DataFrame, entity: str = "MARKET") -> pd.Series:
    """Returns of the longest unbroken stretch of one entity (safe for ACF and tests)."""
    g = feats[feats["entity"] == entity]
    seg = g.groupby("seg")["r"].count().idxmax()
    return g.loc[g["seg"] == seg, "r"].dropna()


def fig_price_and_returns(feats: pd.DataFrame):
    """Cumulative log return and daily return of the market proxy; gaps shaded."""
    m = market_frame(feats)
    level = m["r"].fillna(0).cumsum()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for _, g in m.groupby("seg"):                     # one line per unbroken stretch
        ax1.plot(level.loc[g.index], color=PALETTE["traditional"], lw=1.1)
        ax2.plot(g.index, g["r"], color=PALETTE["baseline"], lw=0.5)
    bounds = m.index.to_series().groupby(m["seg"]).agg(["min", "max"])
    for end, start in zip(bounds["max"].iloc[:-1], bounds["min"].iloc[1:]):
        if (start - end).days > GAP_DAYS:
            closure = end.year == 2020
            for ax in (ax1, ax2):
                ax.axvspan(end, start, color=PALETTE["shock"] if closure else "#bbbbbb",
                           alpha=0.30 if closure else 0.35, lw=0)
    ax1.set_ylabel("cumulative log return (%)")
    ax2.set_ylabel("daily return (%)")
    ax1.set_title("Market proxy: price level (top) and daily returns (bottom)")
    ax1.legend(handles=[Patch(color=PALETTE["shock"], alpha=0.3, label="2020 closure"),
                        Patch(color="#bbbbbb", alpha=0.35, label="no data / gap")],
               loc="upper left")
    fig.tight_layout()
    return fig


def fig_distribution(feats: pd.DataFrame):
    """Histogram with a normal curve, and a QQ plot, of market daily returns."""
    r = market_frame(feats)["r"].dropna()
    mu, sd = r.mean(), r.std()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.hist(r, bins=100, density=True, color=PALETTE["traditional"], alpha=0.75)
    xs = np.linspace(r.min(), r.max(), 400)
    ax1.plot(xs, stats.norm.pdf(xs, mu, sd), color=PALETTE["ml"], lw=1.5, label="normal, same mean and sd")
    ax1.set_xlim(-6, 6)
    ax1.set_xlabel("daily return (%)")
    ax1.set_title("Distribution of market returns")
    ax1.legend()
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
    ax2.scatter(osm, osr, s=4, color=PALETTE["traditional"])
    ax2.plot(osm, slope * osm + intercept, color=PALETTE["ml"], lw=1.5)
    ax2.set_xlabel("normal quantiles")
    ax2.set_ylabel("sample quantiles (%)")
    ax2.set_title("QQ plot against the normal")
    fig.tight_layout()
    return fig


def distribution_summary(feats: pd.DataFrame) -> pd.DataFrame:
    """Mean, sd, skew, excess kurtosis and Jarque-Bera test of market returns."""
    r = market_frame(feats)["r"].dropna()
    jb = stats.jarque_bera(r)
    return pd.DataFrame({"n": [len(r)], "mean": [r.mean()], "sd": [r.std()], "skew": [stats.skew(r)],
                         "excess_kurtosis": [stats.kurtosis(r)], "jb_stat": [jb.statistic],
                         "jb_p_value": [jb.pvalue]}).round(4)


def fig_acf_pacf(feats: pd.DataFrame, lags: int = 30):
    """ACF, PACF of returns and ACF of squared returns, longest market stretch only."""
    x = longest_segment_returns(feats).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    plot_acf(x, lags=lags, ax=axes[0], zero=False, title="ACF of returns")
    plot_pacf(x, lags=lags, ax=axes[1], zero=False, method="ywm", title="PACF of returns")
    plot_acf(x ** 2, lags=lags, ax=axes[2], zero=False, title="ACF of squared returns")
    for ax in axes:
        ax.set_xlabel("lag (trading days)")
    fig.tight_layout()
    return fig


def yearly_table(feats: pd.DataFrame) -> pd.DataFrame:
    """Per calendar year: trading days, return and volatility of the market proxy."""
    r = market_frame(feats)["r"].dropna()
    g = r.groupby(r.index.year)
    out = pd.DataFrame({"n_days": g.count(), "mean_daily_pct": g.mean(), "total_log_return_pct": g.sum(),
                        "annualised_vol_pct": g.std() * np.sqrt(252), "worst_day_pct": g.min(),
                        "best_day_pct": g.max()})
    out.index.name = "year"
    return out.round(3).reset_index()


def fig_correlation(feats: pd.DataFrame):
    """Heat map of pairwise return correlations between the stocks."""
    d = feats[feats["entity"] != "MARKET"].reset_index()
    wide = d.pivot_table(index="date", columns="entity", values="r")
    corr = wide.corr(min_periods=250)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=max(0.5, float(np.nanmax(np.where(corr.values < 1, corr.values, 0)))))
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation of daily returns")
    ax.set_title("Cross-stock return correlation")
    fig.tight_layout()
    return fig, corr.round(3)
