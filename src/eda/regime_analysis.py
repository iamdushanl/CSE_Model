"""Forecast error by regime (task E5 / analysis A2).  Owner: Member 4.

Uses the regimes pre-registered in docs/regimes.md:
  * volatility regime of the market proxy (calm / normal / turbulent), cut-offs from
    TRAINING data only (see src/eda/regimes.py);
  * the 2020 crisis window (2020-03 to 2020-12), looked at inside the test1 window;
  * the post-gap regime (2024 to 2025) is the test2 window itself.

The regime of a forecast is the regime on the day the forecast is MADE (day t), which
is known at that time, so nothing here looks into the future.
This module only analyses saved predictions; it never fits or tunes a model.

Run:  python -m src.eda.regime_analysis   (needs `python run.py models` first)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.eda.regimes import label, load_market, rolling_vol, tercile_cutoffs
from src.eval.config import BASELINE_NAMES
from src.eval.dm import dm_test
from src.eval.metrics import summarise
from src.eval.walkforward import load_all_predictions
from src.viz.style import PALETTE

CRISIS = ("2020-03-01", "2020-12-31")
VOL_ORDER = ["calm", "normal", "turbulent"]
LOW_N_DATES = 30                      # fewer test days than this: treat results as weak evidence
TRADITIONAL = ("template_ar5", "ar", "arma", "ar_gjr_garch_t")
ML = ("logistic", "random_forest", "xgboost", "lstm")


def regime_labels() -> pd.DataFrame:
    """One row per market day with its volatility regime and crisis flag."""
    vol = rolling_vol(load_market())
    lo, hi = tercile_cutoffs(vol)
    out = pd.DataFrame({"vol": label(vol, lo, hi)})
    out["crisis"] = np.where((out.index >= CRISIS[0]) & (out.index <= CRISIS[1]), "crisis_2020", "other")
    return out


def attach_regimes(preds: pd.DataFrame, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add the regime columns to a predictions table; drops days without a regime."""
    labels = regime_labels() if labels is None else labels
    p = preds.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.join(labels, on="date")
    return p.dropna(subset=["vol"])


def error_table(p: pd.DataFrame, windows=("test1", "test2")) -> pd.DataFrame:
    """Metrics and a Diebold-Mariano test for every model inside every regime.

    The DM test compares each model with the best BASELINE inside the same regime, on
    squared errors averaged per date. dm_stat < 0 means the model is better.
    """
    rows = []
    for (win, es, h), g in p[p["window"].isin(windows)].groupby(["window", "entity_set", "horizon"]):
        for rtype, col in (("volatility", "vol"), ("crisis", "crisis")):
            if rtype == "crisis" and win != "test1":
                continue
            for reg, gg in g.groupby(col):
                n_dates = int(gg["date"].nunique())
                stats_by_model = {m: summarise(x) for m, x in gg.groupby("model")}
                base = [b for b in BASELINE_NAMES if b in stats_by_model and "rmse" in stats_by_model[b]]
                best = min(base, key=lambda b: stats_by_model[b]["rmse"]) if base else None

                def loss(model):
                    x = gg[gg["model"] == model].dropna(subset=["y_ret", "y_pred"])
                    return ((x["y_ret"] - x["y_pred"]) ** 2).groupby(x["date"]).mean()

                for m, st in stats_by_model.items():
                    row = {"window": win, "entity_set": es, "horizon": int(h), "regime_type": rtype,
                           "regime": reg, "model": m, "n_dates": n_dates, **st,
                           "best_baseline": best, "low_n": n_dates < LOW_N_DATES}
                    if best and m != best and "rmse" in st:
                        a, b = loss(m).align(loss(best), join="inner")
                        row["dm_stat"], row["dm_p"] = dm_test(a.to_numpy(), b.to_numpy(), h=int(h))
                    rows.append(row)
    cols = ["window", "entity_set", "horizon", "regime_type", "regime", "model", "n_dates", "n", "rmse",
            "mae", "oos_r2", "n_dir", "accuracy", "base_rate_up", "balanced_acc", "auc", "best_baseline",
            "dm_stat", "dm_p", "low_n"]
    return pd.DataFrame(rows).reindex(columns=cols).round(5)


def _color(model: str) -> str:
    if model in ML:
        return PALETTE["ml"]
    if model in TRADITIONAL:
        return PALETTE["traditional"]
    return PALETTE["baseline"]


def fig_regime_r2(table: pd.DataFrame, window: str, entity_set: str = "MARKET"):
    """Out-of-sample R2 (against the zero forecast) by volatility regime, for h=1 and h=5."""
    t = table[(table["window"] == window) & (table["entity_set"] == entity_set)
              & (table["regime_type"] == "volatility") & (table["model"] != "zero")]
    t = t[t.groupby("model")["oos_r2"].transform(lambda x: x.notna().any())]   # drops classifiers (no R2)
    models = sorted(t["model"].unique(), key=lambda m: (m in ML, m in TRADITIONAL, m))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    width = 0.8 / max(len(models), 1)
    for ax, h in zip(axes, (1, 5)):
        th = t[t["horizon"] == h]
        for i, m in enumerate(models):
            vals = [th[(th["model"] == m) & (th["regime"] == r)]["oos_r2"].mean() for r in VOL_ORDER]
            ax.bar(np.arange(3) + i * width - 0.4 + width / 2, vals, width, label=m, color=_color(m),
                   alpha=0.55 + 0.45 * ((i % 3) / 2))
        nd = th.groupby("regime")["n_dates"].first().reindex(VOL_ORDER)
        ax.set_xticks(range(3), [f"{r}\n({int(nd[r]) if pd.notna(nd[r]) else 0} days)" for r in VOL_ORDER])
        ax.axhline(0, color="black", lw=0.8)
        lo = -0.35                                       # naive persistence can reach -1 or lower: clip the axis
        ax.set_ylim(lo, 0.4)
        for bar in ax.patches:
            if bar.get_height() < lo:
                ax.text(bar.get_x() + bar.get_width() / 2, lo + 0.02, f"{bar.get_height():.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=90, color="white")
        ax.set_title(f"{entity_set}, horizon {h} day{'s' if h > 1 else ''}")
        ax.set_ylabel("out-of-sample R2 vs zero forecast")
    axes[0].legend(ncol=2, fontsize=8, loc="best")
    fig.suptitle(f"Forecast accuracy by volatility regime, window {window} (above 0 = beats the zero forecast)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from pathlib import Path

    preds = load_all_predictions()
    if preds.empty:
        raise SystemExit("no predictions yet: run `python run.py models` first")
    tab = error_table(attach_regimes(preds))
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv("reports/tables/regime_errors.csv", index=False)
    print(f"wrote {len(tab)} rows; models: {sorted(tab['model'].unique())}")
