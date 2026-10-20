"""Forecast horizon and liquidity analysis."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.features.build import load_features
from src.eval.metrics import summarise
from src.viz.style import apply_style, save, PALETTE

ROOT = Path(".")
PRED_DIR = ROOT / "results" / "predictions"
TABLE_DIR = ROOT / "reports" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
apply_style()


def read_predictions(model, horizon, window, entity_set):
    path = PRED_DIR / (
        f"{model}__h{horizon}__{window}__{entity_set}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    return pd.read_parquet(path)


def horizon_analysis():
    """Compare return and direction forecasting across horizons/windows."""
    rows = []

    for window in ("val", "test1", "test2"):
        for entity_set in ("MARKET", "PANEL"):
            for horizon in (1, 5):
                for model in ("zero", "random_forest", "xgboost"):
                    pred = read_predictions(
                        model, horizon, window, entity_set
                    )
                    row = {
                        "window": window,
                        "entity_set": entity_set,
                        "horizon": horizon,
                        "model": model,
                    }
                    row.update(summarise(pred))
                    rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(
        TABLE_DIR / "forecast_horizon_metrics.csv", index=False
    )

    # Visualise out-of-sample R-squared for XGBoost.
    xgb = results.loc[results["model"] == "xgboost"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, entity_set in zip(axes, ("MARKET", "PANEL")):
        subset = xgb.loc[xgb["entity_set"] == entity_set]
        for window in ("val", "test1", "test2"):
            part = subset.loc[subset["window"] == window].sort_values(
                "horizon"
            )
            if not part.empty and "oos_r2" in part:
                ax.plot(
                    part["horizon"],
                    part["oos_r2"],
                    marker="o",
                    label=window,
                )

        ax.axhline(0, color=PALETTE["baseline"], linestyle="--")
        ax.set_title(f"{entity_set}: XGBoost")
        ax.set_xlabel("Forecast horizon (trading days)")
        ax.set_xticks([1, 5])
        ax.legend(title="Evaluation window")

    axes[0].set_ylabel("Out-of-sample R-squared")
    fig.suptitle("Forecast performance across horizons")
    fig.tight_layout()
    save(fig, "forecast_horizon_predictability")
    plt.close(fig)

    print("\nHorizon metrics saved.")
    print(xgb[[
        "window", "entity_set", "horizon",
        "n", "rmse", "oos_r2", "accuracy", "auc"
    ]].to_string(index=False))


def liquidity_analysis():
    """Relate stock-level trading characteristics to test2 forecast gain."""
    features = load_features("data/processed/features.parquet")
    features = features.reset_index()
    features["date"] = pd.to_datetime(features["date"])

    # Use stock returns only; MARKET is not an individual company.
    stocks = features.loc[
        features["entity"] != "MARKET"
    ].copy()

    liquidity_rows = []
    for ticker, group in stocks.groupby("entity"):
        valid_r = group.loc[group["r"].notna()].sort_values("date")
        if valid_r.empty:
            continue

        # Estimate lag-1 autocorrelation within segments so gaps do not
        # create artificial adjacent observations.
        correlations = []
        for _, segment in valid_r.groupby("seg"):
            pair = segment["r"].dropna()
            if len(pair) >= 3:
                corr = pair.autocorr(lag=1)
                if pd.notna(corr):
                    correlations.append(corr)

        liquidity_rows.append({
            "entity": ticker,
            "n_returns": int(valid_r["r"].notna().sum()),
            "zero_return_share": float(
                (valid_r["r"] == 0).mean()
            ),
            "lag1_autocorrelation": (
                float(np.mean(correlations)) if correlations else np.nan
            ),
        })

    liquidity = pd.DataFrame(liquidity_rows)

    # Calculate per-stock RMSE gains on the untouched 2024–2025 test2 set.
    # Positive gain means XGBoost has lower RMSE than the zero forecast.
    xgb = read_predictions("xgboost", 1, "test2", "PANEL")
    zero = read_predictions("zero", 1, "test2", "PANEL")

    def per_stock_rmse(frame, name):
        rows = []
        for ticker, group in frame.groupby("entity"):
            valid = group.dropna(subset=["y_ret", "y_pred"])
            if len(valid):
                rmse = float(np.sqrt(
                    np.mean((valid["y_ret"] - valid["y_pred"]) ** 2)
                ))
                rows.append({
                    "entity": ticker,
                    name: rmse,
                    f"{name}_n": len(valid),
                })
        return pd.DataFrame(rows)

    xgb_rmse = per_stock_rmse(xgb, "xgb_rmse")
    zero_rmse = per_stock_rmse(zero, "zero_rmse")

    gain = xgb_rmse.merge(zero_rmse, on="entity", how="outer")
    gain["rmse_gain"] = gain["zero_rmse"] - gain["xgb_rmse"]
    gain["rmse_gain_pct"] = (
        100 * gain["rmse_gain"] / gain["zero_rmse"].replace(0, np.nan)
    )

    summary = liquidity.merge(gain, on="entity", how="left")
    summary.to_csv(
        TABLE_DIR / "liquidity_by_stock.csv", index=False
    )

    # Scatter plot: zero-return share versus relative RMSE gain.
    plot_data = summary.dropna(
        subset=["zero_return_share", "rmse_gain_pct"]
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        plot_data["zero_return_share"],
        plot_data["rmse_gain_pct"],
        color=PALETTE["ml"],
        alpha=0.85,
    )

    for _, row in plot_data.iterrows():
        ax.annotate(
            row["entity"],
            (row["zero_return_share"], row["rmse_gain_pct"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    # Add a fitted linear trend only when enough valid stocks exist.
    if len(plot_data) >= 3 and plot_data["zero_return_share"].nunique() >= 2:
        slope, intercept = np.polyfit(
            plot_data["zero_return_share"],
            plot_data["rmse_gain_pct"],
            1,
        )
        xs = np.linspace(
            plot_data["zero_return_share"].min(),
            plot_data["zero_return_share"].max(),
            100,
        )
        ax.plot(
            xs,
            slope * xs + intercept,
            color=PALETTE["traditional"],
            label="Linear trend",
        )
        ax.legend()

    ax.axhline(0, color=PALETTE["baseline"], linestyle="--")
    ax.set_xlabel("Share of observed returns equal to zero")
    ax.set_ylabel("XGBoost RMSE improvement over zero forecast (%)")
    ax.set_title("Liquidity proxy and stock-level forecast improvement")
    save(fig, "liquidity_vs_forecast_gain")
    plt.close(fig)

    print("\nLiquidity table saved.")
    print(summary.sort_values(
        "zero_return_share", ascending=False
    ).round(4).to_string(index=False))


if __name__ == "__main__":
    horizon_analysis()
    liquidity_analysis()
    print("\nForecast horizon and liquidity analysis completed successfully.")
    print("Tables: reports/tables/forecast_horizon_metrics.csv, reports/tables/liquidity_by_stock.csv")
    print("Figures: reports/figures/forecast_horizon_predictability.png, reports/figures/liquidity_vs_forecast_gain.png")
