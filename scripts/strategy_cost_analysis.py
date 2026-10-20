from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("results/predictions/xgboost__h1__test2__PANEL.parquet")
OUT = Path("reports/tables/strategy_cost_sensitivity.csv")

# Illustrative assumptions, NOT verified brokerage fees.
THRESHOLDS = [0.50, 0.55, 0.60]
COSTS_BPS = [0, 10, 25, 50]


def main():
    df = pd.read_parquet(INPUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["y_ret", "p_up"]).copy()
    df = df.sort_values(["date", "entity"])

    # y_ret is a log return expressed in percent.
    # Convert it to a simple return for portfolio calculations.
    df["simple_return"] = np.expm1(df["y_ret"] / 100.0)

    results = []

    for threshold in THRESHOLDS:
        data = df.copy()
        data["selected"] = data["p_up"] >= threshold

        # Equal-weight selected stocks at each forecast origin.
        data["weight"] = 0.0
        selected_counts = data.groupby("date")["selected"].transform("sum")
        mask = data["selected"] & (selected_counts > 0)
        data.loc[mask, "weight"] = 1.0 / selected_counts[mask]

        weights = data.pivot(
            index="date", columns="entity", values="weight"
        ).fillna(0.0).sort_index()

        returns = data.pivot(
            index="date", columns="entity", values="simple_return"
        ).sort_index()

        # Align realized returns with the portfolio weights.
        # Do not silently replace missing returns with zero.
        returns = returns.reindex(
            index=weights.index, columns=weights.columns
        )

        # A selected stock must have a valid realized return.
        missing_selected_returns = (weights > 0) & returns.isna()

        if missing_selected_returns.any().any():
            count = int(missing_selected_returns.sum().sum())
            raise ValueError(
                f"Backtest stopped: {count} selected stock-date "
                "positions have missing realized returns."
            )

        # Missing returns for zero-weight stocks do not affect the portfolio.
        gross = (weights * returns).sum(axis=1)

        # Trade volume includes buys and sells. First day starts in cash.
        previous = weights.shift(1).fillna(0.0)
        turnover = (weights - previous).abs().sum(axis=1)

        for cost_bps in COSTS_BPS:
            cost_rate = cost_bps / 10000.0
            net = gross - turnover * cost_rate

            results.append({
                "threshold_p_up": threshold,
                "cost_bps_per_unit_turnover": cost_bps,
                "forecast_dates": len(net),
                "active_dates": int((weights.sum(axis=1) > 0).sum()),
                "mean_stocks_held": float(
                    (weights > 0).sum(axis=1).mean()
                ),
                "mean_turnover": float(turnover.mean()),
                "mean_gross_return_pct": float(gross.mean() * 100),
                "mean_net_return_pct": float(net.mean() * 100),
                "cumulative_net_return_pct": float(
                    (np.prod(1 + net) - 1) * 100
                ),
                "positive_net_period_share_pct": float(
                    (net > 0).mean() * 100
                ),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(results)
    result.to_csv(OUT, index=False)

    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved results to: {OUT}")


if __name__ == "__main__":
    main()
