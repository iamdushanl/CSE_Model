"""Teddy, Tasks T6 & T7: ML Interpretability (SHAP, Feature Importance) and Stability Audit.

Produces:
  - reports/figures/ml_val_comparison.png
  - reports/figures/ml_val_direction.png
  - reports/figures/ml_shap_summary.png
  - reports/figures/ml_feature_importance.png
  - reports/tables/ml_stability_audit.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.eval.config import SEED, WINDOWS
from src.eval.metrics import summarise
from src.eval.walkforward import walk_forward
from src.features.build import load_features
from src.models.ml.logistic import LogisticForecaster
from src.models.ml.random_forest import RandomForestForecaster
from src.models.ml.xgboost_model import XGBoostForecaster
from src.viz.style import PALETTE, apply_style, save


def plot_validation_comparison():
    apply_style()
    val_path = Path("reports/tables/metrics_val.csv")
    if not val_path.exists():
        print("metrics_val.csv not found, skipping validation comparison plot")
        return
    df = pd.read_csv(val_path)

    # Filter to MARKET and PANEL h=1
    for es in ("MARKET", "PANEL"):
        sub = df[(df["entity_set"] == es) & (df["horizon"] == 1)].dropna(subset=["oos_r2"])
        if sub.empty:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        models = sub["model"].tolist()
        r2s = sub["oos_r2"].tolist()
        colors = [PALETTE["ml"] if m in ("logistic", "random_forest", "xgboost")
                  else PALETTE["baseline"] for m in models]

        ax1.barh(models, r2s, color=colors)
        ax1.axvline(0, color="grey", linestyle="--", alpha=0.7)
        ax1.set_xlabel("Out-of-Sample $R^2$")
        ax1.set_title(f"Forecast Accuracy ({es}, h=1)")

        sub_dir = df[(df["entity_set"] == es) & (df["horizon"] == 1)].dropna(subset=["accuracy"])
        if not sub_dir.empty:
            dir_models = sub_dir["model"].tolist()
            accs = (sub_dir["accuracy"] * 100).tolist()
            base_rate = sub_dir["base_rate_up"].iloc[0] * 100
            dir_colors = [PALETTE["ml"] if m in ("logistic", "random_forest", "xgboost")
                          else PALETTE["baseline"] for m in dir_models]
            ax2.barh(dir_models, accs, color=dir_colors)
            ax2.axvline(base_rate, color=PALETTE["shock"], linestyle="--", label=f"Base Rate ({base_rate:.1f}%)")
            ax2.set_xlabel("Directional Accuracy (%)")
            ax2.set_title(f"Direction Prediction ({es}, h=1)")
            ax2.legend(loc="lower right")

        plt.tight_layout()
        save(fig, f"ml_val_comparison_{es.lower()}")
        plt.close(fig)
        print(f"Saved reports/figures/ml_val_comparison_{es.lower()}.png")


def run_shap_analysis():
    apply_style()
    print("Running SHAP analysis on XGBoost...")
    feats = load_features("data/processed/features.parquet")
    mkt = feats[feats["entity"] == "MARKET"].sort_index()

    # Train on rows before 2018-01-01
    train = mkt[mkt.index < "2018-01-01"].dropna(subset=["y_ret_1"])
    test = mkt[(mkt.index >= "2018-01-01") & (mkt.index <= "2019-12-31")].dropna(subset=["y_ret_1"])

    model = XGBoostForecaster(random_state=SEED)
    model.fit(train, horizon=1)

    X_train = train[model.feature_cols_]
    X_test = test[model.feature_cols_]

    explainer = shap.TreeExplainer(model.regressor_)
    shap_values = explainer(X_test)

    # Plot SHAP summary (beeswarm)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("XGBoost Return Forecast: SHAP Feature Attributions (Validation 2018-2019)", fontsize=11)
    plt.tight_layout()
    save(fig, "ml_shap_summary")
    plt.close(fig)
    print("Saved reports/figures/ml_shap_summary.png")

    # Plot Random Forest Feature Importance (MDI)
    rf_model = RandomForestForecaster(random_state=SEED)
    rf_model.fit(train, horizon=1)
    if rf_model.feature_importances_ is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        imp = rf_model.feature_importances_.sort_values(ascending=True)
        imp.plot(kind="barh", color=PALETTE["ml"], ax=ax)
        ax.set_title("Random Forest MDI Feature Importance (Market Proxy h=1)", fontsize=11)
        ax.set_xlabel("Mean Decrease in Impurity")
        plt.tight_layout()
        save(fig, "ml_feature_importance")
        plt.close(fig)
        print("Saved reports/figures/ml_feature_importance.png")


def run_stability_audit():
    print("Running Stability Audit across 5 random seeds (Task T7)...")
    feats = load_features("data/processed/features.parquet")
    seeds = [42, 100, 2024, 777, 999]
    records = []

    for seed in seeds:
        for model_cls, model_name in [
            (RandomForestForecaster, "random_forest"),
            (XGBoostForecaster, "xgboost"),
        ]:
            def make_model():
                return model_cls(random_state=seed)

            pred = walk_forward(make_model, feats, window="val", horizon=1, entity_set="MARKET")
            stats = summarise(pred)
            records.append({
                "seed": seed,
                "model": model_name,
                "rmse": stats.get("rmse"),
                "oos_r2": stats.get("oos_r2"),
                "accuracy": stats.get("accuracy"),
                "auc": stats.get("auc"),
            })

    audit_df = pd.DataFrame(records)
    out_path = Path("reports/tables/ml_stability_audit.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    # Summary table
    grouped = audit_df.groupby("model")[["rmse", "oos_r2", "accuracy", "auc"]].agg(["mean", "std"])
    print("\nStability Audit Summary across Seeds (Mean ± Std):")
    print(grouped.round(5))


if __name__ == "__main__":
    plot_validation_comparison()
    run_shap_analysis()
    run_stability_audit()
