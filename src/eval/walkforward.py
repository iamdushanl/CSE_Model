"""Walk-forward evaluator.  Owner: Teddy.  Every model is judged by this code only.

For each calendar year inside a test window:
    train = all rows dated BEFORE that year (expanding window), purged of the last
            `horizon` rows per entity so no training target peeks into the test year;
    fit once, predict that year, move on.
Hyper-parameters stay frozen from the validation step; this loop never tunes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.config import BASELINE_NAMES, TRAIN_START, WINDOWS
from src.eval.dm import dm_test
from src.eval.metrics import summarise
from src.features.build import BASE_FEATURES, TARGETS


def _entity_mask(table: pd.DataFrame, entity_set) -> pd.Series:
    if entity_set == "MARKET":
        return table["entity"] == "MARKET"
    if entity_set == "PANEL":
        return table["entity"] != "MARKET"
    if isinstance(entity_set, (list, tuple, set)):
        return table["entity"].isin(entity_set)
    raise ValueError(f"unknown entity_set {entity_set!r}")


def _purge(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Drop the last `horizon` rows of every entity (their targets reach into the test year)."""
    rank_from_end = train.groupby("entity").cumcount(ascending=False)
    return train[rank_from_end.to_numpy() >= horizon]


def walk_forward(make_model, table: pd.DataFrame, window, horizon: int = 1,
                 entity_set="MARKET", windows: dict = WINDOWS) -> pd.DataFrame:
    """Run one model over one window. Returns one row per (date, entity) forecast."""
    w = windows[window] if isinstance(window, str) else window
    t0, t1 = pd.Timestamp(w["test_start"]), pd.Timestamp(w["test_end"])
    sub = table[_entity_mask(table, entity_set)]
    sub = sub.dropna(subset=BASE_FEATURES).sort_index(kind="stable")
    ycol, dcol = f"y_ret_{horizon}", f"y_dir_{horizon}"
    frames, model_name = [], None
    for year in range(t0.year, t1.year + 1):
        cut = pd.Timestamp(f"{year}-01-01")
        lo, hi = max(cut, t0), min(pd.Timestamp(f"{year}-12-31"), t1)
        test = sub[(sub.index >= lo) & (sub.index <= hi)]
        train = sub[(sub.index >= TRAIN_START) & (sub.index < cut)]
        if test.empty or train.empty:
            continue
        model = make_model()
        model_name = model.name
        extra = [c for c in getattr(model, "features", []) if c not in BASE_FEATURES]
        if extra:                                  # rows must be complete for the model's own columns
            train, test = train.dropna(subset=extra), test.dropna(subset=extra)
            if test.empty or train.empty:
                continue
        train = _purge(train, horizon).dropna(subset=[ycol])
        model.fit(train, horizon)
        pred = model.predict(train, test.drop(columns=TARGETS, errors="ignore"))
        if len(pred) != len(test):
            raise ValueError(f"{model.name}: predict returned {len(pred)} rows for {len(test)}")
        frames.append(pd.DataFrame({
            "date": test.index, "entity": test["entity"].to_numpy(),
            "y_ret": test[ycol].to_numpy(), "y_dir": test[dcol].to_numpy(),
            "y_pred": pred["y_pred"].to_numpy(), "p_up": pred["p_up"].to_numpy(),
            "fit_year": year}))
    if not frames:
        return pd.DataFrame(columns=["date", "entity", "y_ret", "y_dir", "y_pred", "p_up",
                                     "fit_year", "model", "horizon", "window", "entity_set"])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["y_ret"])                       # cannot score without a target
    out["model"], out["horizon"] = model_name, horizon
    out["window"] = window if isinstance(window, str) else "custom"
    out["entity_set"] = entity_set if isinstance(entity_set, str) else "CUSTOM"
    return out


def save_predictions(pred: pd.DataFrame, out_dir: str | Path = "results/predictions") -> Path:
    """results/predictions/<model>__h<horizon>__<window>__<entity_set>.parquet"""
    r = pred.iloc[0]
    path = Path(out_dir) / f"{r['model']}__h{int(r['horizon'])}__{r['window']}__{r['entity_set']}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(path, index=False)
    return path


def load_all_predictions(pred_dir: str | Path = "results/predictions") -> pd.DataFrame:
    files = sorted(Path(pred_dir).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def metrics_table(preds: pd.DataFrame, baselines=BASELINE_NAMES) -> pd.DataFrame:
    """One row per (window, entity_set, horizon, model) with metrics and DM test.

    The DM test compares each model with the BEST baseline (lowest RMSE) in its group,
    on squared errors averaged per date. dm_stat < 0 means the model is better.
    """
    rows = []
    for (win, es, h), g in preds.groupby(["window", "entity_set", "horizon"]):
        stats_by_model = {m: summarise(x) for m, x in g.groupby("model")}
        base = [b for b in baselines if b in stats_by_model and "rmse" in stats_by_model[b]]
        best = min(base, key=lambda b: stats_by_model[b]["rmse"]) if base else None

        def loss(model):
            x = g[g.model == model].dropna(subset=["y_ret", "y_pred"])
            return ((x.y_ret - x.y_pred) ** 2).groupby(x.date).mean()

        for m, st in stats_by_model.items():
            row = {"window": win, "entity_set": es, "horizon": int(h), "model": m, **st,
                   "best_baseline": best}
            if best and m != best and "rmse" in st:
                a, b = loss(m).align(loss(best), join="inner")
                row["dm_stat"], row["dm_p"] = dm_test(a.to_numpy(), b.to_numpy(), h=int(h))
            rows.append(row)
    cols = ["window", "entity_set", "horizon", "model", "n", "rmse", "mae", "oos_r2", "n_dir",
            "accuracy", "base_rate_up", "balanced_acc", "auc", "logloss", "best_baseline",
            "dm_stat", "dm_p"]
    df = pd.DataFrame(rows)
    return df.reindex(columns=cols).sort_values(["window", "entity_set", "horizon", "model"])


def write_metrics(preds: pd.DataFrame, out_dir: str | Path = "reports/tables") -> list[Path]:
    """reports/tables/metrics_<window>.csv"""
    mt = metrics_table(preds)
    paths = []
    for win, g in mt.groupby("window"):
        p = Path(out_dir) / f"metrics_{win}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        g.round(5).to_csv(p, index=False)
        paths.append(p)
    return paths
