"""Volatility regimes for the market proxy.  Owner: Member 4 (task E2).

Regimes are pre-registered in docs/regimes.md. The cut-offs are computed on the
TRAINING period only (2002 to 2017), then applied unchanged to every later date.

Run:  python -m src.eda.regimes
"""
from __future__ import annotations

import pandas as pd

from src.eval.config import WINDOWS

GAP_DAYS = 10                 # same rule as the feature table: a longer gap breaks the series
TRAIN_END = "2017-12-31"
VOL_WINDOW = 22


def load_market(path: str = "data/processed/market_proxy.parquet") -> pd.Series:
    """Market proxy log return in PERCENT, indexed by date."""
    return pd.read_parquet(path)["MARKET"] * 100.0


def segment_ids(index: pd.DatetimeIndex) -> pd.Series:
    """0, 1, 2 ... : a new id starts after every gap longer than GAP_DAYS."""
    s = index.to_series()
    return (s.diff().dt.days > GAP_DAYS).cumsum()


def rolling_vol(r: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Rolling std of returns, never mixing days from before and after a gap."""
    seg = segment_ids(r.index)
    return r.groupby(seg).transform(lambda x: x.rolling(window).std()).rename("vol22")


def tercile_cutoffs(vol: pd.Series, train_end: str = TRAIN_END) -> tuple[float, float]:
    """Calm/normal and normal/turbulent cut-offs from training rows only."""
    train = vol[vol.index <= train_end].dropna()
    lo, hi = train.quantile([1 / 3, 2 / 3])
    return float(lo), float(hi)


def label(vol: pd.Series, lo: float, hi: float) -> pd.Series:
    """'calm', 'normal' or 'turbulent' for each day (NaN where vol is missing)."""
    out = pd.Series(pd.NA, index=vol.index, dtype="object")
    out[vol <= lo] = "calm"
    out[(vol > lo) & (vol <= hi)] = "normal"
    out[vol > hi] = "turbulent"
    return out.rename("regime")


def summary(vol: pd.Series, lo: float, hi: float) -> pd.DataFrame:
    """Share of calm / normal / turbulent days inside each evaluation window."""
    reg = label(vol, lo, hi)
    rows = [{"window": "train", "start": "2002-01-01", "end": TRAIN_END}]
    rows += [{"window": k, "start": v["test_start"], "end": v["test_end"]} for k, v in WINDOWS.items()]
    out = []
    for w in rows:
        x = reg[w["start"]:w["end"]].dropna()
        shares = x.value_counts(normalize=True).reindex(["calm", "normal", "turbulent"]).fillna(0)
        out.append({**w, "n_days": len(x), **shares.round(3).to_dict(), "cut_lo": round(lo, 4), "cut_hi": round(hi, 4)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    from pathlib import Path
    r = load_market()
    v = rolling_vol(r)
    lo, hi = tercile_cutoffs(v)
    table = summary(v, lo, hi)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    table.to_csv("reports/tables/regime_summary.csv", index=False)
    print(f"cut-offs (training only): calm <= {lo:.4f} < normal <= {hi:.4f} < turbulent")
    print(table.to_string(index=False))