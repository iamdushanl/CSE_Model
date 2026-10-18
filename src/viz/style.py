"""Shared figure style.  Owner: Member 4.  Everybody imports this so figures match."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PALETTE = {
    "traditional": "#1f6fb2",   # blue
    "ml": "#d1603d",            # orange
    "baseline": "#7a7a7a",      # grey
    "highlight": "#2a9d8f",     # teal
    "shock": "#c0392b",         # red, for crisis windows
}
FIG_DIR = Path("reports/figures")


def apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (9, 4.5), "figure.dpi": 120, "axes.grid": True,
        "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def save(fig, name: str) -> Path:
    """Save to reports/figures/<name>.png. Use this so the report can find every figure."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
