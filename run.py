"""Task runner that works on Windows, macOS and Linux (no `make` needed).

    python run.py data        clean the raw file, write data/processed and audit tables
    python run.py models      run every registered model through the walk-forward evaluator
    python run.py metrics     rebuild reports/tables/metrics_*.csv from saved predictions
    python run.py test        run the unit tests
    python run.py all         data, models, metrics
"""
from __future__ import annotations

import subprocess
import sys


def task_data() -> None:
    from src.data.clean import build_processed
    info = build_processed()
    print("core tickers:", ", ".join(info["core_tickers"]))
    print({k: v for k, v in info.items() if k != "core_tickers"})


def task_models(only: list[str] | None = None) -> None:
    from src.eval.config import ENTITY_SETS, HORIZONS, WINDOWS
    from src.eval.walkforward import save_predictions, walk_forward
    from src.features.build import load_features
    from src.models.registry import MODELS

    table = load_features()
    names = only or list(MODELS)
    for name in names:
        factory = MODELS[name]
        for window in WINDOWS:
            for es in (("MARKET",) if name in {"ar", "arma", "ar_gjr_garch_t"} else ("PANEL",) if name == "stock_arx" else ENTITY_SETS):
                for h in HORIZONS:
                    pred = walk_forward(factory, table, window, h, es)
                    if len(pred):
                        save_predictions(pred)
        print("done:", name)
    task_metrics()


def task_metrics() -> None:
    from src.eval.walkforward import load_all_predictions, write_metrics
    preds = load_all_predictions()
    if preds.empty:
        print("no predictions yet: run `python run.py models`")
        return
    for p in write_metrics(preds):
        print("wrote", p)


def task_test() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q"])


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "data":
        task_data()
    elif cmd == "models":
        task_models(rest or None)
    elif cmd == "metrics":
        task_metrics()
    elif cmd == "test":
        return task_test()
    elif cmd == "all":
        task_data()
        task_models()
    else:
        print(f"unknown task {cmd!r}\n{__doc__}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
