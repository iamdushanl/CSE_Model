# How we work

## Branches and commits
- Never commit to `main`. Branch from an up-to-date `main`: `feat/<name>-<topic>`, `fix/<name>-<topic>`, `docs/<name>-<topic>`.
- Commit small and often; push daily. Message: `type(scope): what changed`
  (types: feat, fix, docs, test, refactor, data). Example: `feat(ts): add GJR-GARCH walk-forward`.
- Open a draft pull request early. One approval from your review pair, tests green, then **squash and merge**.

## Review pairs
Teddy and Chamathka review each other's modelling code. Member 3 and Member 4 review each other. Member 5 reviews figures
and tables; Teddy reviews Member 5's code.

## The three contracts (change only through a pull request Teddy approves)
1. Feature table: columns and rules in `docs/features.md` and `src/features/build.py`.
2. Model interface: `Forecaster` in `src/models/base.py` (`fit(train, horizon)`, `predict(history, test)`).
3. Results files: `results/predictions/*.parquet` and `reports/tables/metrics_<window>.csv`.

## Hard rules (a reviewer will block these)
- No random train/test split. Chronological only.
- Never tune or choose a model by looking at `test1` or `test2` scores. Tune on `val`.
- Features use only data up to day t. The leakage tests in `tests/` must keep passing.
- Fit scalers and imputers on training rows only.
- Numbers in the report come from `reports/tables/*.csv`.
- Seed 42 everywhere (`src/eval/config.py`).

## Notebooks
Logic lives in `src/`. One owner per notebook. Clear outputs before committing (`pip install nbstripout`, then `nbstripout --install`).

## Adding a model
1. Subclass `Forecaster`, set `name` and (if you use extra columns) `features`.
2. Add one line to `src/models/registry.py`.
3. `python run.py models <name>` writes predictions and metrics. Add a test if the model has non-trivial logic.
