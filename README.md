# Forecasting Sri Lankan equities: traditional time series versus machine learning

Group assignment: *Real-World Financial Time Series Analytics*.
Data: S&P SL20 constituents on the Colombo Stock Exchange (CSE), daily, 2002 to 2025.

**Research question.** How accurately can next-day and 5-day returns, and their direction (up or down),
be forecast for CSE equities with traditional time-series models versus machine-learning models, and how
does the answer change across horizons, stocks and market regimes?

## The project in 60 seconds
1. Clean the messy Excel file (it mixes share classes, has multi-year gaps and unadjusted splits).
2. Build leakage-safe features from past returns and volatility.
3. Forecast with baselines, traditional models (AR, ARMA, GJR-GARCH) and ML models (Logistic, Random Forest, XGBoost, LSTM).
4. Judge **every** model with one shared walk-forward evaluator (chronological splits only).
5. Explain *why* the models differ: regimes, horizons, feature importance, thin trading.

The full explanation, glossary, roles and plan are in the team handbook (`docs/HANDBOOK.md`).

## Quick start
```bash
git clone https://github.com/iamdushanl/CSE_Model.git
cd CSE_Model
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py test      # 16 tests must pass
python run.py data      # ~20 s: clean data, features, audit tables
python run.py models    # runs every model in src/models/registry.py through the evaluator
```
Outputs: `reports/tables/metrics_<window>.csv`, `results/predictions/*.parquet`, `data/processed/*`.

## Layout
| Folder | What | Owner |
| --- | --- | --- |
| `src/data/` | cleaning | Member 3 |
| `src/features/` | feature table (the contract everyone uses) | Member 3 + Teddy |
| `src/eval/` | walk-forward evaluator, metrics, Diebold-Mariano | Teddy |
| `src/models/baselines.py`, `src/models/ts/` | baselines, AR, ARMA, GARCH | Chamathka |
| `src/models/ml/` | Logistic, Random Forest, XGBoost, LSTM | Teddy |
| `src/viz/` | shared figure style | Member 4 |
| `notebooks/` | exploration and figures (one owner each) | see checklist in each notebook |
| `docs/` | decisions, features, regimes, literature, limitations, contribution logs | everyone |
| `reports/`, `report/` | generated tables and figures; report text | Member 5 assembles |

## Evaluation design
Train 2002 to 2017, validation 2018 to 2019 (tune here only), test1 2020 to March 2021 (COVID shock),
test2 2024 to 2025 (new regime). Expanding window, refit once per calendar year. See `src/eval/config.py`.

## Rules
Read `CONTRIBUTING.md`. In short: branch, small commits, pull request with one review, tests green, no random splits, no tuning on test windows.
