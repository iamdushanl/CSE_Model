# Regime definitions (Member 4) - COMMIT THIS BEFORE ANY MODEL RESULTS EXIST
Pre-registering the regimes stops anyone cherry-picking them after seeing results.

1. Crisis windows (calendar): COVID closure and reopening, 2020-03 to 2020-12.
2. Volatility regimes: calm / normal / turbulent = terciles of rolling 22-day volatility of the market proxy,
   with the cut-offs computed on the TRAINING period only (2002 to 2017):
   calm <= 0.6127, normal <= 0.9758, turbulent above (daily %, see `reports/tables/regime_summary.csv`).
   Code: `src/eda/regimes.py`.
3. Post-gap regime: 2024 to 2025.
4. Rationale for each choice:
   - Crisis window: the 2020 market closure and reopening is the largest shock in the test data.
   - Volatility terciles: volatility clusters, so splitting by recent volatility separates quiet from stressed markets without using any future data.
   - Post-gap regime: trading resumes in 2024 after a 642-day data gap, so the market may behave differently.

Date committed: 2026-10-20