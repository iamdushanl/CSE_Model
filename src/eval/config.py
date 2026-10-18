"""One place for the evaluation design (handbook section 4).  Owner: Teddy.

Change anything here only through a pull request that Teddy approves.
"""
SEED = 42
TRAIN_START = "2002-01-01"

# Each window is scored with an expanding training set, refit once per calendar
# year: the model for year Y is trained on all rows dated before Y-01-01.
WINDOWS = {
    "val":   {"test_start": "2018-01-01", "test_end": "2019-12-31"},   # tune here only
    "test1": {"test_start": "2020-01-01", "test_end": "2021-03-31"},   # COVID shock
    "test2": {"test_start": "2024-01-01", "test_end": "2025-12-31"},   # new regime
}
HORIZONS = (1, 5)
ENTITY_SETS = ("MARKET", "PANEL")          # equal-weighted proxy, pooled 13 stocks
BASELINE_NAMES = ("zero", "hist_mean", "persistence")
