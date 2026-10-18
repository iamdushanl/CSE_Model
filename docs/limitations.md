# Limitations and improvements (Member 5, all contribute)
Seed list. Add evidence (numbers from `reports/tables/`) as results arrive.

- No data for 2021-04 to 2021-12 and 2022-04 to 2023-12, so the 2022 crisis is missing.
- Corporate actions are not adjusted in the source. We removed returns above 0.35 (log) and checked them (D3).
- Survivorship bias: only current index members are included.
- Thin trading and stale prices create autocorrelation that may not be tradable.
- The market series is our own equal-weighted proxy, not the official S&P SL20.
- Small test windows (about 240 and 455 days), so significance tests have limited power.
- Daily closes only: no intraday data, so volatility proxies are noisy.
- Only ordinary voting shares are used.
Improvements: adjust splits using an external corporate-actions file, add intraday or high-low volatility
estimators, add macro and FX variables, longer out-of-sample history once the missing years are sourced.
