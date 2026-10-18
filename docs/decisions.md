# Decision log
Add one row per decision: date, who, decision, reason. Newest at the bottom.

| ID | Date | Who | Decision | Reason |
| --- | --- | --- | --- | --- |
| D-001 | 2026-10-18 | Teddy | Forecast returns and direction (1-day and 5-day), not the price level | Prices are non-stationary; any model looks 99% accurate on levels |
| D-002 | 2026-10-18 | Teddy | Use `MainType == N` rows only | The sheet interleaves share classes; its return and volatility columns are wrong |
| D-003 | 2026-10-18 | Teddy | Gap above 10 calendar days starts a new segment | Features must never cross the 2020 closure or the 2021 to 2023 holes |
| D-004 | 2026-10-18 | Teddy | Returns above 0.35 in absolute log terms are treated as corporate actions | Unadjusted splits (for example JKH 2024). Member 3 verifies by hand (task D3) |
| D-005 | 2026-10-18 | Teddy | Core universe = 13 stocks with at least 93% ordinary-share coverage | Others lack too many ordinary-share rows |
| D-006 | 2026-10-18 | Teddy | Windows: train 2002 to 2017, validation 2018 to 2019, test1 2020 to Mar 2021, test2 2024 to 2025 | See `src/eval/config.py` |
