# Feature list (agreed by Teddy and Member 3)
All features use information up to and including day t. Returns are in percent.

| Feature | Definition | Why |
| --- | --- | --- |
| `r` | today's log return | Latest information, lag 0 |
| `l1` to `l4` | returns 1 to 4 days ago | Autocorrelation (lag-1 autocorrelation of the market proxy is about 0.15) |
| `m5`, `m22` | mean return over 5 and 22 days | Short and medium momentum |
| `v5`, `v22` | standard deviation over 5 and 22 days | Volatility clustering (ARCH-LM) |
| `abs_r` | absolute return today | Size of the latest shock |
| `zero_ret` | 1 if today's return is exactly 0 | Stale price signal for thin trading |
| `dow` | weekday 0 to 4 | Calendar effect (test it, do not assume it) |
| `mkt` | same-day market proxy return (stocks only) | Systematic component of a stock's move |

Ideas to evaluate (add through a pull request, with a leakage justification): log volume change,
turnover share, 22-day mean of zero_ret, previous-day market return.
