# Traditional time-series and volatility models — Chamathka

## Baselines
Zero return, historical mean and persistence serve as simple benchmarks. Their purpose is to test whether added model complexity improves out-of-sample accuracy. Insert verified values from `reports/tables/metrics_*.csv` after running the shared evaluator.

## Identification and return forecasts
Training-only market ACF/PACF and AIC/BIC comparisons over p,q <= 5 are produced by `scripts/chamathka_diagnostics.py`. AR(p) and ARMA(p,q) forecasts use recursive multi-step conditional means for 1- and 5-day cumulative log returns. Orders must be fixed using training/validation only, never test1/test2. Selected orders and comparative accuracy remain to be populated from actual output.

## GJR-GARCH
AR(1)-GJR-GARCH(1,1) with standardized Student-t errors models conditional volatility and asymmetric shock response. Daily returns are measured in percent. The five-day expected cumulative return is the sum of forecast conditional means. Its direction probability uses an approximate Student-t distribution and must not be presented as exact. Yearly convergence and coefficient estimates, standardized residual ACF, squared residual ACF, Ljung–Box, ARCH-LM and QQ figures should be reported from verified diagnostic files.

## Stock-level ARX
A separate stock-level regression uses observed stock and same-day market returns to forecast subsequent returns. Report per-stock performance only after running the shared walk-forward evaluator and aggregating its saved predictions.

## Five-day realized variance
The optional volatility analysis compares HAR, GJR-GARCH, Random Forest and XGBoost forecasts of `y_rv_5`, measured as the sum of five subsequent squared percent returns. QLIKE is computed as mean(log(forecast variance) + realized variance / forecast variance). Test windows must not be used for tuning.

## Limitations and results status
No numerical performance claims are made here because the required empirical outputs have not been generated in this environment. Important limitations include illiquidity, stale quotes, segment gaps, parameter instability, forecast-distribution approximation, and the cost of repeated GARCH filtering. Results and interpretations must be filled in only after successful end-to-end execution and review.
