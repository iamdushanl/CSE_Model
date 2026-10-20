# Exploratory Data Analysis and Market Regimes

*Member 4 (tasks E1 to E5). Every number below comes from `reports/tables/stat_tests.csv`, `yearly_market_stats.csv`, `regime_summary.csv` and `regime_errors.csv`. If the models are re-run, regenerate these tables and update the numbers.*

## 1. Purpose

This section describes the behaviour of the cleaned return series before modelling, and then checks whether forecast accuracy depends on the market regime. The EDA decides which model families are reasonable (for example, whether a volatility model is needed). The regime analysis tests whether a single average accuracy figure hides large differences between calm and stressed periods.

## 2. Data used

The analysis uses the cleaned ordinary-share data (13 stocks) and the equal-weighted market proxy (`MARKET`). Returns are daily log returns in percent. A gap of more than 10 calendar days starts a new segment, and no statistic or rolling window crosses a segment boundary. The three gaps in the market proxy end on 2020-05-11 (52 days), 2022-01-03 (278 days) and 2024-01-02 (642 days). Annual statistics are therefore missing for 2023, and 2021 and 2022 hold only 58 and 57 trading days.

## 3. Price and return behaviour

**Prices are non-stationary, returns are not.** Across the 51 entity-segments tested, the ADF test fails to reject a unit root in log prices in 45, and the KPSS test rejects stationarity in 49. For returns the ADF test rejects the unit root in 50 of 51, and KPSS does not reject stationarity in 44 of 51. This supports the decision (D-001) to forecast returns and direction rather than price levels. The seven KPSS rejections for returns include the long 2009 to 2020 market segment (KPSS 1.28), which spans several economic conditions and probably contains level shifts.

**Returns are heavy-tailed.** The Jarque-Bera test rejects normality in all 51 segments. The market proxy has excess kurtosis of 21.7 (2003 to 2008) and 10.8 (2009 to 2020), and individual stocks reach as high as 37.5 (HAYL). Student-t errors are therefore a more suitable assumption than Gaussian errors for the GARCH model.

**Volatility clusters.** The ARCH-LM test is significant in 32 of 51 segments, and the Ljung-Box test on squared returns in 35 of 51. For the market proxy it is significant in every segment except the short post-reopening segment (2020-05 to 2021-03, 218 days), where it finds no clustering. This is evidence for a conditional-volatility model and for volatility features (`v5`, `v22`).

**Linear autocorrelation is present but small.** The Ljung-Box test on returns is significant in 30 of 51 segments, and for the market proxy in all four segments. This is consistent with thin trading and slow information diffusion in an equal-weighted proxy. It is the reason AR and ARMA models are worth testing, but the 21 segments without significant autocorrelation show it is not universal.

**Annual profile.** Annualised market volatility ranged from 7.7% (2017) to 64.0% (2003, only 43 days) in the table. It was 26.1% in 2020, 11.3% to 12.5% in 2018 to 2019, and 14.2% to 18.0% in 2024 to 2025. The worst years for the proxy were 2008 (total log return -57.4%) and 2022 (-33.7%), and the strongest were 2009 (+75.6%), 2010 (+68.8%) and 2024 (+43.8%). The two evaluation windows therefore cover one stressed period (2020) and one strong recovery period (2024 to 2025), which is worth remembering when generalising.

*Figures:* `eda_price_returns.png`, `eda_distribution.png`, `eda_acf_pacf.png`, `eda_correlation.png`.

## 4. Pre-registered regimes

The regimes were written into `docs/regimes.md` and committed before any model result existed, so they could not be chosen after seeing the results.

1. **Volatility regimes.** Calm, normal and turbulent are the terciles of the 22-day rolling volatility of the market proxy. The cut-offs (0.6127 and 0.9758, daily %) were computed on the training years 2002 to 2017 only and applied unchanged afterwards. The regime of a forecast is the regime on the day it is made, which is known at that time.
2. **Crisis window.** March to December 2020, inside the test1 window.
3. **Post-gap regime.** 2024 to 2025, which is the test2 window.

The regime mix differs strongly between windows:

| Window | Days | Calm | Normal | Turbulent |
|---|---:|---:|---:|---:|
| Train (2002 to 2017) | 3,225 | 33.3% | 33.3% | 33.3% |
| Validation (2018 to 2019) | 481 | 54.1% | 32.8% | 13.1% |
| Test 1 (2020 to Mar 2021) | 245 | 6.5% | 33.1% | 60.4% |
| Test 2 (2024 to 2025) | 456 | 13.4% | 52.4% | 34.2% |

Test 1 contains only 16 calm forecast days, so every calm-regime result for that window is weak evidence. These cells are flagged `low_n` in the table (fewer than 30 dates).

## 5. Forecast accuracy by regime

Table: out-of-sample R² against the zero forecast, market proxy, 1-day horizon, volatility regimes. Positive means better than forecasting zero.

| Window | Model | Calm | Normal | Turbulent |
|---|---|---:|---:|---:|
| Test 1 | Random forest | 0.160* | 0.157 | 0.086 |
| Test 1 | XGBoost | 0.121* | 0.175 | 0.082 |
| Test 1 | AR | 0.121* | 0.146 | 0.020 |
| Test 1 | ARMA | 0.149* | 0.177 | 0.051 |
| Test 1 | Persistence | 0.315* | 0.068 | -0.623 |
| Test 2 | Random forest | 0.277 | 0.113 | 0.055 |
| Test 2 | XGBoost | 0.222 | 0.110 | 0.037 |
| Test 2 | AR | 0.168 | -0.127 | -0.051 |
| Test 2 | ARMA | 0.198 | -0.101 | -0.050 |
| Test 2 | Persistence | 0.051 | -0.508 | -0.809 |

\* fewer than 30 forecast days; treat as indicative only.

Main findings:

- **Predictability falls as volatility rises.** In the 2024 to 2025 window the machine-learning models have their largest gain on calm days and their smallest on turbulent days (random forest 0.277, 0.113, 0.055). The updated AR and ARMA models achieve positive out-of-sample R² in calm conditions, but negative R² in normal and turbulent conditions in Test 2. This indicates that the traditional models are less reliable across changing volatility regimes than Random Forest and XGBoost in this evaluation window. Large shocks appear to carry little forecastable information.
- **Naive persistence is unreliable.** Its R² is strongly negative in normal and turbulent regimes (down to -0.81), so a model that merely repeats yesterday's return should not be used as a benchmark that is easy to beat or a strategy that is safe to follow.
-  **Performance during the COVID-period test window.** Test 1 is dominated by turbulent conditions, representing approximately 60% of forecast dates. At the 1-day horizon, Random Forest and XGBoost achieve positive out-of-sample R² across all three volatility regimes. The updated AR and ARMA models also achieve positive R² across these regimes, although their performance is weaker in turbulent conditions (0.020 and 0.051, respectively). These findings suggest that forecasting performance is sensitive to market volatility. The calm-regime results should be interpreted cautiously because they contain only 16 forecast dates.
- **Gains are small for individual stocks.** For the pooled 13-stock panel the 1-day R² of the machine-learning models is only about 0.02 to 0.06 in all regimes, well below the market-proxy values. The market proxy is smoother and more autocorrelated than single stocks.
- **Direction.** In test 2 the random forest reaches 65.6% accuracy on calm days against an up-day base rate of 59.0%, and XGBoost 68.4% on turbulent days against 61.9%. Because up days are the majority in this window, accuracy should always be read next to the base rate, and AUC (about 0.63 to 0.70 for the machine-learning models) is the better summary.

**Statistical tests.** Each model was compared with the best baseline in the same regime using a Diebold-Mariano test on squared errors. For the 1-day market-proxy forecast in test 2, the random forest is significantly better than the best baseline in calm (p = 0.018) and normal (p = 0.044) regimes but not in the turbulent regime (p = 0.343). These p-values come from many correlated tests and are **not corrected for multiple testing**, so a few significant results are expected by chance. The consistent pattern across regimes is more convincing than any single p-value.

*Figures:* `regime_r2_test1.png`, `regime_r2_test2.png`.

## 6. Limitations specific to this section

- The updated regime analysis includes the baseline models, AR, ARMA, stock ARX, logistic regression, Random Forest and XGBoost. However, GJR-GARCH and LSTM predictions are not currently available in the local prediction files, so these models are excluded from the present regime comparison. The regime analysis should be rerun and this report updated when their predictions become available.
- Test windows are short (245 and 456 days) and the calm regime in test 1 has 16 days.
- Regimes are defined from the market proxy. Individual stocks can be in a different state from the market on the same day.
- The market proxy is an equal-weighted internal series, not the official S&P SL20 index.
- Volatility regimes are a description of conditions, not a trading signal, and no claim of profitability is made here.

## 7. Summary

Prices are non-stationary and returns are close to stationary, heavy-tailed and show volatility clustering, so a conditional-volatility model and Student-t errors are justified. Forecast gains over a zero forecast exist but are modest. They are largest in calm markets and shrink as volatility rises, they are weaker for single stocks than for the market proxy, and they survive the COVID shock in reduced form.