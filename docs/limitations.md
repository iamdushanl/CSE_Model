# Limitations and Improvements

## Limitations

1. **Missing historical data:** The dataset has gaps from April to December 2021 and April 2022 to December 2023. Consequently, the analysis does not cover the full market history, including much of the 2022 economic crisis. Findings may not generalise to those periods.

2. **Corporate actions:** The source data are not fully adjusted for corporate actions. Extreme returns above the project's specified log-return threshold of 0.35 were removed and checked, but this procedure may not eliminate every adjustment-related error.

3. **Survivorship bias:** The analysis includes current index constituents. Companies that left the index or ceased trading may be missing, potentially making historical performance appear more favourable than it was.

4. **Thin trading and stale prices:** Zero-return shares vary substantially across the 13 analysed stocks, from 11.41% for JKH to 39.74% for DIAL. This suggests that trading inactivity may affect some series more strongly than others. However, zero-return share is only a proxy for liquidity; it is not a direct measure of trading volume, bid–ask spread, or market depth.

5. **Modest forecast improvements:** For the evaluated stocks, XGBoost's RMSE improvement over a zero-return benchmark ranges from approximately 0.23% for LIOC to 3.48% for SAMP. These gains are modest and do not, by themselves, establish that a strategy is profitable after trading costs.

6. **Market proxy:** The market series is an internally constructed equal-weighted proxy, not the official S&P SL20 index. Results based on this proxy should not be interpreted as official index forecasts.

7. **Limited out-of-sample evaluation:** The evaluation windows contain approximately 240 and 455 dates. These windows may not represent all market regimes, and statistical tests may have limited power.

8. **Daily data limitations:** Daily closing prices do not provide intraday price movements or reliable high-frequency volatility measures. They also limit the precision of liquidity and execution-cost analysis.

9. **Share selection:** Only ordinary voting shares are included, which limits the coverage and generalisability of the findings.

10. **Strategy backtest assumptions:** The preliminary strategy analysis uses illustrative transaction costs rather than a verified brokerage fee schedule. The prediction dataset contains 5,853 stock-date rows out of 5,915 possible combinations across its 455 dates and 13 stocks. Missing stock-date predictions are not automatically filled with zero returns; stocks without predictions are excluded from selection on those dates. Prediction timing, target alignment, turnover calculations, and the treatment of incomplete coverage still require validation before the strategy's economic performance can be interpreted reliably.

11. **Unverified economic performance:** The strategy's cumulative returns are simulation outputs, not evidence of achievable live investment profits. They should not be reported as realistic investment performance until the return units, forecast timing, missing-data treatment, turnover calculation, and transaction costs have been validated.

## Improvements

- Obtain the missing historical data and test performance across additional market regimes.
- Adjust prices and returns using a verified corporate-actions dataset.
- Include historical index constituents to reduce survivorship bias.
- Add direct liquidity measures such as trading volume, turnover, bid–ask spreads, and zero-trade days where available.
- Compare against suitable naïve benchmarks and assess whether improvements are statistically and economically meaningful.
- Validate prediction-date and target-return alignment and check for look-ahead bias.
- Re-run the strategy without automatically treating missing stock returns as zero; document how missing observations are handled.
- Use realistic transaction fees, taxes, slippage, and execution assumptions relevant to the Colombo Stock Exchange.
- Test robustness across thresholds, periods, stocks, and alternative portfolio-weighting methods.
- Explore intraday or high–low volatility estimators and relevant macroeconomic and foreign-exchange variables.
- Reassess results when longer, consistently adjusted out-of-sample data become available.
