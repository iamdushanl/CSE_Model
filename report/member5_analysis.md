# Member 5: Forecast Horizon, Liquidity and Strategy Cost Analysis

## 1. Objective

This analysis examines whether forecast performance changes across one-day
and five-day horizons, whether predictive performance varies across stocks
with different zero-return frequencies, and how transaction costs affect
an illustrative forecast-based trading strategy.

## 2. Forecast Horizon Analysis

The evaluation compares Random Forest (RF) and XGBoost (XGB) against a
zero-return benchmark. Results are considered separately for the market-level
series and the stock panel. RMSE values should be compared against the
corresponding benchmark within the same dataset and forecast horizon.

### 2.1 Test 2 results

| Dataset | Horizon | Model | RMSE | OOS R-squared | Direction accuracy | AUC |
|---|---|---|---:|---:|---:|---:|
| Market | 1 day | RF | 0.9932 | 0.0967 | 60.22% | 0.6456 |
| Market | 1 day | XGB | 0.9971 | 0.0896 | 62.86% | 0.6496 |
| Market | 5 days | RF | 2.7717 | 0.1269 | 63.42% | 0.6406 |
| Market | 5 days | XGB | 2.7498 | 0.1407 | 65.41% | 0.6370 |
| Panel | 1 day | RF | 1.6325 | 0.0384 | 56.89% | 0.5970 |
| Panel | 1 day | XGB | 1.6288 | 0.0428 | 56.87% | 0.5984 |
| Panel | 5 days | RF | 4.0705 | 0.0533 | 56.52% | 0.5846 |
| Panel | 5 days | XGB | 4.0703 | 0.0534 | 57.02% | 0.5877 |

The market-level models show stronger test-period performance than the
stock-panel models on the reported direction and discrimination metrics.
For the market series, XGBoost achieves the highest reported five-day
direction accuracy (65.41%) and an out-of-sample R-squared of approximately
0.141. For the panel, improvements are smaller, with XGBoost's five-day
direction accuracy at approximately 57.02%.

The results differ by horizon and metric. For example, on the market-level
one-day forecast, Random Forest has a slightly lower RMSE than XGBoost,
while XGBoost has a higher AUC. Therefore, no single model dominates every
metric.

Five-day RMSE values should not be compared directly with one-day RMSE
values because the target horizons differ. These findings describe the
evaluated test windows and do not establish that the same performance will
continue in future periods.

## 3. Liquidity Analysis

Zero-return share is used as a proxy for infrequent trading. It is not a
direct measure of liquidity. The table compares this proxy with the
reported XGBoost RMSE gain relative to the zero-return forecast baseline.

| Stock | Zero-return share | RMSE gain (%) |
|---|---:|---:|
| AEL | 18.22% | 1.53% |
| CFIN | 18.39% | 2.61% |
| DFCC | 11.51% | 2.19% |
| DIAL | 39.74% | 1.46% |
| HAYL | 16.76% | 1.45% |
| HHL | 17.53% | 3.28% |
| JKH | 11.41% | 2.36% |
| LIOC | 18.50% | 0.23% |
| MELS | 17.84% | 2.45% |
| NDB | 12.03% | 3.30% |
| NTB | 20.25% | 2.70% |
| PLC | 27.13% | 1.53% |
| SAMP | 11.82% | 3.48% |

All 13 stocks have a positive reported RMSE gain relative to the zero-return
benchmark in this evaluation. The gain ranges from 0.23% for LIOC to 3.48%
for SAMP. DIAL has the highest zero-return share (39.74%), while JKH has
the lowest (11.41%).

These results suggest that predictive gains are modest and vary across
stocks. They do not by themselves establish a causal relationship between
liquidity and predictability. Zero returns may reflect infrequent trading,
but they can also occur when prices genuinely remain unchanged. Future work
should use direct liquidity measures such as trading volume, bid-ask spreads
and market depth where reliable data are available.

## 4. Strategy Cost Sensitivity

The illustrative strategy varies the predicted upward-movement probability
threshold and the assumed transaction cost per unit of turnover. All
scenarios use 455 forecast dates.

| Probability threshold | Mean turnover | Mean gross return (%) | Mean net return at 10 bps (%) | At 25 bps (%) | At 50 bps (%) |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.7992 | 0.2689 | 0.1890 | 0.0691 | -0.1307 |
| 0.55 | 0.5402 | 0.1766 | 0.1226 | 0.0416 | -0.0935 |
| 0.60 | 0.2525 | 0.0755 | 0.0503 | 0.0124 | -0.0507 |

Mean net return decreases as assumed transaction costs increase. At 50
basis points, all three probability thresholds have negative mean net
returns. Raising the probability threshold reduces mean turnover and
exposure, but also reduces the reported gross return.

For the 0.50 threshold, the reported cumulative net return is 131.97% at
10 basis points and -45.97% at 50 basis points. These cumulative figures
are highly sensitive to assumptions and must not be interpreted as
verified achievable investment returns without further validation.

Before drawing economic conclusions, the strategy requires checks of
forecast availability at decision time, target and return alignment,
turnover and position calculations, treatment of overlapping five-day
returns, execution assumptions, and the compounding method. Transaction
cost assumptions should also be compared with realistic fees, taxes,
slippage and market impact for the relevant stocks and trading sizes.

## 5. Limitations

1. Zero-return share is only a proxy for liquidity and may not distinguish
   infrequent trading from genuinely unchanged prices.
2. Forecast gains relative to the zero-return baseline are generally modest.
3. Performance varies by dataset, horizon, model and evaluation metric.
4. Results from the selected test periods may not generalise to other market
   regimes or future observations.
5. The strategy analysis depends on assumed probability thresholds and
   transaction costs; cumulative results require methodological verification.
6. Missing stock-date predictions and differences in evaluation coverage
   may affect comparability across stocks and models.
7. Historical share-price adjustments, corporate actions and data coverage
   can affect the reliability of calculated returns.
8. Zero-return forecasting benchmarks and model RMSE values must be
   calculated on the same observations for valid comparisons.

## 6. Recommended Improvements

- Validate forecast timing and ensure no future information enters model
  features or trading decisions.
- Recheck target alignment, turnover, overlapping return horizons and
  compounding calculations.
- Incorporate realistic brokerage fees, taxes, slippage and market impact.
- Use direct liquidity indicators where available.
- Report uncertainty and test robustness across additional periods,
  market regimes, stocks and probability thresholds.
- Document the exact evaluation sample used for every model and benchmark.

## 7. Conclusion

The results indicate measurable but uneven forecast improvements over a
zero-return baseline, with stronger reported direction metrics for the
market-level series than for the stock panel. Liquidity-proxy measures and
forecast gains vary across stocks. The strategy analysis shows that
transaction costs can materially reduce returns and make the reported mean
net return negative under the 50-basis-point scenario. These findings
support cautious interpretation and further validation rather than claims
of guaranteed predictive or trading performance.

### 2.2 Test 1 results

| Dataset | Horizon | Model | RMSE | OOS R-squared | Direction accuracy | AUC |
|---|---|---|---:|---:|---:|---:|
| Market | 1 day | RF | 1.5729 | 0.1009 | 59.67% | 0.6514 |
| Market | 1 day | XGB | 1.5588 | 0.1170 | 58.44% | 0.6269 |
| Market | 5 days | RF | 4.3204 | 0.0714 | 56.17% | 0.6318 |
| Market | 5 days | XGB | 4.2984 | 0.0808 | 57.02% | 0.6214 |
| Panel | 1 day | RF | 2.4885 | 0.0283 | 58.70% | 0.5985 |
| Panel | 1 day | XGB | 2.4695 | 0.0431 | 58.18% | 0.5951 |
| Panel | 5 days | RF | 6.0703 | 0.0331 | 57.76% | 0.5880 |
| Panel | 5 days | XGB | 6.0889 | 0.0271 | 57.97% | 0.5899 |

In Test 1, XGBoost achieves lower RMSE than Random Forest for the market
series at both horizons and for the one-day panel forecast. Random Forest
has lower five-day panel RMSE. Direction accuracy and AUC also vary by
model, reinforcing that model performance depends on the evaluation metric.

Positive out-of-sample R-squared values indicate improvement over the
corresponding benchmark used in the evaluation. These results should be














cat >> report/member5_analysis.md <<'EOF'

### 2.2 Test 1 results

| Dataset | Horizon | Model | RMSE | OOS R-squared | Direction accuracy | AUC |
|---|---|---|---:|---:|---:|---:|
| Market | 1 day | RF | 1.5729 | 0.1009 | 59.67% | 0.6514 |
| Market | 1 day | XGB | 1.5588 | 0.1170 | 58.44% | 0.6269 |
| Market | 5 days | RF | 4.3204 | 0.0714 | 56.17% | 0.6318 |
| Market | 5 days | XGB | 4.2984 | 0.0808 | 57.02% | 0.6214 |
| Panel | 1 day | RF | 2.4885 | 0.0283 | 58.70% | 0.5985 |
| Panel | 1 day | XGB | 2.4695 | 0.0431 | 58.18% | 0.5951 |
| Panel | 5 days | RF | 6.0703 | 0.0331 | 57.76% | 0.5880 |
| Panel | 5 days | XGB | 6.0889 | 0.0271 | 57.97% | 0.5899 |

In Test 1, XGBoost achieves lower RMSE than Random Forest for the market
series at both horizons and for the one-day panel forecast. Random Forest
has lower five-day panel RMSE. Direction accuracy and AUC also vary by
model, reinforcing that model performance depends on the evaluation metric.

Positive out-of-sample R-squared values indicate improvement over the
corresponding benchmark used in the evaluation. These results should be
interpreted within each dataset and horizon and should not be treated as
evidence of guaranteed future performance.

### 2.2 Test 1 results

| Dataset | Horizon | Model | RMSE | OOS R-squared | Direction accuracy | AUC |
|---|---|---|---:|---:|---:|---:|
| Market | 1 day | RF | 1.5729 | 0.1009 | 59.67% | 0.6514 |
| Market | 1 day | XGB | 1.5588 | 0.1170 | 58.44% | 0.6269 |
| Market | 5 days | RF | 4.3204 | 0.0714 | 56.17% | 0.6318 |
| Market | 5 days | XGB | 4.2984 | 0.0808 | 57.02% | 0.6214 |
| Panel | 1 day | RF | 2.4885 | 0.0283 | 58.70% | 0.5985 |
| Panel | 1 day | XGB | 2.4695 | 0.0431 | 58.18% | 0.5951 |
| Panel | 5 days | RF | 6.0703 | 0.0331 | 57.76% | 0.5880 |
| Panel | 5 days | XGB | 6.0889 | 0.0271 | 57.97% | 0.5899 |

In Test 1, XGBoost achieves lower RMSE than Random Forest for the market
series at both horizons and for the one-day panel forecast. Random Forest
has lower five-day panel RMSE. Direction accuracy and AUC also vary by
model, reinforcing that model performance depends on the evaluation metric.

Positive out-of-sample R-squared values indicate improvement over the
corresponding benchmark used in the evaluation. These results should be
interpreted within each dataset and horizon and should not be treated as
evidence of guaranteed future performance.
