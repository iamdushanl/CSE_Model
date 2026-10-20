# Data Preparation, Cleaning, and Feature Engineering

## 1. Data Source and Description

The study uses the CSE dataset covering the period from 2 January 2002 to 31 December 2025. The raw dataset contains 86,950 rows representing 20 companies. After selecting ordinary shares (`MainType = N`), the dataset contains 79,088 rows across 13 companies.

## 2. Data Cleaning and Return Calculation

Daily log returns were calculated from closing prices and expressed as percentages:

\[
r_t = 100 \times \ln\left(\frac{P_t}{P_{t-1}}\right)
\]

where \(P_t\) is the closing price on day \(t\).

To avoid calculating returns across long gaps in trading records, observations separated by more than 10 days were assigned to different segments. Potential extreme returns were flagged using a split threshold of 0.35. Under this threshold, 54 split flags were identified, including 28 split-only flags that were not also marked as gap flags. A total of 490 returns were removed during the cleaning process.

Flagged returns require manual verification against reliable evidence before they can be conclusively classified. The manual review checklist has been prepared, but verification remains pending.

## 3. Data Audit and Threshold Sensitivity

The sensitivity analysis examined thresholds of 0.25, 0.35, and 0.50.

| Threshold | All split flags | Split-only flags | Gap flags |
|---:|---:|---:|---:|
| 0.25 | 134 | 89 | 462 |
| 0.35 | 54 | 28 | 462 |
| 0.50 | 26 | 15 | 462 |

The results show that the number of potential split-related flags decreases as the threshold increases, while the number of gap flags remains unchanged. This comparison helps assess how the cleaning rule affects the data.

## 4. Market Proxy Construction

An equal-weighted market proxy was constructed from valid stock returns. A minimum of eight valid stock returns was required for a market-proxy value on a given date.

The resulting `market_proxy.parquet` dataset contains 4,548 dates, from 17 October 2003 to 31 December 2025. The proxy contains no missing values on its available dates. Some feature-table dates do not have a proxy value because the minimum-stock requirement was not met.

## 5. Feature Engineering and Leakage Prevention

The feature table contains 58,971 rows and 21 columns. It includes current and lagged returns, rolling means and volatility measures, absolute returns, zero-return indicators, day-of-week information, and the market proxy where available.

The table includes five prediction targets: `y_ret_1`, `y_dir_1`, `y_ret_5`, `y_dir_5`, and `y_rv_5`. These represent future return, direction, and volatility outcomes over the specified horizons.

Feature calculations are grouped by company and segment so that lagged values and rolling windows do not cross segment breaks. Features use information available up to and including day \(t\), while targets look forward from day \(t\). Target columns must not be used as model input features.

## 6. Processed Datasets

The data pipeline produces three principal processed datasets:

- `returns_clean.parquet`: cleaned daily returns and associated flags.
- `market_proxy.parquet`: the equal-weighted market proxy.
- `features.parquet`: the feature table and prediction targets.

The feature table was checked for duplicate company-date keys; no duplicates were found.

## 7. Reproducibility and Limitations

The project tests passed in the development environment, and a fresh-clone test reproduced the principal data-processing counts. The data pipeline can be regenerated using the project instructions in the README.

The main limitations at this stage are the pending manual verification of flagged returns, missing market-proxy values on dates that do not meet the minimum-stock requirement, and the sensitivity of split flags to the selected threshold. These limitations should be considered when interpreting model results.