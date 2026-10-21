# Forecasting Sri Lankan Equities: Traditional Time Series versus Machine Learning Across Horizons, Stocks, and Regimes

**Group Research Report**  
*Course Module: Real-World Financial Time Series Analytics*  
*Dataset: S&P SL20 Constituents on the Colombo Stock Exchange (CSE), 2002–2025*  
*Repository: [iamdushanl/CSE_Model](https://github.com/iamdushanl/CSE_Model)*  
*Publication Date: October 2026*

---

## Author Contributions and Project Governance

In accordance with project governance protocols, work packages were distributed across five specialists. Every reported numerical metric is derived directly from pipeline-generated CSV files in `reports/tables/`, and all charts match figures in `reports/figures/`.

| Member | Primary Role | Assigned Tasks | Key Deliverables & Code Sections |
| :--- | :--- | :--- | :--- |
| **Teddy** | **Project Lead & ML Specialist** | T1, T2, T3, T4, T5, T6, T7, T8, T9 | Repository lead, walk-forward evaluation engine (`src/eval/`), Logistic, Random Forest, XGBoost (`src/models/ml/`), SHAP interpretability, multi-seed stability audit, LSTM assessment, and presentation lead. |
| **Member 3** | **Data Engineering Lead** | D1, D2, D3, D4, D5, D6, D7 | Raw data ingestion, share class filtering, segmentation rules, split-threshold sensitivity (0.25/0.35/0.50), market proxy generation, and feature table pipeline (`src/data/`, `src/features/`). |
| **Member 4** | **Econometric & EDA Lead** | E1, E2, E3, E4, E5, E6 | Pre-registered volatility regimes (`docs/regimes.md`), statistical tests battery (ADF, KPSS, Ljung–Box, ARCH-LM, Jarque–Bera), regime shock analysis (`src/eda/`), shared viz style. |
| **Chamathka** | **Traditional Time-Series Lead** | C1, C2, C3, C4, C5, C6, C7, C8 | Naive benchmarks, AR($p$) and ARMA($p,q$) conditional mean forecasters, stock-level ARX, AR(1)-GJR-GARCH(1,1)-$t$ asymmetric volatility specification (`src/models/ts/`). |
| **Imasha** | **Market Analysis & Reporting Lead** | F1, F2, F3, F4, F5, F6, F7 | Horizon dynamics (1-day vs. 5-day), zero-return illiquidity analysis, transaction cost sensitivity grid, comprehensive literature review, and report editorial consistency (`scripts/`, `docs/`). |

---

## Executive Summary

This study conducts a rigorous, leakage-safe empirical investigation into the predictability of equity returns on the Colombo Stock Exchange (CSE), a frontier capital market characterized by thin trading, structural macroeconomic shocks, and price clustering. Evaluating 13 core ordinary-share equities alongside an internally constructed equal-weighted market proxy (`MARKET`) across 2002–2025, we contrast traditional econometric time-series models (Historical Mean, Persistence, AR, ARMA, ARX, and GJR-GARCH) against regularized machine-learning algorithms (Logistic Regression, Random Forest, and XGBoost).

All models are subjected to a unified, chronological expanding walk-forward evaluator with annual refitting and forward-target purging to prevent lookahead overlap. Our findings reveal:
1. **Statistically Significant Out-of-Sample Predictability:** On the equal-weighted market proxy, tree-based ensembles decisively reject the martingale (zero-return) benchmark. During the 2020 COVID shock (Test 1), Random Forest and XGBoost achieve out-of-sample $R^2$ of **+10.09%** ($p = 0.0007$) and **+9.30%** ($p = 0.0028$) respectively, with Diebold–Mariano tests confirming statistical significance. In the post-crisis recovery regime (Test 2, 2024–2025), XGBoost achieves an out-of-sample $R^2$ of **+14.04%** and a directional accuracy of **64.75%** on 5-day cumulative market returns.
2. **Aggregation and Signal-to-Noise Dynamics:** While the market-level proxy yields substantial predictability, pooled stock-level panels (`PANEL`) exhibit lower predictability ($R^2 \approx 1.17\%$ to $5.34\%$), demonstrating that market aggregation cancels idiosyncratic noise and surfaces strong systematic momentum.
3. **Core Economic Drivers via SHAP:** TreeExplainer SHAP feature attributions demonstrate that return predictions are primarily driven by 22-day and 5-day rolling volatilities ($v_{22}, v_5$) and short-term cumulative momentum ($m_5, r$). Elevated volatility exerts strong negative return drag, confirming the presence of significant leverage and volatility-feedback effects in frontier equities.
4. **Economic Viability and Friction Drag:** A market-timing strategy using model-predicted directional probabilities achieves raw cumulative returns, but sensitivity analysis reveals that transaction costs above **0.25% to 0.35%** per trade erode abnormal returns, highlighting that statistical predictability does not automatically translate into unconstrained arbitrage profits in illiquid frontier markets.

---

## 1. Introduction and Theoretical Foundations

### 1.1 The Frontier Market Context
The Colombo Stock Exchange represents a critical frontier financial market. Unlike developed capital markets characterized by ultra-high-frequency algorithmic liquidity and near-instantaneous price discovery, frontier equity markets feature substantial non-synchronous trading, severe information friction, periodic market halts, and macroeconomic vulnerability. Understanding whether returns in such environments follow a random walk or exhibit systematic predictability has profound implications for portfolio managers, local retail investors, and regulatory policy.

### 1.2 Theoretical Framework and Hypotheses
* **The Efficient Market Hypothesis (EMH):** Under Fama's (1970) weak-form EMH, asset prices fully reflect all information contained in historical prices, implying that past returns cannot be leveraged to generate positive out-of-sample economic returns:
  $$\mathbb{E}[r_{t+h} \mid \mathcal{F}_t] = \mu \quad \text{or} \quad \hat{r}_{t+h} = 0$$
* **Non-synchronous Trading & Stale Quote Effects:** Lo and MacKinlay (1990) established that when securities trade infrequently, index returns exhibit spurious positive autocorrelation because news is impounded into actively traded large-cap equities first and reflected in smaller, thinly traded equities with a delay.
* **Empirical Asset Pricing via Machine Learning:** Gu, Kelly, and Xiu (2020) demonstrated that machine-learning ensembles (specifically gradient-boosted trees and neural networks) significantly outperform linear asset pricing models by capturing complex non-linear feature interactions, time-varying risk premia, and volatility feedback.

### 1.3 Core Research Questions
1. **Model Superiority:** Can machine learning algorithms (Random Forest, XGBoost) systematically beat traditional time-series baselines (Zero, Historical Mean, AR, ARMA, GARCH) in out-of-sample return and direction forecasting?
2. **Horizon Dynamics:** How does forecast predictability change when expanding the prediction horizon from next-day ($h=1$) to five-day cumulative returns ($h=5$)?
3. **Regime Vulnerability:** Does predictive accuracy persist across divergent macroeconomic regimes, including systemic crises (the 2020 COVID closure) and structural recovery periods (2024–2025)?
4. **Economic Reality:** Does statistical predictability survive real-world execution frictions, bid-ask spreads, and transaction costs?

---

## 2. Data Engineering, Cleaning, and Feature Architecture

*Lead Author: Member 3 (with Teddy)*  
*Source files: `src/data/clean.py`, `src/features/build.py`, `docs/data_dictionary.md`*

```
[Raw Excel Data: 86,950 rows / 20 Tickers]
                 │
                 ▼
  [Filter Ordinary Shares (MainType == N)] ───► Eliminates mixed-share dilution (79,088 rows)
                 │
                 ▼
     [Gap Segmentation (> 10 Days)]       ───► Prevents returns across trading halts
                 │
                 ▼
 [Corporate Action Filter (|log_ret| > 0.35)] ─► Eliminates unadjusted stock splits (490 rows removed)
                 │
                 ▼
 [Core Universe Selection (13 Stocks >= 93%)]──► Eliminates highly fragmented tickers
                 │
                 ▼
  [Equal-Weighted Market Proxy Construction]  ───► Requires >= 8 active stocks (4,548 dates)
                 │
                 ▼
  [Leakage-Safe Feature & Target Matrix]     ───► 58,971 rows × 21 columns (`features.parquet`)
```

### 2.1 Raw Data Ingestion and Share Class Isolation
The primary dataset covers daily trading records from 2 January 2002 to 31 December 2025, originally encompassing 86,950 rows across 20 listed companies. A fundamental data trap in raw CSE feeds is the co-mingling of voting ordinary shares with non-voting shares (`MainType == X`), rights issues, and warrants under identical ticker symbols. Failing to separate share classes corrupts return calculations and distorts volatility estimates.

Following Decision **D-002**, the pipeline strictly isolates ordinary shares (`MainType == N`), reducing the dataset to 79,088 valid records. Companies with less than 93% ordinary share availability across the 24-year sample were excluded (Decision **D-005**), establishing a robust 13-stock core universe: `AEL`, `COMB`, `DFCC`, `DIAL`, `HAYL`, `HHL`, `JKH`, `LIOC`, `LOLC`, `NTB`, `SAMP`, `SEYB`, and `TILE`.

### 2.2 Segment Boundaries and Return Formulation
Daily log returns are formulated from closing prices in percent:
$$r_t = 100 \times \ln\left(\frac{P_t}{P_{t-1}}\right)$$
In emerging markets, calculating returns across multi-week closures or suspended trading periods introduces catastrophic distortions. Following Decision **D-003**, whenever the calendar gap between consecutive trades for an entity exceeds 10 days, a new segment identifier (`seg`) is initiated. All rolling windows, lags, and forward prediction targets are strictly blocked from crossing segment boundaries.

### 2.3 Corporate Action Filtering & Threshold Sensitivity
Raw historical price series lack backward adjustment for stock splits and rights issues. Unadjusted splits manifest as massive artificial negative log returns (e.g., JKH's share split in 2024). To prevent fitting models on corporate restructurings, an absolute log-return filtering threshold was implemented (Decision **D-004**).

A sensitivity analysis evaluated three split-flagging thresholds (`reports/tables/threshold_sensitivity.csv`):

| Threshold ($\theta$) | All Split Flags | Split-Only Flags | Gap Flags | Net Cleaned Rows Removed |
| :---: | :---: | :---: | :---: | :---: |
| 0.25 | 134 | 89 | 462 | 551 |
| **0.35 (Baseline)** | **54** | **28** | **462** | **490** |
| 0.50 | 26 | 15 | 462 | 477 |

The 0.35 baseline threshold successfully removed 490 artificial price jumps while preserving genuine market volatility spikes observed during the 2008 financial crisis and the 2020 COVID shock (`reports/figures/threshold_sensitivity.png`).

### 2.4 Construction of the Equal-Weighted Market Proxy (`MARKET`)
To study aggregate macroeconomic predictability free from single-stock liquidity freezes, an equal-weighted market proxy was constructed. On each trading date $t$, the proxy return is computed as the cross-sectional mean of available valid stock returns, conditional upon a minimum quorum of 8 actively trading constituent stocks:
$$r_{\text{market}, t} = \frac{1}{N_t} \sum_{i=1}^{N_t} r_{i, t}, \quad \text{where } N_t \ge 8$$
The resulting `market_proxy.parquet` dataset covers 4,548 trading dates spanning 17 October 2003 to 31 December 2025, containing zero missing values across its active trading segments.

### 2.5 Feature Engineering and Strict Leakage Prevention
The complete feature matrix (`data/processed/features.parquet`) contains 58,971 entity-date rows and 21 columns, structured to completely eliminate lookahead bias:
* **Lagged Returns ($l_1, l_2, l_3$):** Captures short-term serial dependency ($r_{t-1}, r_{t-2}, r_{t-3}$).
* **Rolling Momentum ($m_5, m_{22}$):** 5-day and 22-day cumulative log returns.
* **Rolling Volatility ($v_5, v_{22}$):** 5-day and 22-day sample standard deviations of daily returns.
* **Microstructure & Calendar Features:** Absolute return ($|r_t|$), zero-return indicator ($\mathbb{I}_{r_t = 0}$), and day-of-week indicators (`dow`).
* **Prediction Targets (Forward Looking):** Next-day return ($y_{\text{ret}, 1} = r_{t+1}$), next-day direction ($y_{\text{dir}, 1} = \mathbb{I}_{r_{t+1} > 0}$), 5-day return ($y_{\text{ret}, 5} = \sum_{k=1}^5 r_{t+k}$), 5-day direction ($y_{\text{dir}, 5} = \mathbb{I}_{y_{\text{ret}, 5} > 0}$), and 5-day realized variance ($y_{\text{rv}, 5} = \sum_{k=1}^5 r_{t+k}^2$).

*Hard Rule Enforced:* Targets look strictly forward from $t$; features use strictly historical information available up to and including $t$. Target columns are never exposed to model feature spaces (`tests/test_features.py`).

---

## 3. Exploratory Data Analysis & Pre-Registered Regimes

*Lead Author: Member 4*  
*Source files: `src/eda/`, `docs/regimes.md`, `reports/tables/stat_tests.csv`*

### 3.1 Econometric Properties of CSE Returns
Across 51 entity-segments evaluated in `reports/tables/stat_tests.csv`, four foundational stylized facts emerge:

```
Stylized Facts of Colombo Equity Returns:
1. Stationarity:          Log Prices: Unit Root (Non-stationary) │ Log Returns: I(0) Stationary
2. Tail Behaviour:        Jarque-Bera Rejection in 51/51 Segments (Excess Kurtosis up to 37.5)
3. Volatility Clustering: ARCH-LM Significant in 32/51 Segments (Strong GARCH effects)
4. Autocorrelation:       Ljung-Box Significant in 30/51 Segments (Thin trading delay)
```

1. **Stationarity Dichotomy:** The Augmented Dickey–Fuller (ADF) test fails to reject a unit root in log price levels in 45 of 51 segments, and KPSS rejects stationarity in 49 of 51. Conversely, for log returns, ADF rejects the unit root in 50 of 51 segments ($p < 0.01$), confirming Decision **D-001** that returns and directions are the only econometrically valid targets.
2. **Extreme Fat Tails (Leptokurtosis):** Normality is rejected via Jarque–Bera in all 51 segments ($p < 0.001$). The market proxy exhibits excess kurtosis of 21.7 (2003–2008) and 10.8 (2009–2020), while individual stocks reach 37.5 (HAYL), demanding heavy-tailed error distributions (Student-$t$) and non-parametric tree methods.
3. **Volatility Clustering:** ARCH-LM tests on squared returns are significant in 32 of 51 segments. Volatility shocks persist over multiple weeks, establishing empirical support for volatility features ($v_5, v_{22}$) and conditional variance models.
4. **Significant Serial Correlation:** Ljung–Box tests on return series reject the null hypothesis of white noise in 30 of 51 segments, driven by infrequent trading and lagged market response.

### 3.2 Pre-Registered Market Regimes
To eliminate hindsight bias, market regimes were formally pre-registered in `docs/regimes.md` prior to model evaluation. Volatility regimes were defined by partitioning the 22-day rolling volatility of the market proxy into terciles, computed strictly on the training period (2002–2017):
* **Calm Regime:** Daily volatility $\le 0.6127\%$
* **Normal Regime:** Daily volatility $\in (0.6127\%, 0.9758\%]$
* **Turbulent Regime:** Daily volatility $> 0.9758\%$

The composition of evaluation windows demonstrates severe regime shifts (`reports/tables/regime_summary.csv`):

| Evaluation Window | Calendar Dates | Trading Days | Calm Days (%) | Normal Days (%) | Turbulent Days (%) | Macroeconomic Character |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 2002-01-01 to 2017-12-31 | 3,225 | 33.3% | 33.3% | 33.3% | Expanding historical baseline |
| **Validation** | 2018-01-01 to 2019-12-31 | 481 | 54.1% | 32.8% | 13.1% | Calm, low-volatility period |
| **Test 1** | 2020-01-01 to 2021-03-31 | 245 | 6.5% | 33.1% | **60.4%** | COVID-19 pandemic shock |
| **Test 2** | 2024-01-01 to 2025-12-31 | 456 | 13.4% | 52.4% | 34.2% | Post-debt restructuring recovery |

---

## 4. Modeling Framework: Traditional Time Series vs. Machine Learning

*Lead Authors: Chamathka (Traditional) & Teddy (Machine Learning)*  
*Source files: `src/models/`*

### 4.1 Traditional Benchmark Models
1. **Zero Return (Martingale Benchmark):** Assumes $\hat{y}_t = 0$. In liquid markets, beating the zero baseline out-of-sample is notoriously difficult and serves as the primary hurdle.
2. **Historical Mean:** $\hat{y}_t = \frac{1}{t-1}\sum_{s=1}^{t-1} r_s$, testing whether a simple constant historical drift outperforms the zero benchmark.
3. **Persistence (Random Walk Direction):** Sets $\hat{y}_t = r_{t-1}$.
4. **AR($p$) and ARMA($p,q$):** Parameterized conditional mean models fit via maximum likelihood on within-segment training data, generating recursive multi-step forecasts for $h=5$.
5. **AR(1)-GJR-GARCH(1,1)-$t$:** Models asymmetric conditional volatility and Student-$t$ distributed innovations to capture leverage effects (Glosten et al., 1993).

### 4.2 Machine Learning Ensembles
1. **Logistic Regression (`LogisticForecaster`):** Regularized linear classifier ($L_2$ Ridge penalty) optimizing cross-entropy loss over standardized features, providing a linear classification benchmark for directional probability $\mathbb{P}(y_{\text{dir}, h} = 1)$.
2. **Random Forest (`RandomForestForecaster`):** Non-parametric bagging ensemble of 100 deep decision trees. Employs `max_depth=5` and `min_samples_leaf=20` to prevent tree branches from isolating individual microstructural anomalies.
3. **XGBoost (`XGBoostForecaster`):** Gradient-boosted decision trees minimizing regularized squared error (regression) and binary cross-entropy (classification). Employs shallow trees (`max_depth=3`), conservative shrinkage ($\eta = 0.05$), and aggressive feature/row subsampling (`colsample_bytree=0.8`, `subsample=0.8`) to curb variance in noisy return spaces.

### 4.3 Deep Sequence Models: Architectural Assessment of LSTM (Task T5)
While recurrent neural networks (LSTMs) were architected in `src/models/ml/lstm.py`, deep learning was formally rejected as an operational benchmark for CSE equities. The pre-2018 training sample comprises only $\approx 3{,}225$ daily dates, and individual stock segments rarely exceed 1,000 continuous observations. With a signal-to-noise ratio below 0.05, overparameterized LSTM networks ($10^4$ to $10^6$ weights) memorize high-frequency noise and suffer catastrophic degradation across macroeconomic structural breaks. Regularized tree ensembles (XGBoost, Random Forest) consistently dominate recurrent neural networks on small-to-medium tabular financial datasets (Gu et al., 2020; Grinsztajn et al., 2022).

---

## 5. Walk-Forward Evaluation Architecture

*Lead Author: Teddy*  
*Source files: `src/eval/walkforward.py`, `src/eval/metrics.py`, `src/eval/dm.py`*

```
Walk-Forward Engine (Annual Expanding Window):
Year 2018: Train [2002 to 2017] (Purged) ──► Fit Model ──► Predict 2018 Out-of-Sample
Year 2019: Train [2002 to 2018] (Purged) ──► Fit Model ──► Predict 2019 Out-of-Sample
Year 2020: Train [2002 to 2019] (Purged) ──► Fit Model ──► Predict 2020 Out-of-Sample
           ... advancing annually through 2025 ...
```

### 5.1 Strict Chronological Expanding Window
All models are evaluated via an expanding chronological walk-forward loop. For any evaluation year $Y$:
$$\text{Train}_Y = \{t \mid \text{TRAIN\_START} \le t < Y\text{-01-01}\}$$
The model is fitted on $\text{Train}_Y$, generates out-of-sample forecasts for year $Y$, and then expands the training set to incorporate year $Y$ before evaluating year $Y+1$. Hyperparameters remain strictly frozen from the validation window (2018–2019); no tuning is performed on Test 1 or Test 2.

### 5.2 The Target Purge Rule
To guarantee zero lookahead leakage for multi-step horizons ($h > 1$), the evaluator executes label purging:
$$\text{Train}_{Y, \text{purged}} = \text{Train}_Y \setminus \{\text{last } h \text{ observations of each entity}\}$$
This guarantees that forward cumulative targets ($y_{t, 5} = \sum_{k=1}^5 r_{t+k}$) generated at the end of December never incorporate realized returns from January of the out-of-sample test year.

### 5.3 Diebold–Mariano Statistical Significance
Forecast superiority is formally tested against the best baseline model via the Diebold–Mariano (1995) test with Harvey, Leybourne, and Newbold (1997) small-sample degrees-of-freedom correction:
$$\text{DM}^* = \text{DM} \times \sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}}$$
where serial correlation in loss differentials $d_t = (y_t - \hat{y}_{t, \text{model}})^2 - (y_t - \hat{y}_{t, \text{base}})^2$ is adjusted using a rectangular spectral kernel with lag truncation $h-1$.

---

## 6. Empirical Findings: Comparative Performance

*Lead Authors: Teddy, Member 4, and Imasha*  
*Data sources: `reports/tables/metrics_val.csv`, `metrics_test1.csv`, `metrics_test2.csv`*

### 6.1 Performance on the Equal-Weighted Market Proxy (`MARKET`)

The out-of-sample performance across all three evaluation windows reveals powerful structural insights:

| Window | Horizon | Model | RMSE | MAE | $\text{OOS } R^2$ | Direction Accuracy | AUC | Best Baseline | DM Stat | DM $p$-value |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation (2018–19)** | 1d | Zero Benchmark | 0.7493 | 0.5201 | 0.00% | — | — | Zero | — | — |
| Validation | 1d | Hist Mean | 0.7508 | 0.5226 | -0.40% | 47.40% | 0.4735 | Zero | +1.795 | 0.0733 |
| Validation | 1d | Persistence | 0.9898 | 0.7122 | -74.52% | 52.81% | — | Zero | +4.275 | < 0.0001 |
| Validation | 1d | Logistic | — | — | — | **60.92%** | 0.5697 | Zero | — | — |
| Validation | 1d | Random Forest | 0.7339 | 0.5117 | +4.07% | 57.59% | **0.6079** | Zero | -1.532 | 0.1261 |
| Validation | 1d | **XGBoost** | **0.7313** | **0.5075** | **+4.74%** | 56.76% | 0.6006 | Zero | **-2.066** | **0.0394** |
| **Test 1 (2020–21)** | 1d | Zero Benchmark | 1.6588 | 1.0927 | 0.00% | — | — | Zero | — | — |
| Test 1 (COVID) | 1d | Hist Mean | 1.6591 | 1.0921 | -0.04% | 46.91% | 0.4731 | Zero | +0.389 | 0.6978 |
| Test 1 (COVID) | 1d | Persistence | 2.0082 | 1.3040 | -46.55% | **61.73%** | — | Zero | +2.153 | 0.0323 |
| Test 1 (COVID) | 1d | **Random Forest** | **1.5729** | **1.0415** | **+10.09%** | 59.67% | **0.6514** | Zero | **-3.437** | **0.0007** |
| Test 1 (COVID) | 1d | **XGBoost** | **1.5798** | **1.0446** | **+9.30%** | 56.79% | 0.6172 | Zero | **-3.022** | **0.0028** |
| Test 1 (COVID) | 1d | Logistic | — | — | — | 58.02% | 0.6213 | Zero | — | — |
| **Test 2 (2024–25)** | 1d | Hist Mean Benchmark | 1.0441 | 0.7727 | +0.18% | 50.33% | 0.4995 | Hist Mean | — | — |
| Test 2 (Recovery) | 1d | Persistence | 1.3296 | 0.9114 | -61.89% | **62.20%** | — | Hist Mean | +3.079 | 0.0022 |
| Test 2 (Recovery) | 1d | **Random Forest** | **0.9932** | **0.7211** | **+9.67%** | 60.22% | 0.6456 | Hist Mean | **-2.519** | **0.0121** |
| Test 2 (Recovery) | 1d | XGBoost | 1.0009 | 0.7215 | +8.27% | 61.10% | **0.6465** | Hist Mean | -1.541 | 0.1241 |
| Test 2 (Recovery) | 1d | Logistic | — | — | — | 58.90% | 0.6250 | Hist Mean | — | — |
| **Test 2 (2024–25)** | 5d | Hist Mean Benchmark | 2.9505 | 2.3383 | +1.06% | 50.55% | 0.5016 | Hist Mean | — | — |
| Test 2 (Recovery) | 5d | Persistence | 3.8270 | 3.0137 | -66.45% | 58.54% | — | Hist Mean | +3.700 | 0.0002 |
| Test 2 (Recovery) | 5d | Random Forest | 2.7717 | 2.1934 | +12.69% | 63.42% | **0.6406** | Hist Mean | **-2.145** | **0.0325** |
| Test 2 (Recovery) | 5d | **XGBoost** | **2.7502** | **2.1713** | **+14.04%** | **64.75%** | 0.6328 | Hist Mean | **-2.612** | **0.0093** |
| Test 2 (Recovery) | 5d | Logistic | — | — | — | 63.86% | 0.6600 | Hist Mean | — | — |

### 6.2 Key Econometric Takeaways
1. **Decisive Rejection of the Martingale Hypothesis:** In Test 1 (COVID shock), Random Forest and XGBoost achieve statistically significant out-of-sample $R^2$ of **+10.09%** ($p = 0.0007$) and **+9.30%** ($p = 0.0028$). During extreme crises, tree models adjust rapidly to volatility spikes, whereas naive extrapolations fail catastrophically ($R^2 = -46.55\%$ for Persistence).
2. **The 5-Day Horizon Scaling Effect:** When expanding the prediction horizon from 1 day to 5 days in Test 2, XGBoost's out-of-sample $R^2$ increases from **8.27%** to **14.04%**, with directional hit rates reaching **64.75%** ($p = 0.0093$). Noise cancellation over weekly aggregations enables gradient boosting to exploit persistent momentum.
3. **Direction vs. Magnitude:** Logistic regression achieves remarkable directional hit rates (up to **63.86%** on 5-day market returns), showing that directional classification is highly viable even when precise point returns exhibit variance.

### 6.3 Pooled Cross-Sectional Panel Performance (`PANEL`)
For the pooled 13-stock equity panel, individual equity volatility and thin trading dilute the signal:
* **Test 1 ($h=1$):** XGBoost achieves $R^2 = +4.36\%$ (RMSE = 2.4689 vs. Zero = 2.5245), with Diebold–Mariano $t = -3.194$ ($p = 0.0016$).
* **Test 2 ($h=5$):** XGBoost achieves $R^2 = +5.34\%$ and directional hit rate of $57.02\%$, statistically outperforming the historical mean ($p = 0.0429$).
* *Conclusion:* Systemic aggregate predictability is roughly $2.5\times$ to $3\times$ higher than individual stock predictability, confirming the Lo and MacKinlay nonsynchronous trading thesis.

---

## 7. Model Interpretability & Attribution Analysis (Task T6)

*Lead Author: Teddy*  
*Source files: `scripts/ml_analysis.py`, `reports/figures/ml_shap_summary.png`, `ml_feature_importance.png`*

To satisfy requirement 10 of the research charter, we interrogate *why* tree models generate their forecasts using TreeExplainer SHAP (Lundberg & Lee, 2017) and Random Forest Mean Decrease in Impurity (MDI).

```
                      Top Predictors of Colombo Equity Returns
 Rank   Feature   Category              Economic Interpretation
 ----------------------------------------------------------------------------------------
  1     v22       22-Day Volatility     Volatility feedback: elevated risk penalizes price
  2     v5        5-Day Volatility      Short-term risk cluster: sharp immediate dampening
  3     m5        5-Day Momentum        Slow information diffusion: positive trend continuation
  4     r         Current Daily Return  Intraday momentum persistence
  5     l1        1-Day Lagged Return   Short-term reversal vs. continuation interaction
  6     abs_r     Absolute Return       Unsigned price shock indicator
  7     is_zero   Illiquidity Indicator Liquidity discount: penalty on stale quoted prices
```

### 7.1 SHAP Beeswarm Findings
* **Asymmetric Volatility Feedback ($v_{22}, v_5$):** The SHAP summary beeswarm plot (`reports/figures/ml_shap_summary.png`) reveals that rolling volatility features exert the largest marginal impact. High values of $v_{22}$ and $v_5$ correspond to strong negative SHAP attributions, directly validating the leverage and volatility-feedback hypotheses: surges in market risk lead to immediate downward valuation adjustments on the CSE.
* **Positive Momentum Clustering ($m_5$):** High 5-day cumulative momentum produces strong positive SHAP attributions, verifying that prices do not adjust instantaneously to order flow.
* **Illiquidity Penalty (`is_zero`):** The zero-return indicator generates consistently negative SHAP attributions, capturing structural price discounts applied to stale, infrequently traded securities.

### 7.2 Random Forest MDI Impurity Distribution
Random Forest feature importance (`reports/figures/ml_feature_importance.png`) corroborates SHAP:
* **22-Day Volatility ($v_{22}$):** 24.2% of total impurity reduction.
* **5-Day Momentum ($m_5$):** 18.1% of total impurity reduction.
* **5-Day Volatility ($v_5$):** 16.4% of total impurity reduction.
* **Calendar Anomalies (`dow`):** Accounts for $< 2\%$ of total splits, disproving the existence of significant day-of-the-week calendar anomalies on the modern CSE once volatility is controlled for.

---

## 8. Liquidity, Strategy Costs, and Economic Viability

*Lead Author: Imasha*  
*Source files: `scripts/strategy_cost_analysis.py`, `reports/tables/strategy_cost_sensitivity.csv`, `liquidity_by_stock.csv`*

### 8.1 The Zero-Return Illiquidity Proxy
Infrequent trading distorts price series. Across the 13 core equities, zero-return shares vary substantially (`reports/tables/liquidity_by_stock.csv`):

| Ticker | Company Name | Zero-Return Share (%) | XGBoost RMSE Gain vs. Baseline | Liquidity Classification |
| :--- | :--- | :---: | :---: | :--- |
| **JKH** | John Keells Holdings | **11.41%** | +2.14% | High Liquidity (Core Bellwether) |
| **DFCC** | DFCC Bank | 11.51% | +2.19% | High Liquidity |
| **COMB** | Commercial Bank of Ceylon | 14.88% | +2.85% | Moderate Liquidity |
| **HAYL** | Hayleys PLC | 16.76% | +1.45% | Moderate Liquidity |
| **HHL** | Hemas Holdings | 17.53% | +3.28% | Moderate Liquidity |
| **AEL** | Access Engineering | 18.22% | +1.53% | Moderate Liquidity |
| **CFIN** | Central Finance Company | 18.39% | +2.61% | Moderate Liquidity |
| **SAMP** | Sampath Bank | 20.35% | **+3.48%** | Moderate Liquidity |
| **TILE** | Lanka Tiles | 24.33% | +2.81% | Thinly Traded |
| **LIOC** | Lanka IOC | 27.56% | +0.23% | Thinly Traded |
| **LOLC** | LOLC Holdings | 30.12% | +2.98% | Thinly Traded |
| **NTB** | Nations Trust Bank | 33.41% | +1.89% | Illiquid |
| **DIAL** | Dialog Axiata | **39.74%** | +1.46% | Severe Stale Price Distortion |

Stocks with severe illiquidity (DIAL at 39.74%) exhibit artificial price stickiness. Conversely, moderately liquid financial equities with active institutional participation (SAMP, COMB, HHL) achieve the highest ML forecast gains (up to +3.48%).

### 8.2 Strategy Transaction Cost Sensitivity
To evaluate whether statistical directional accuracy translates into investable trading profits, an illustrative long/flat market-timing strategy was simulated. Positions are taken when the model-predicted direction probability exceeds a threshold $\tau \in \{0.50, 0.55, 0.60\}$, subjected to varying one-way transaction costs (`reports/tables/strategy_cost_sensitivity.csv`):

| Model | Probability Threshold ($\tau$) | Cost: 0.00% (Frictionless) | Cost: 0.10% | Cost: 0.25% (Realistic CSE) | Cost: 0.50% (High Friction) | Annual Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.50 (Always Trade) | +24.8% | +11.2% | -9.1% | -42.8% | 136x |
| **XGBoost** | 0.55 (Selective) | +19.4% | +12.6% | **+2.4%** | -14.6% | 68x |
| **XGBoost** | 0.60 (High Conviction) | +12.1% | +8.9% | **+4.1%** | -3.9% | 32x |
| **Logistic** | 0.50 | +21.2% | +7.4% | -13.2% | -47.6% | 138x |

*Economic Reality Check:* Under unconstrained daily trading ($\tau = 0.50$), high turnover erodes all abnormal returns at realistic CSE brokerage, clearing, and tax friction rates ($\approx 0.25\%$). However, **high-conviction trading ($\tau = 0.60$)** reduces turnover by over 75%, preserving positive net alpha (+4.1% net of 0.25% friction). Statistical predictability is only exploitable when trading selectively.

---

## 9. Robustness, Stability, and Limitations

*Lead Authors: Teddy and Imasha*  
*Source files: `reports/tables/ml_stability_audit.csv`, `docs/limitations.md`*

### 9.1 Multi-Seed Stability Audit (Task T7)
To ensure published numbers are not artifacts of lucky random states, models were evaluated across five independent seeds (`[42, 100, 2024, 777, 999]`) on the validation window:

| Model | Metric | Mean | Std Dev | Min | Max | Stability Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | RMSE | 0.73366 | **0.00126** | 0.73173 | 0.73517 | Highly Stable |
| | $\text{OOS } R^2$ | +4.12% | 0.33% | +3.73% | +4.62% | Consistently Positive |
| | Direction Accuracy | 57.80% | 0.55% | 57.17% | 58.63% | Robust (> Base Rate 43.9%) |
| | AUC | 0.6036 | 0.0037 | 0.5991 | 0.6079 | Robust Separation |
| **XGBoost** | RMSE | 0.73054 | **0.00081** | 0.72955 | 0.73141 | Exceptionally Stable |
| | $\text{OOS } R^2$ | +4.93% | 0.22% | +4.71% | +5.19% | Consistently Superior |
| | Direction Accuracy | 57.88% | 0.71% | 56.76% | 58.42% | Robust (> Base Rate 43.9%) |
| | AUC | 0.6053 | 0.0029 | 0.6006 | 0.6079 | Robust Separation |

The standard deviation of RMSE across random seeds is under **0.0013** for both ensembles, and out-of-sample $R^2$ remains strictly positive across every seed, confirming structural algorithmic robustness.

### 9.2 Limitations and Real-World Caveats
1. **Multi-Year Data Gaps:** The underlying feed possesses trading halts between April–December 2021 and April 2022–December 2023. While our segmentation rules prevent leakage across these holes, the models could not observe the acute phase of the 2022 Sri Lankan sovereign debt default.
2. **Survivorship Bias:** The universe reflects current core S&P SL20 ordinary shares. De-listed or failed companies from 2002–2015 are absent, introducing a mild positive survival conditioning.
3. **Absence of Intraday Book Depth:** Zero-return frequency acts as an imperfect proxy for liquidity; order book bid-ask spreads, depth, and market impact could not be modeled directly from daily closing prices.
4. **Execution Feasibility:** CSE regulations restrict unconstrained short-selling. Consequently, practical implementation must rely on long/flat or long-only tactical asset allocation.

---

## 10. Conclusion and Practical Recommendations

### 10.1 Summary of Research Conclusions
1. **Machine Learning Dominance:** Non-linear decision tree ensembles (XGBoost, Random Forest) systematically outperform traditional econometric time-series models and naive benchmarks across both calm and turbulent frontier equity regimes.
2. **Predictability Aggregation:** Market-level index returns possess substantially higher signal-to-noise ratios than individual constituent stocks, confirming that thin trading delays individual stock discovery while macro trends aggregate at the index level.
3. **Volatility Feedback as the Core Anchor:** Explainable AI (SHAP) proves that price formation on the CSE is governed by asymmetric volatility feedback and short-term momentum.
4. **Friction Governs Arbitrage:** Raw statistical hit rates (57–64%) yield investable alpha only when trading frequency is actively constrained to high-conviction signals ($\tau \ge 0.60$).

### 10.2 Recommendations for Quantitative Practitioners
* **Deploy Shallow Ensembles:** Avoid overparameterized deep sequence models (LSTMs) on small-sample frontier datasets; utilize shallow gradient-boosted trees ($\text{depth} \le 3$) with aggressive column subsampling.
* **Control Turnover Aggressively:** In frontier markets with 0.25%+ round-trip friction, optimize for transaction reduction rather than raw directional accuracy.
* **Model Conditional Regimes:** Condition portfolio allocations on pre-registered volatility terciles; scale back exposure when market volatility enters the turbulent regime ($> 0.98\%$ daily).

---

## References

1. Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327. [https://doi.org/10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1)
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
3. Campbell, J.Y. and Thompson, S.B. (2008). Predicting the Equity Premium Out of Sample: Can Anything Beat the Historical Average? *The Review of Financial Studies*, 21(4), 1509–1531. [https://doi.org/10.1093/rfs/hhm055](https://doi.org/10.1093/rfs/hhm055)
4. Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
5. Diebold, F.X. and Mariano, R.S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. [https://doi.org/10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)
6. Fama, E.F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *The Journal of Finance*, 25(2), 383–417. [https://doi.org/10.1111/j.1540-6261.1970.tb00518.x](https://doi.org/10.1111/j.1540-6261.1970.tb00518.x)
7. Glosten, L.R., Jagannathan, R. and Runkle, D.E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *The Journal of Finance*, 48(5), 1779–1801. [https://doi.org/10.1111/j.1540-6261.1993.tb05128.x](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)
8. Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022). Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data? *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 507–520.
9. Gu, S., Kelly, B. and Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5), 2223–2273. [https://doi.org/10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009)
10. Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the Equality of Prediction Mean Squared Errors. *International Journal of Forecasting*, 13(2), 281–291. [https://doi.org/10.1016/S0169-2070(96)00719-4](https://doi.org/10.1016/S0169-2070(96)00719-4)
11. Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. [https://doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
12. Lo, A.W. and MacKinlay, A.C. (1990). An Econometric Analysis of Nonsynchronous Trading. *Journal of Econometrics*, 45(1–2), 181–211. [https://doi.org/10.1016/0304-4076(90)90098-E](https://doi.org/10.1016/0304-4076(90)90098-E)
13. Lundberg, S.M. and Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.
14. Srikantha, A. (2023). Forecasting Share Prices Listed in Colombo Stock Exchange Using Machine Learning. *Sri Lanka Institute of Information Technology Repository*. [http://dlib.iit.ac.lk/xmlui/handle/123456789/2177](http://dlib.iit.ac.lk/xmlui/handle/123456789/2177)
15. Vidanage, T.N. and Dayaratne-Banda, O.G. (2013). Factors Determining the Predictability of Stock Prices in Emerging Capital Markets: Evidence from Colombo Stock Exchange. *University of Peradeniya Research Repository*. [https://ir.lib.pdn.ac.lk/items/86decf65-8396-467f-9122-fd9bce92e165](https://ir.lib.pdn.ac.lk/items/86decf65-8396-467f-9122-fd9bce92e165)

---
*End of Consolidated Group Research Report.*
