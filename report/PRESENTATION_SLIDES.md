# Forecasting Sri Lankan Equities: Traditional Time Series vs. Machine Learning
## Final Presentation Slide Deck (12 Slides)

**Module:** *Real-World Financial Time Series Analytics*  
**Team Members:** Teddy (Lead & ML), Member 3 (Data), Member 4 (EDA & Regimes), Chamathka (Traditional TS), Imasha (Analysis & Reporting)  
**Dataset:** Colombo Stock Exchange (CSE) S&P SL20 Constituents (2002–2025)

---

### Slide 1: Title & Executive Introduction
* **Title:** Forecasting Sri Lankan Equities: Traditional Time Series versus Machine Learning Across Horizons, Stocks, and Regimes
* **Research Core:** Does historical return and volatility data contain predictive signal on the Colombo Stock Exchange (CSE), or do prices follow a random walk?
* **Comparative Framework:** Naive Benchmarks (Zero, Mean, Persistence) & Traditional TS (AR, ARMA, GJR-GARCH) **VS.** Machine Learning (Logistic, Random Forest, XGBoost).
* **Key Finding in 10 Seconds:** Tree ensembles significantly beat the martingale benchmark on market-level returns ($p < 0.01$), scaling up to **$R^2 = +14.04\%$** on 5-day forecasts, primarily driven by volatility-feedback and short-term momentum.

---

### Slide 2: The Frontier Market Challenge (CSE Context)
* **What Makes the CSE Different?**
  * Severe non-synchronous trading and stale pricing (zero-return days up to 39.7%).
  * Exogenous structural shocks (2008 GFC, 2020 COVID shutdown, 2022 macroeconomic debt crisis).
  * High transaction friction (brokerage, clearing, taxes $\approx 0.25\%+$).
* **The Literature Anchor:**
  * *Fama (1970):* Testing Weak-Form Market Efficiency.
  * *Lo & MacKinlay (1990):* Spurious index autocorrelation driven by infrequent trading.
  * *Gu, Kelly & Xiu (2020):* Tree-based machine learning dominance in empirical asset pricing.

---

### Slide 3: Data Engineering & Leakage Avoidance
* **Raw Ingestion:** 86,950 rows across 20 tickers.
* **Cleaning Pipeline (Member 3):**
  1. *Share Class Isolation:* Isolated ordinary voting shares (`MainType == N`), discarding mixed share dilution (79,088 rows remaining).
  2. *Gap Segmentation:* Calendar gaps $> 10$ days start a new segment (`seg`). No rolling window or forward target crosses a segment boundary.
  3. *Unadjusted Split Removal:* Absolute log return filter at 0.35 removed 490 artificial corporate action jumps.
  4. *Core Universe:* 13 liquid ordinary shares with $\ge 93\%$ coverage.
* **Equal-Weighted Market Proxy (`MARKET`):** Formed from $\ge 8$ concurrent valid stock returns (4,548 active trading days).

---

### Slide 4: Exploratory Data Analysis & Pre-Registered Regimes
* **Stylized Statistical Facts (Member 4):**
  * *Non-stationary Prices, Stationary Returns:* ADF rejects unit root in returns across 50/51 segments ($p < 0.01$).
  * *Fat Tails:* Jarque-Bera rejects normality in 100% of segments (kurtosis up to 37.5).
  * *Volatility Clustering:* ARCH-LM significant in 32/51 segments.
* **Pre-Registered Volatility Regimes (`docs/regimes.md`):**
  * Terciles fixed strictly on 2002–2017 training window: Calm ($\le 0.61\%$), Normal ($0.61\% - 0.98\%$), Turbulent ($> 0.98\%$).
  * *Regime Stress:* Test 1 (2020 COVID) was **60.4% Turbulent**, while Validation (2018–19) was **54.1% Calm**.

---

### Slide 5: Evaluation Design: Strict Walk-Forward Protocol
* **The Engine (Teddy):**
  * Chronological expanding window starting 2002-01-01.
  * Refit once per calendar year, predict out-of-sample, advance.
* **Tuning Freeze:**
  * Hyperparameters tuned **only** on Validation (2018–2019). Zero tuning on Test 1 or Test 2.
* **Target Purging Rule:**
  * Purges the final $h$ observations per entity before fitting to eliminate target overlap across calendar years.
* **Statistical Testing:**
  * Harvey–Leybourne–Newbold corrected Diebold–Mariano test against the lowest-RMSE baseline.

---

### Slide 6: Model Architecture: Traditional vs. ML
* **Traditional Benchmarks (Chamathka):**
  * *Zero / Martingale:* $\hat{y}_t = 0$ (the efficient market hurdle).
  * *Historical Mean & Persistence:* Constant drift and lagged return.
  * *AR($p$) & ARMA($p,q$):* Recursive conditional mean.
  * *GJR-GARCH(1,1)-$t$:* Asymmetric conditional volatility with heavy-tailed Student-$t$ errors.
* **Machine Learning Suite (Teddy):**
  * *Logistic Regression:* Regularized linear probability classifier ($L_2$ Ridge).
  * *Random Forest:* Bagged trees (`max_depth=5`, `min_samples_leaf=20`).
  * *XGBoost:* Gradient boosting (`depth=3`, $\eta = 0.05$, `subsample=0.8`, `colsample=0.8`).
* **LSTM Feasibility (T5):** Omitted due to small-sample underdetermination ($N \approx 3{,}225$), low SNR, and non-stationarity.

---

### Slide 7: Empirical Results: 1-Day Market Returns
* **Validation Window (2018–2019 Calm):**
  * XGBoost achieves $\text{OOS } R^2 = \mathbf{+4.74\%}$ (DM $t = -2.066, p = 0.0394^*$).
  * Logistic Directional Accuracy = **60.92%** (Base rate = 43.87%).
* **Test 1 Window (2020 COVID Crisis):**
  * Random Forest achieves $\text{OOS } R^2 = \mathbf{+10.09\%}$ (DM $t = -3.437, p = 0.0007^{***}$).
  * XGBoost achieves $\text{OOS } R^2 = \mathbf{+9.30\%}$ (DM $t = -3.022, p = 0.0028^{***}$).
  * Persistence fails catastrophically ($\text{OOS } R^2 = -46.55\%$).
* **Test 2 Window (2024–2025 Recovery):**
  * Random Forest achieves $\text{OOS } R^2 = \mathbf{+9.67\%}$ ($p = 0.0121^*$).
  * XGBoost achieves $\text{OOS } R^2 = \mathbf{+8.27\%}$, Direction Accuracy = **61.10%** (AUC = 0.6465).

---

### Slide 8: The Horizon Scaling Effect: 1-Day vs. 5-Day
* **Lead Author:** Imasha
* **Market 1-Day vs. 5-Day Predictability (Test 2):**
  * 1-Day XGBoost: $R^2 = +8.27\%$, Accuracy = 61.10%
  * **5-Day XGBoost:** $R^2 = \mathbf{+14.04\%}$, Accuracy = **64.75%** (DM $t = -2.612, p = 0.0093^{**}$)
* **Why Does 5-Day Predictability Increase?**
  * Day-to-day microstructure noise and bid-ask bounce wash out over 5-day cumulative windows.
  * Aggregation highlights persistent medium-term trends and macroeconomic momentum in frontier markets.

---

### Slide 9: Model Interpretability: What Drives Forecasts?
* **SHAP Beeswarm Analysis (TreeExplainer on XGBoost):**
  1. **Rolling Volatilities ($v_{22}, v_5$):** Top importance ranking. High volatility drives large **negative** predicted returns. Direct empirical confirmation of the *volatility-feedback effect*.
  2. **Short-Term Momentum ($m_5, r$):** Strong positive correlation with future returns. Confirms slow information diffusion.
  3. **Illiquidity Penalty (`is_zero`):** Inactive trading days generate persistent negative drag.
* **Random Forest MDI:**
  * $v_{22}$ (24.2%), $m_5$ (18.1%), $v_5$ (16.4%). Calendar features (`dow`) contribute $< 2\%$.

---

### Slide 10: Market vs. Individual Stocks & Liquidity Drag
* **The Aggregation Advantage:**
  * Market Proxy $R^2$ reaches **+14.04%**, while pooled stock panel (`PANEL`) $R^2$ peaks at **+5.34%**.
  * Demonstrates that idiosyncratic noise cancels out, isolating systematic frontier risk premia.
* **The Zero-Return Drag:**
  * Liquid bellwethers (JKH at 11.4% zero-return share, COMB at 14.9%) reflect news efficiently.
  * Illiquid stocks (DIAL at 39.7%, NTB at 33.4%) suffer from stale price distortion, damping predictive gains.

---

### Slide 11: Economic Viability & Strategy Costs
* **Can You Trade This Profitably?**
  * Evaluated long/flat timing strategy across transaction cost tiers ($0.00\%$ to $0.50\%$).
* **The Turnover Trap:**
  * Daily trading ($\tau = 0.50$) generates 136x annual turnover; friction $> 0.25\%$ turns profits negative (-9.1%).
* **The Selective Solution:**
  * High-conviction threshold ($\tau = 0.60$) cuts turnover to 32x, preserving **+4.1% net annual alpha** after realistic CSE friction (0.25%).
* **Takeaway:** Statistical predictability only generates economic value when trade turnover is actively controlled.

---

### Slide 12: Summary, Conclusions & Recommendations
* **Core Takeaways:**
  1. *EMH Rejected on Index Level:* Sri Lankan equities exhibit statistically significant predictability via tree ensembles ($R^2$ up to 14%).
  2. *Tree Ensembles Beat Linear/Deep Models:* XGBoost and RF dominate both classical linear models and deep neural networks on small, noisy tabular series.
  3. *Volatility & Momentum are Fundamental:* Predictability is rooted in risk-premium feedback and delayed information diffusion.
* **Practical Quantitative Rules for Frontier Markets:**
  * Use shallow trees (`depth=3`) with column subsampling.
  * Trade only high-conviction signals to survive local brokerage and stamp duty frictions.
  * Scale exposures down when entering turbulent pre-registered volatility regimes.

---
*End of Presentation Slide Deck.*
