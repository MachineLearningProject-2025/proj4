# 🚀 CS53744 Team Project 4 – Hull Tactical Market Prediction

**Predicting S&P 500 Excess Returns & Designing a Volatility-Constrained Strategy**

This repository contains the full implementation, experiments, and report assets for **Team Project 4 (Hull Tactical Market Prediction)** of *CS 53744 Machine Learning*.
The goal of the project is to:

1. **Predict daily excess returns of the S&P 500 (market_forward_excess_returns)**
2. **Construct a daily allocation strategy w ∈ [0, 2]**
3. **Ensure portfolio volatility ≤ 120% of the benchmark**
4. **Maximize a Modified Sharpe Ratio**

This work integrates **time-series cross-validation, feature engineering, blending models, and backtesting**.
Kaggle competition link: [https://www.kaggle.com/competitions/hull-tactical-market-prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)

---

## 📌 Repository Structure

```
├── data/ 
│   ├── train.csv
│   ├── test.csv (mock)
│
├── notebooks/
│   ├── 01_EDA_and_StylizedFacts.ipynb
│   ├── 02_TS_CV_Modeling.ipynb
│   ├── 03_Blend_and_Strategy.ipynb
│   ├── 04_Kaggle_Inference_Demo.ipynb
│
├── models_fe_rich/
│   ├── global_scaler.pkl
│   ├── global_pca.pkl
│   ├── 
│
├── src/
│   ├── feature_engineering.py
│   ├── train_full_model.py
│   ├── backtest.py
│   ├── kaggle_predict.py
│
├── figures/
│   ├── eda_returns_distribution.png
│   ├── ts_cv_rmse.png
│   ├── cumulative_returns_comparison.png
│   ├── volatility_ratio_plot.png
│
├── report/
│   ├── Assignment4_TeamID_StudentID_Lastname_Firstname.pdf
│
├── requirements.txt
│
└── README.md
```

---

# 1. 🧠 Problem Overview

The Hull Tactical Market Prediction challenge requires predicting the next-day **excess return**:

[
y_t = \text{forward_returns}_t - \text{risk_free_rate}_t
]

and turning predictions into **allocation weights**:

[
w_t \in [0, 2], \qquad
\text{Volatility}(w_t y_t) \le 1.2 \times \text{Volatility}(y_t)
]

The evaluation metric is a **Modified Sharpe Ratio**, penalizing excessive volatility or poor performance.

We frame the task as a **no-leakage time-series ML problem**, then convert predictions into a constrained investment strategy.

---

# 2. 📊 Dataset Description & EMH Perspective

The dataset contains ~9,000 daily observations and 98 market-related features:

* **M***: market dynamics
* **E***: macroeconomic variables
* **V***: volatility features
* **I***: interest rate features
* **P***: valuation features
* **S***: sentiment
* **MOM***: momentum
* **D***: binary indicators
* **Forward returns / RF rate / excess returns** (train only)

Empirical findings (stylized facts):

* Daily excess returns ≈ **zero mean + high noise**
* **Fat-tailed**, **leptokurtic** distribution
* **Almost no autocorrelation** in returns
* Strong **latent factor structure** (multicollinearity)
  → Consistent with the **Efficient Market Hypothesis (EMH)**:
  short-term direction is extremely hard to predict, but *market states (volatility/regime/macro shock)* may show limited predictability.

---

# 3. ⚙️ Feature Engineering

To capture market structure (rather than direction), we designed a rich FE pipeline:

### ✔ Lag Features

* y_{t−1}, y_{t−2}, y_{t−5}, y_{t−10}, y_{t−21}, y_{t−63}

### ✔ Rolling Statistics (no leakage; always based on lagged values)

* rolling mean / std / min / max
* rolling z-scores
* vol21, vol63, high_vol flag, vol_slope
* crisis regime (rolling quantile)

### ✔ Macro Shock Indicators (E*)

* rolling z-scores
* shock flags (>|2σ|)
* macro_shock_sum
* macro_crisis = 1 if ≥3 macro shocks occur simultaneously

### ✔ Interaction Features

* momentum × volatility : M_i × V_j
* macro spreads (E2–E11, E7–E12, …)

### ✔ Return Shock Indicator

* |y_{t−1}| > 2 × rolling std

### ✔ Train vs. Predict Mode

* Train uses target.shift(1)
* Kaggle online inference uses lagged_market_forward_excess_returns (provided by test.csv)

This FE significantly increased the model’s ability to detect *regimes* rather than “directions”.

---

# 4. 🤖 Modeling & Time-Series Validation

We train the following models:

* **ElasticNet (with StandardScaler + PCA)**
* **LightGBM**
* **(XGBoost tested; excluded in final blend)**

### Time-Series Cross-Validation

We use **walk-forward TS-CV (5 splits)** to avoid leakage.

### Results Summary

| Model           | RMSE (OOF)     | Corr (OOF) |
| --------------- | -------------- | ---------- |
| Baseline (mean) | ~0.0108        | ~0.00      |
| ElasticNet      | ~0.0111        | ~0.03–0.04 |
| LightGBM        | ~0.0122        | ~0.01–0.02 |
| XGBoost         | Underperformed | —          |

➡ **Small but consistent predictive signal**, aligned with EMH expectations.

---

# 5. 🔗 Model Blending

Grid search over weights (0.05 step):

### **Best Blend (RMSE-optimal)**

* **ElasticNet: 0.95**
* **LightGBM: 0.05**
* XGBoost: 0.00

This blend provides the most stable OOF performance.

---

# 6. 📈 Strategy Construction

Weights generated as:

[
z_t = \frac{\hat{y}_t - \mu}{\sigma}, \quad
w_t = \text{clip}(1 + k z_t, 0, 2)
]

We search k ∈ [0, 50]:

### **Best k = 0.5

(under volatility ≤ 120%)**

### Final Strategy vs Benchmark

| Metric                  | Benchmark | Strategy  |
| ----------------------- | --------- | --------- |
| Annualized Return       | ~0.000265 | ~0.000331 |
| Annualized Vol          | same      | ×1.20     |
| Sharpe                  | **0.378** | **0.393** |
| Final Cumulative Return | **0.400** | **0.491** |

The improvement is small but statistically meaningful, given EMH constraints.

---

# 7. 🧪 Kaggle Submission Pipeline

We export:

* 

Kaggle `predict()` implements *online feature engineering* using the provided:

* `lagged_forward_returns`
* `lagged_risk_free_rate`
* `lagged_market_forward_excess_returns`

Buffering & rolling logic reconstructs train-time FE without leakage.

---

# 8. 📝 Final Deliverables

Included in this repo:

* **Prediction file (test.csv)**
* **Kaggle Notebook** (with modeling, FE, backtesting, plots)
* **4-page Report (PDF)**
* **GitHub repo with code, models, figures**
* **Leaderboard screenshot**

---

# 9. 🔍 Limitations & Future Work

* Deep sequence models (LSTM/Transformer) not included
* Early test timesteps have incomplete FE
* PCA may obscure macro interpretability
* Additional external datasets could help (news sentiment, VIX, macro releases)

---

# 10. 📚 How to Reproduce

```
git clone <this-repo>
cd project

pip install -r requirements.txt

python src/train_full_model.py     # Re-train FE + models
python src/backtest.py             # Evaluate strategy locally
```

To submit to Kaggle:

1. Upload `models_fe_rich/` as a Kaggle Dataset
2. Attach dataset to your Notebook
3. Run `kaggle_predict.py` with the evaluation API

---

# 11. 🙌 Team & Acknowledgements

This project was completed as part of **CS 53744 Machine Learning**
Instructor: **Prof. Jongmin Lee**
Team Members: *[Insert Names]*

We thank Kaggle and Hull Tactical for providing the research environment.