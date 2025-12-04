# [MLP Project 4] Hull Tactical – Market Prediction under Volatility Constraints

## 📌 1. Project Overview

| Detail                | Description                                                                                                                                                                                                 |
| :-------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Course**            | CS 53744 Machine Learning Project                                                                                                                                                                           |
| **Task**              | Time-series regression to predict **daily excess returns** of the S&P 500 and design a **volatility-constrained allocation strategy**.                                                                      |
| **Dataset**           | [Kaggle Competition – Hull Tactical: Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)                                                                                |
| **Goal**              | (1) Predict `market_forward_excess_returns`, (2) map predictions to **daily weights w ∈ [0, 2]**, (3) satisfy **σ_strategy ≤ 1.2 × σ_benchmark**, (4) maximize a **Modified Sharpe ratio**.                 |
| **Evaluation Metric** | Kaggle: Modified Sharpe ratio. Local: OOF RMSE & Correlation (for prediction quality) + **Sharpe, volatility ratio, cumulative return** (for strategy performance).                                         |
| **Final Model**       | **ElasticNet (PCA features) + LightGBM (raw FE)** blended (0.95 / 0.05) + **volatility-constrained allocation strategy**                                                                                    |
| **Baseline Models**   | Mean-prediction baseline, standalone ElasticNet, standalone LightGBM, standalone XGBoost (tested, excluded from final blend)                                                                                |
| **Key Insight**       | No model significantly beats the baseline in RMSE (consistent with EMH), but a **carefully regularized blend + mild leverage (k = 0.5)** achieves a **small Sharpe improvement under 120% volatility cap**. |

---

## 👥 2. Team Information

| Role   | Name | GitHub ID     |
| :----- | :--- | :------------ |
| Member | 박원규  | `@keiro23`    |
| Member | 이유정  | `@yousrchive` |
| Member | 정승환  | `@whan0767`   |

---

## 🏆 3. Final Performance Summary

The final pipeline consists of:

1. **Rich feature engineering (lags, rolling stats, regimes, macro shocks, interactions)**
2. **Time-series cross-validation (walk-forward) without leakage**
3. **Model comparison & blending (ElasticNet + LightGBM)**
4. **Strategy evaluation under a volatility constraint (≤ 120% of S&P 500)**

### 3.1 Prediction Performance (OOF, Time-Series CV)

Using 5-fold TimeSeriesSplit on the feature-engineered train set:

| Model              | RMSE (mean ± std) | Corr (mean ± std) | Comment                                     |
| :----------------- | :---------------- | :---------------- | :------------------------------------------ |
| **Baseline**       | ≈ 0.0108 ± 0.0027 | ≈ 0.00            | Train-mean prediction                       |
| ElasticNet         | ≈ 0.0111 ± 0.0028 | ≈ 0.03–0.04       | PCA(15) + ElasticNet                        |
| LightGBM           | ≈ 0.0122 ± 0.0025 | ≈ 0.02–0.03       | Raw FE, tree-based boosting                 |
| XGBoost            | ≈ 0.0124 ± 0.0025 | ≈ 0.03–0.04       | Slightly worse than ElasticNet / LightGBM   |
| **Blend (EN+LGB)** | ≈ 0.0115          | ≈ 0.035           | 0.95 ElasticNet + 0.05 LightGBM (RMSE-opt.) |

→ **Takeaway:** No single model clearly dominates the baseline; any predictability is extremely weak, consistent with EMH.

### 3.2 Strategy Performance (Vol-Constrained Allocation)

We convert blended predictions to daily weights:

* Standardize blended prediction: z_t
* Define weights: w_t = clip(1 + k·z_t, 0, 2)
* Search k ∈ [0, 50] with step 0.5 under constraint σ_strategy ≤ 1.2 × σ_benchmark

**Best k (under constraint)**: **k = 0.5**

| Metric                   | Benchmark (w = 1) | Blend Strategy (k = 0.5) |
| :----------------------- | :---------------- | :----------------------- |
| Mean daily excess return | ≈ 0.000265        | ≈ 0.000331               |
| Volatility ratio         | 1.0               | ≈ 1.20 (capped)          |
| Annualized Sharpe        | ≈ 0.378           | ≈ 0.393                  |
| Final cumulative return  | ≈ 0.400           | ≈ 0.491                  |

**Interpretation:**
The performance gap is **small**, but under a strict volatility cap it indicates that **weak yet non-zero structure** in the feature space can be translated into a slight Sharpe improvement, which is conceptually consistent with EMH’s “very limited predictability” view.

---

## ⚙️ 4. How to Reproduce Results

We separate the workflow into:

1. **Local / offline pipeline** (EDA, FE, TS-CV, backtesting, model export)
2. **Kaggle online inference** (evaluation API using `predict(test: pl.DataFrame)`)

### 4.1. Environment Setup & Dependencies

1. **Create & activate a virtual environment (local):**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\activate  # Windows
```

2. **Install required packages:**

```bash
pip install -r requirements.txt
```

Key libraries:

* `numpy`, `pandas`, `polars`
* `scikit-learn`
* `lightgbm`, `xgboost` (optional)
* `matplotlib`

---

### 4.2. Data Preparation

1. Download `train.csv` and `test.csv` from the **Hull Tactical** Kaggle competition.
2. Place them inside the `data/` directory at the project root:

```text
Project4/
└── data/
    ├── train.csv
    └── test.csv  # mock test for structure reference
```

---

### 4.3. Local Pipeline: Feature Engineering, TS-CV & Backtesting

The offline pipeline does three things:

1. **Feature engineering on train** (`generate_FE_interaction_regime`)
2. **Time-series CV + OOF predictions** (`ts_cv_oof_predictions`)
3. **Blend optimization + strategy backtest** (`search_best_k_for_blend`)

---

### 4.4. Kaggle Inference: Online predict() with Evaluation API

The final result earned from the local backtest is applied to the submission code in `src/elastic-lgmb_weight_scale.ipynb`.

Kaggle’s evaluation environment:

* No internet
* You receive **test batches** with:

  * Features M*, E*, I*, P*, V*, S*, MOM*, D*
  * `lagged_forward_returns`, `lagged_risk_free_rate`, `lagged_market_forward_excess_returns`
* You must implement:

```python
def predict(test: pl.DataFrame) -> float:
    ...
```

Core idea:

* Maintain a **buffer** of past rows in memory
* Use `lagged_market_forward_excess_returns` as the y_{t−1} equivalent
* Reproduce the **same FE logic as train**, but online & incremental
* Select `feature_list` columns in the correct order
* Apply models and blend predictions as trained locally.

You then typically plug this into the provided evaluation template (`default_inference_server`) and submit your solution. This process is often demonstrated within a Kaggle-specific notebook.

---

## 📁 5. Project Directory Structure

```text
Project4/
├── data/
│   ├── submission.csv
│   ├── test.csv
│   └── train.csv
│
├── notebook/
│   ├── 01_eda_baseline.ipynb
│   ├── 02_Feature_Engineering_PCA.ipynb
│   └── 03_Modeling_with_Backtest.ipynb
│
├── src/
│   ├── baseline_submission.ipynb
│   ├── elastic-lgmb_weight_scale.ipynb
│   └── submission.csv
│
├── requirements.txt
└── README.md
```

---

## 🧩 6. Notes & Alignment with Course Requirements

* **Baseline vs Improved Models**

  * Baseline: mean predictor
  * Improved: ElasticNet, LightGBM, blended model

* **Feature Engineering & Validation Strategy**

  * Rich FE on lagged targets and macro variables
  * TimeSeriesSplit walk-forward CV to avoid leakage

* **Local Sharpe-variant & Volatility Plots**

  * Backtesting code generates cumulative return and volatility ratio plots

* **Kaggle Leaderboard**

  * Final submission created via the Kaggle inference pipeline above
  * Screenshot and commentary included in the PDF report

* **EMH Discussion**

  * Report links small Sharpe improvement and weak predictability back to EMH (approx. weak-form consistency).