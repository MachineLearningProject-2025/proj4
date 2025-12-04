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

Typical usage:

```bash
cd src

# 1) Run full training pipeline: FE + TS-CV + blending
python train_full_model.py

# 2) Run backtest & Sharpe evaluation using OOF predictions
python backtest.py
```

`train_full_model.py` (core steps):

* Load `train.csv`
* Drop leakage columns: `forward_returns`, `risk_free_rate` (keep only target `market_forward_excess_returns`)
* Apply **rich FE** (lags, rolling stats, volatility regimes, macro shocks, interactions, return shocks), always based on **past** information to avoid leakage
* Perform TimeSeriesSplit CV, training:

  * **ElasticNet(PCA)**: `StandardScaler` → `PCA(15)` → `ElasticNet`
  * **LightGBM**: raw FE
  * (Optionally) XGBoost
* Save:

  * `global_scaler.pkl`
  * `global_pca.pkl`
  * `elasticnet_model.pkl`
  * `lightgbm_model.txt`
  * `feature_list.json` (the final feature columns used for training)

`backtest.py`:

* Load OOF predictions and ground-truth target
* Construct blend: **0.95 ElasticNet + 0.05 LightGBM**
* Compute benchmark and strategy returns under k-grid search
* Enforce volatility ≤ 120% of benchmark
* Report:

  * mean returns, volatilities, Sharpe
  * cumulative return curves (Figure: `cumulative_returns_comparison.png`)

Outputs are saved under `models_fe_rich/` and `figures/`.

---

### 4.4. Kaggle Inference: Online predict() with Evaluation API

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
* Apply:

  * `global_scaler` → `global_pca` → `elasticnet_model`
  * `lightgbm_model` on **raw FE**
  * Blend: `0.95 * pred_enet + 0.05 * pred_lgb`

Example (simplified) Kaggle-side script:

```python
import numpy as np
import polars as pl
import pickle
import json
import lightgbm as lgb

MODEL_PATH = "/kaggle/input/hull-tactical-dataset"

with open(f"{MODEL_PATH}/global_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(f"{MODEL_PATH}/global_pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open(f"{MODEL_PATH}/elasticnet_model.pkl", "rb") as f:
    enet = pickle.load(f)

lgb_model = lgb.Booster(model_file=f"{MODEL_PATH}/lightgbm_model.txt")

with open(f"{MODEL_PATH}/feature_list.json", "r") as f:
    feature_list = json.load(f)

BUFFER = { "rows": [] }

def make_test_FE(row: pl.DataFrame):

    global BUFFER
    BUFFER["rows"].append(row.to_dicts()[0])
    df = pl.from_dicts(BUFFER["rows"])

    # use lagged_market_forward_excess_returns as y_{t-1}
    df = df.with_columns([
        pl.col("lagged_market_forward_excess_returns").alias("y_lag1")
    ])

    # (1) rolling stats on y_lag1
    for w in [5,10,21,63]:
        df = df.with_columns([
            pl.col("y_lag1").rolling_mean(w).alias(f"roll_mean_{w}"),
            pl.col("y_lag1").rolling_std(w).alias(f"roll_std_{w}"),
        ])

    # (2) volatility regime
    df = df.with_columns([
        pl.col("y_lag1").rolling_std(21).alias("vol21"),
        pl.col("y_lag1").rolling_std(63).alias("vol63")
    ])
    df = df.with_columns([
        (pl.col("vol21") > pl.col("vol63")).cast(pl.Int8).alias("high_vol"),
        (pl.col("vol21") / (pl.col("vol63") + 1e-9)).alias("vol_slope"),
    ])

    # (3) macro shock on E*
    macro_cols = [c for c in df.columns if c.startswith("E")]
    for col in macro_cols:
        df = df.with_columns([
            ((pl.col(col) - pl.col(col).rolling_mean(63)) /
             (pl.col(col).rolling_std(63) + 1e-9)).alias(f"{col}_z")
        ])
        df = df.with_columns([
            (pl.col(f"{col}_z").abs() > 2).cast(pl.Int8).alias(f"{col}_shock")
        ])

    shock_cols = [c for c in df.columns if c.endswith("_shock")]
    if shock_cols:
        df = df.with_columns([
            sum([pl.col(c) for c in shock_cols]).alias("macro_shock_sum"),
            (pl.col("macro_shock_sum") >= 3).cast(pl.Int8).alias("macro_crisis"),
        ])
    else:
        df = df.with_columns([
            pl.lit(0).alias("macro_shock_sum"),
            pl.lit(0).cast(pl.Int8).alias("macro_crisis"),
        ])

    # (4) interaction: first few M and V
    m_cols = [c for c in df.columns if c.startswith("M")][:5]
    v_cols = [c for c in df.columns if c.startswith("V")][:5]
    for m in m_cols:
        for v in v_cols:
            df = df.with_columns((pl.col(m) * pl.col(v)).alias(f"{m}_x_{v}"))

    last = df.tail(1)

    # align with train-time features
    last = last.select([c for c in feature_list if c in last.columns])

    return last

def predict(test: pl.DataFrame) -> float:

    fe = make_test_FE(test)
    X = fe.to_numpy()

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    pred_en = enet.predict(X_pca)

    pred_lgb = lgb_model.predict(X)

    pred = 0.95 * pred_en + 0.05 * pred_lgb

    return float(pred[0])
```

You then plug this into the provided evaluation template (`default_inference_server`) and submit the notebook.

---

## 📁 5. Project Directory Structure

```text
Project4/
├── data/
│   ├── train.csv
│   └── test.csv                  # mock structure (not used for scoring)
│
├── src/
│   ├── feature_engineering.py    # generate_FE_interaction_regime, shared logic
│   ├── train_full_model.py       # FE + TS-CV + model training + export
│   ├── backtest.py               # blend & strategy evaluation, plots
│   ├── kaggle_predict.py         # predict() demo for Kaggle evaluation API
│   └── utils.py                  # helper functions (metrics, plotting, etc.)
│
├── models_fe_rich/
│   ├── global_scaler.pkl
│   ├── global_pca.pkl
│   ├── elasticnet_model.pkl
│   ├── lightgbm_model.txt
│   ├── feature_list.json
│
├── notebooks/
│   ├── 01_EDA_and_StylizedFacts.ipynb
│   ├── 02_TS_CV_Modeling.ipynb
│   ├── 03_Blend_and_Strategy.ipynb
│   └── 04_Kaggle_Inference_Demo.ipynb
│
├── figures/
│   ├── eda_returns_distribution.png
│   ├── ts_cv_rmse.png
│   ├── cumulative_returns_comparison.png
│   ├── volatility_ratio_plot.png
│
├── report/
│   └── Assignment4_TeamID_StudentID_Lastname_Firstname.pdf
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