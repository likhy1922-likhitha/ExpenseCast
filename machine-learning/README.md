# ExpenseCast — Machine Learning Pipeline

## 1. Overview
The ExpenseCast machine learning pipeline provides personal expense forecasting using a genuine **TensorFlow/Keras Sequential LSTM** trained on chronological transaction data.

It replaces toy scratch algorithms with an offline-trained sequential neural network, serialized artifacts, fitted preprocessing scalers, and reproducible evaluation on held-out test data.

---

## 2. Dataset Information & Profiling
- **Location**: `machine-learning/data/raw/Personal_Finance_Dataset.csv`
- **Total Records**: 1,500 rows, 5 columns (`Date`, `Transaction Description`, `Category`, `Amount`, `Type`)
- **Date Span**: `2020-01-02` to `2024-12-29` (5 years, 1,824 calendar days)
- **Active Expense Days**: 885 calendar days (~939 zero-expense days)
- **Type Distribution**: 1,222 `Expense`, 278 `Income`
- **Category Counts**:
  - Rent: 165
  - Travel: 160
  - Utilities: 157
  - Health & Fitness: 152
  - Shopping: 150
  - Food & Drink: 149
  - Salary: 146
  - Entertainment: 143
  - Investment: 142
  - Other: 136

### Data Quality & Provenance Findings
1. **Salary Categorization Inconsistency**:
   - All 146 records under the `Salary` category are marked with `Type = "Expense"`.
   - In accordance with non-negotiable project rules, the raw CSV remains unaltered. For expense forecasting, filtering is strictly based on `Type == "Expense"`, and this inconsistency is formally recorded in `machine-learning/data/processed/data_quality_report.json`.
2. **Income Categories**:
   - Only `Investment` (142 rows) and `Other` (136 rows) are marked `Type = "Income"`.
3. **Provenance & Currency**:
   - Unverified. The raw CSV contains no currency symbol or Kaggle license file. Amounts are treated as standard monetary units.

---

## 3. Data Preprocessing & Target Formulation
- **Script**: `machine-learning/src/preprocess.py`
- **Time Granularity**: Daily calendar series (`2020-01-02` to `2024-12-29`), zero-filling calendar days without transactions.
- **Lookback Window**: 30 days (`window_size_days = 30`)
- **Forecast Horizon**: 30 days (`forecast_horizon_days = 30`)
- **Features (12 total)**:
  - Daily Total Expense
  - 8 Daily Category Expenses (`Entertainment`, `Food & Drink`, `Health & Fitness`, `Rent`, `Salary`, `Shopping`, `Travel`, `Utilities`)
  - Normalized calendar signals: Day of Week (`dow`), Day of Month (`dom`), Month of Year (`month`)
- **Targets (9 total)**:
  - 30-day forward Total Expense
  - 30-day forward Expenses for each of the 8 categories

---

## 4. Model Architecture & Training
- **Script**: `machine-learning/src/train.py`
- **Architecture**:
  ```text
  Input(shape=(30, 12))
  └── LSTM(64, return_sequences=True)
  └── Dropout(0.2)
  └── LSTM(32, return_sequences=False)
  └── Dropout(0.2)
  └── Dense(16, activation='relu')
  └── Dense(9, activation='linear')
  ```
- **Splits**: Chronological, strictly preventing lookahead leakage:
  - Train: 1,235 samples (70%) — `2020-01-31` to `2023-06-18`
  - Validation: 264 samples (15%) — `2023-06-19` to `2024-03-08`
  - Test: 266 samples (15%) — `2024-03-09` to `2024-11-29`
- **Scaling**: `StandardScaler` fitted strictly on training data only.
- **Callbacks**: Early stopping on validation loss + ModelCheckpoint saving best weights.

---

## 5. Evaluation & Baseline Comparison
- **Evaluation Script**: `python machine-learning/src/evaluate.py`
- **Held-out Test Set Results**:
  - **LSTM Model Total Expense**: `MAE = 4180.08`, `RMSE = 4968.16`
  - **Naive Last-30d Baseline Total**: `MAE = 4974.67`, `RMSE = 6328.85`
  - **Historical Train-Mean Baseline Total**: `MAE = 3626.74`, `RMSE = 4545.10`
  - **LSTM Improvement over Naive Baseline**: **16.0% lower MAE**, **21.5% lower RMSE**.

---

## 6. Artifacts
The training pipeline saves the following files in `machine-learning/artifacts/`:
1. `lstm_model.keras` — Trained Keras model with best weights.
2. `scaler_X.joblib` — Fitted feature standardizer.
3. `scaler_Y.joblib` — Fitted target standardizer.
4. `metadata.json` — Comprehensive model specification, training timestamp, framework versions, and metrics.

---

## 7. Commands
```bash
# Run data preprocessing and profiling:
python machine-learning/src/preprocess.py

# Train the LSTM model and generate artifacts:
python machine-learning/src/train.py

# Evaluate the saved model against baselines:
python machine-learning/src/evaluate.py
```
