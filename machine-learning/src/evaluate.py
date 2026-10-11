"""
ExpenseCast - Model Evaluation Script
Evaluates the trained TensorFlow/Keras LSTM model on the held-out test dataset
and compares with naive and mean baselines.
"""

import os
import json
from pathlib import Path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib

from preprocess import (
    EXPENSE_CATEGORIES,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_sequences
)

def evaluate_model():
    root = Path(__file__).resolve().parents[2]
    proc_csv = root / 'machine-learning' / 'data' / 'processed' / 'daily_expenses.csv'
    artifacts_dir = root / 'machine-learning' / 'artifacts'
    model_path = artifacts_dir / 'lstm_model.keras'
    scaler_X_path = artifacts_dir / 'scaler_X.joblib'
    scaler_Y_path = artifacts_dir / 'scaler_Y.joblib'
    metadata_path = artifacts_dir / 'metadata.json'

    if not model_path.exists() or not scaler_X_path.exists() or not scaler_Y_path.exists():
        raise FileNotFoundError("Model or scaler artifacts missing. Please run train.py first.")

    df_daily = pd.read_csv(proc_csv, index_col='Date', parse_dates=True)
    window = 30
    horizon = 30
    X, Y, dates = build_sequences(df_daily, window=window, horizon=horizon)

    n_samples = len(X)
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)
    n_test = n_samples - n_train - n_val

    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]
    test_dates = dates[n_train + n_val:]

    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)

    X_test_s = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    model = keras.models.load_model(str(model_path))
    preds_s = model.predict(X_test_s, verbose=0)
    preds = scaler_Y.inverse_transform(preds_s)
    preds = np.maximum(0.0, preds)

    # Metrics
    def m(a, p):
        return {
            'mae': round(float(np.mean(np.abs(a - p))), 2),
            'rmse': round(float(np.sqrt(np.mean((a - p) ** 2))), 2),
        }

    total_metrics = m(Y_test[:, 0], preds[:, 0])

    # Baseline: Naive last-30d
    naive_preds = np.zeros_like(Y_test)
    for i in range(len(X_test)):
        naive_preds[i, 0] = np.sum(X_test[i, :, 0])
        for c in range(len(EXPENSE_CATEGORIES)):
            naive_preds[i, c + 1] = np.sum(X_test[i, :, c + 1])
    naive_total = m(Y_test[:, 0], naive_preds[:, 0])

    print("=" * 60)
    print("EXPENSECAST MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Test samples: {n_test} (from {test_dates[0]} to {test_dates[-1]})")
    print(f"LSTM Model Total Expense:      MAE = {total_metrics['mae']}, RMSE = {total_metrics['rmse']}")
    print(f"Naive Last-30d Baseline Total: MAE = {naive_total['mae']}, RMSE = {naive_total['rmse']}")
    improvement = ((naive_total['mae'] - total_metrics['mae']) / naive_total['mae']) * 100
    print(f"LSTM MAE Improvement over Baseline: {improvement:.1f}%")
    print("-" * 60)
    print("Category Breakdown (MAE / RMSE):")
    for idx, cat in enumerate(EXPENSE_CATEGORIES, start=1):
        cat_m = m(Y_test[:, idx], preds[:, idx])
        naive_c = m(Y_test[:, idx], naive_preds[:, idx])
        print(f"  {cat:18s} LSTM: MAE={cat_m['mae']:8.2f}, RMSE={cat_m['rmse']:8.2f} | Naive: MAE={naive_c['mae']:8.2f}")
    print("=" * 60)

if __name__ == '__main__':
    evaluate_model()
