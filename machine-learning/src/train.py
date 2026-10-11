
import os
import json
import random
from pathlib import Path
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import joblib
from sklearn.preprocessing import StandardScaler

from preprocess import (
    EXPENSE_CATEGORIES,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    preprocess_daily_expenses,
    build_sequences,
    get_file_hash,
)

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def compute_metrics(actual: np.ndarray, predicted: np.ndarray):
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {'mae': round(mae, 2), 'rmse': round(rmse, 2)}

def train_lstm_pipeline():
    set_seeds(42)
    root = Path(__file__).resolve().parents[2]
    raw_path = root / 'machine-learning' / 'data' / 'raw' / 'Personal_Finance_Dataset.csv'
    proc_dir = root / 'machine-learning' / 'data' / 'processed'
    artifacts_dir = root / 'machine-learning' / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    proc_csv = proc_dir / 'daily_expenses.csv'
    if not proc_csv.exists():
        print("Processed data not found. Running preprocessing...")
        df_daily = preprocess_daily_expenses(raw_path, proc_dir)
    else:
        df_daily = pd.read_csv(proc_csv, index_col='Date', parse_dates=True)

    window = 30
    horizon = 30

    print(f"Building sequence windows (lookback={window}d, horizon={horizon}d)...")
    X, Y, dates = build_sequences(df_daily, window=window, horizon=horizon)
    n_samples = len(X)
    print(f"Constructed {n_samples} sequence samples.")

    # Chronological Split: 70% Train, 15% Validation, 15% Test
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)
    n_test = n_samples - n_train - n_val

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]

    split_dates = {
        'train': {'start': dates[0], 'end': dates[n_train - 1], 'count': n_train},
        'val': {'start': dates[n_train], 'end': dates[n_train + n_val - 1], 'count': n_val},
        'test': {'start': dates[n_train + n_val], 'end': dates[-1], 'count': n_test},
    }
    print(f"Split breakdown: Train={n_train}, Val={n_val}, Test={n_test}")

    # Fit scalers STRICTLY on training split
    scaler_X = StandardScaler()
    scaler_X.fit(X_train.reshape(-1, X_train.shape[-1]))

    X_train_s = scaler_X.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_s = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_s = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    scaler_Y = StandardScaler()
    scaler_Y.fit(Y_train)

    Y_train_s = scaler_Y.transform(Y_train)
    Y_val_s = scaler_Y.transform(Y_val)
    Y_test_s = scaler_Y.transform(Y_test)

    # Save fitted scalers
    scaler_X_path = artifacts_dir / 'scaler_X.joblib'
    scaler_Y_path = artifacts_dir / 'scaler_Y.joblib'
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_Y, scaler_Y_path)
    print(f"Saved scalers to {artifacts_dir}")

    # Baseline Calculations on Test Set
    # Baseline 1: Naive Last-30d sum (what was spent in the lookback window is projected for forward window)
    # Feature 0 in X is the daily total, and features 1..8 are the categories
    baseline_naive_preds = np.zeros_like(Y_test)
    for i in range(len(X_test)):
        # sum of total (col 0) and categories (cols 1..8) over the 30 days
        baseline_naive_preds[i, 0] = np.sum(X_test[i, :, 0])
        for c in range(len(EXPENSE_CATEGORIES)):
            baseline_naive_preds[i, c + 1] = np.sum(X_test[i, :, c + 1])

    baseline_naive_total = compute_metrics(Y_test[:, 0], baseline_naive_preds[:, 0])
    
    # Baseline 2: Train Mean Baseline (predict mean of training set)
    train_mean = np.mean(Y_train, axis=0)
    baseline_mean_preds = np.tile(train_mean, (len(Y_test), 1))
    baseline_mean_total = compute_metrics(Y_test[:, 0], baseline_mean_preds[:, 0])

    print(f"Baseline Naive Last-30d Total: MAE={baseline_naive_total['mae']}, RMSE={baseline_naive_total['rmse']}")
    print(f"Baseline Train-Mean Total:     MAE={baseline_mean_total['mae']}, RMSE={baseline_mean_total['rmse']}")

    # Build genuine Sequential LSTM Model
    n_features = X_train.shape[-1]
    n_targets = Y_train.shape[-1]

    model = keras.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(n_targets)
    ], name="ExpenseCast_LSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )

    model_path = artifacts_dir / 'lstm_model.keras'
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=14,
        restore_best_weights=True,
        verbose=1
    )

    print("Training TensorFlow/Keras LSTM model...")
    history = model.fit(
        X_train_s, Y_train_s,
        validation_data=(X_val_s, Y_val_s),
        epochs=75,
        batch_size=32,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # Evaluate on held-out test set
    print("Evaluating model on held-out test set...")
    # Load best checkpoint to ensure strictly best weights are evaluated
    best_model = keras.models.load_model(str(model_path))
    preds_s = best_model.predict(X_test_s, verbose=0)
    preds = scaler_Y.inverse_transform(preds_s)
    # Clip negative predictions to 0 for expenses
    preds = np.maximum(0.0, preds)

    lstm_total_metrics = compute_metrics(Y_test[:, 0], preds[:, 0])
    print(f"LSTM Test Total: MAE={lstm_total_metrics['mae']}, RMSE={lstm_total_metrics['rmse']}")

    category_metrics = {}
    for idx, cat in enumerate(EXPENSE_CATEGORIES, start=1):
        c_m = compute_metrics(Y_test[:, idx], preds[:, idx])
        category_metrics[cat] = c_m
        print(f"  {cat:18s} MAE={c_m['mae']}, RMSE={c_m['rmse']}")

    metadata = {
        'model_name': 'ExpenseCast_LSTM',
        'model_type': 'TensorFlow/Keras Sequential LSTM',
        'framework_versions': {
            'tensorflow': tf.__version__,
            'keras': keras.__version__,
            'numpy': np.__version__,
            'pandas': pd.__version__,
        },
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'dataset': {
            'file': str(raw_path.name),
            'sha256': get_file_hash(raw_path),
            'date_range': {'min': str(df_daily.index.min()), 'max': str(df_daily.index.max())},
            'provenance': 'Personal_Finance_Dataset.csv (unverified public source)',
            'currency': 'Unverified generic monetary units',
        },
        'architecture': {
            'input_shape': [window, n_features],
            'output_shape': [n_targets],
            'layers': [
                {'type': 'Input', 'shape': [window, n_features]},
                {'type': 'LSTM', 'units': 64, 'return_sequences': True},
                {'type': 'Dropout', 'rate': 0.2},
                {'type': 'LSTM', 'units': 32},
                {'type': 'Dropout', 'rate': 0.2},
                {'type': 'Dense', 'units': 16, 'activation': 'relu'},
                {'type': 'Dense', 'units': n_targets, 'activation': 'linear'}
            ]
        },
        'features': FEATURE_COLUMNS,
        'targets': TARGET_COLUMNS,
        'expense_categories': EXPENSE_CATEGORIES,
        'window_size_days': window,
        'forecast_horizon_days': horizon,
        'splits': split_dates,
        'epochs_trained': len(history.history['loss']),
        'evaluation': {
            'lstm_test_total': lstm_total_metrics,
            'lstm_test_categories': category_metrics,
            'baselines': {
                'naive_last_30d_total': baseline_naive_total,
                'train_mean_total': baseline_mean_total,
            }
        }
    }

    metadata_path = artifacts_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Training completed successfully. Artifacts and metadata saved to {artifacts_dir}")
    return metadata

if __name__ == '__main__':
    train_lstm_pipeline()
