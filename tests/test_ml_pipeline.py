"""
Tests for Machine Learning Pipeline, Artifact Loading, Smoke Training, and Inference
"""

import sys
import os
import json
from pathlib import Path
import pytest
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'machine-learning' / 'src'))
sys.path.insert(0, str(ROOT))

from backend.ml_service import ml_service, EXPENSE_CATEGORIES

ARTIFACTS_DIR = ROOT / 'machine-learning' / 'artifacts'

def test_artifacts_exist():
    assert (ARTIFACTS_DIR / 'lstm_model.keras').exists(), "Saved Keras model file must exist"
    assert (ARTIFACTS_DIR / 'scaler_X.joblib').exists(), "Input scaler artifact must exist"
    assert (ARTIFACTS_DIR / 'scaler_Y.joblib').exists(), "Target scaler artifact must exist"
    assert (ARTIFACTS_DIR / 'metadata.json').exists(), "Metadata artifact must exist"

def test_metadata_contents():
    metadata_path = ARTIFACTS_DIR / 'metadata.json'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    assert meta['model_name'] == 'ExpenseCast_LSTM'
    assert meta['model_type'] == 'TensorFlow/Keras Sequential LSTM'
    assert meta['window_size_days'] == 30
    assert meta['forecast_horizon_days'] == 30
    assert 'lstm_test_total' in meta['evaluation']
    assert meta['evaluation']['lstm_test_total']['mae'] > 0
    assert meta['evaluation']['lstm_test_total']['rmse'] > 0
    assert 'baselines' in meta['evaluation']

def test_ml_service_is_available_and_status():
    assert ml_service.is_available() is True, "MLService should successfully load artifacts"
    status = ml_service.get_status()
    assert status['available'] is True
    assert 'TensorFlow/Keras' in status['engine']
    assert status['window_size_days'] == 30
    assert status['forecast_horizon_days'] == 30

def test_ml_training_smoke_test():
    """Smoke test: Confirms a real TensorFlow/Keras LSTM can build and fit without errors."""
    window = 10
    n_features = 5
    n_targets = 3
    
    model = keras.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(16),
        layers.Dense(n_targets)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Synthetic mini-batch
    X_dummy = np.random.randn(8, window, n_features).astype(np.float32)
    Y_dummy = np.random.randn(8, n_targets).astype(np.float32)
    
    history = model.fit(X_dummy, Y_dummy, epochs=2, batch_size=4, verbose=0)
    assert len(history.history['loss']) == 2
    preds = model.predict(X_dummy, verbose=0)
    assert preds.shape == (8, n_targets)

def test_ml_service_inference_with_mock_history():
    """Tests ML inference on transactions spanning at least 30 days."""
    dates = pd_dates = [f"2026-08-{d:02d}" for d in range(1, 32)] + [f"2026-09-{d:02d}" for d in range(1, 31)]
    transactions = []
    for idx, d in enumerate(dates):
        transactions.append({
            'date': d,
            'type': 'Expense',
            'amount': 250.0 + (idx % 5) * 50,
            'merchant': 'Grocery Store',
            'category': EXPENSE_CATEGORIES[idx % len(EXPENSE_CATEGORIES)],
            'recurring': 0
        })

    res = ml_service.predict_for_user(transactions, budget=15000.0)
    assert res['status'] == 'success'
    assert res['predicted_total'] > 0
    assert len(res['category_predictions']) == len(EXPENSE_CATEGORIES)
    assert 'highest_increase_category' in res
    assert res['overrun'] >= 0.0

def test_ml_service_insufficient_history():
    """Tests honest response when transactions span fewer than 30 calendar days."""
    transactions = [
        {'date': '2026-09-01', 'type': 'Expense', 'amount': 100.0, 'category': 'Food & Drink'},
        {'date': '2026-09-05', 'type': 'Expense', 'amount': 200.0, 'category': 'Shopping'},
    ]
    res = ml_service.predict_for_user(transactions, budget=10000.0)
    assert res['status'] == 'insufficient_history'
    assert 'warning' in res
    assert res['predicted_total'] == 0.0
