"""
ExpenseCast - ML Service
Manages loading of trained TensorFlow/Keras LSTM model artifacts
and performs real inference on user transaction history.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import joblib

EXPENSE_CATEGORIES = [
    'Entertainment',
    'Food & Drink',
    'Health & Fitness',
    'Rent',
    'Salary',
    'Shopping',
    'Travel',
    'Utilities',
]

FEATURE_COLUMNS = ['total'] + EXPENSE_CATEGORIES + ['dow', 'dom', 'month']
TARGET_COLUMNS = ['total'] + EXPENSE_CATEGORIES

class MLService:
    def __init__(self, artifacts_dir: Path | None = None):
        if artifacts_dir is None:
            root = Path(__file__).resolve().parents[1]
            artifacts_dir = root / 'machine-learning' / 'artifacts'
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.scaler_X = None
        self.scaler_Y = None
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = self.artifacts_dir / 'lstm_model.keras'
        scaler_X_path = self.artifacts_dir / 'scaler_X.joblib'
        scaler_Y_path = self.artifacts_dir / 'scaler_Y.joblib'
        metadata_path = self.artifacts_dir / 'metadata.json'

        if not (model_path.exists() and scaler_X_path.exists() and scaler_Y_path.exists()):
            self.model = None
            return

        try:
            self.model = keras.models.load_model(str(model_path))
            self.scaler_X = joblib.load(scaler_X_path)
            self.scaler_Y = joblib.load(scaler_Y_path)
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
        except Exception as e:
            print(f"Error loading ML artifacts: {e}")
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None and self.scaler_X is not None and self.scaler_Y is not None

    def get_status(self) -> dict:
        if not self.is_available():
            return {
                'available': False,
                'status': 'Model artifacts not found or failed to load',
                'engine': 'TensorFlow/Keras LSTM',
                'artifacts_path': str(self.artifacts_dir),
            }

        return {
            'available': True,
            'status': 'Model loaded and ready for inference',
            'engine': 'TensorFlow/Keras Sequential LSTM',
            'model_name': self.metadata.get('model_name') if self.metadata else 'ExpenseCast_LSTM',
            'trained_at': self.metadata.get('trained_at') if self.metadata else None,
            'dataset': self.metadata.get('dataset') if self.metadata else {},
            'window_size_days': self.metadata.get('window_size_days', 30) if self.metadata else 30,
            'forecast_horizon_days': self.metadata.get('forecast_horizon_days', 30) if self.metadata else 30,
            'metrics': self.metadata.get('evaluation', {}).get('lstm_test_total', {}) if self.metadata else {},
            'features': FEATURE_COLUMNS,
            'categories': EXPENSE_CATEGORIES,
            'framework_versions': {
                'tensorflow': tf.__version__,
                'keras': keras.__version__,
            }
        }

    def predict_for_user(self, transactions: list[dict], budget: float = 30000.0) -> dict:
        """
        Runs real LSTM inference on user's chronological expense transactions.
        Requires at least 30 calendar days of span to construct the 30-day input window.
        """
        if not self.is_available():
            return {
                'status': 'unavailable',
                'engine': 'Unavailable (model artifacts missing)',
                'predicted_total': 0.0,
                'budget': budget,
                'overrun': 0.0,
                'highest_increase_category': 'None',
                'category_predictions': {},
                'months_of_history': 0,
                'days_of_history': 0,
                'warning': 'Trained LSTM model artifacts are currently unavailable.',
            }

        # Filter strictly to Expense transactions
        expenses = [t for t in transactions if t.get('type') == 'Expense']
        if not expenses:
            return {
                'status': 'insufficient_history',
                'engine': 'TensorFlow/Keras LSTM',
                'predicted_total': 0.0,
                'budget': budget,
                'overrun': 0.0,
                'highest_increase_category': 'None',
                'category_predictions': {},
                'months_of_history': 0,
                'days_of_history': 0,
                'warning': 'No expense transactions recorded yet. Add spending history to generate forecasts.',
            }

        # Convert to daily time series
        df = pd.DataFrame(expenses)
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        
        min_date = df['date'].min()
        max_date = df['date'].max()
        days_span = (max_date - min_date).days + 1
        months_count = len(df['date'].dt.to_period('M').unique())

        if days_span < 30:
            return {
                'status': 'insufficient_history',
                'engine': 'TensorFlow/Keras LSTM',
                'predicted_total': 0.0,
                'budget': budget,
                'overrun': 0.0,
                'highest_increase_category': 'None',
                'category_predictions': {},
                'months_of_history': months_count,
                'days_of_history': days_span,
                'warning': f'ExpenseCast requires at least 30 calendar days of transaction history (currently {days_span} days).',
            }

        # Build full calendar daily index over the last 30 days up to max_date
        window_start = max_date - timedelta(days=29)
        cal_idx = pd.date_range(start=window_start, end=max_date, freq='D')

        # Filter to the 30-day window
        window_df = df[(df['date'] >= window_start) & (df['date'] <= max_date)]

        # Pivot categories
        cat_pivot = window_df.pivot_table(
            index='date', columns='category', values='amount', aggfunc='sum', fill_value=0.0
        )

        # Standardize categories against EXPENSE_CATEGORIES
        for c in EXPENSE_CATEGORIES:
            if c not in cat_pivot.columns:
                cat_pivot[c] = 0.0
        cat_pivot = cat_pivot[EXPENSE_CATEGORIES].reindex(cal_idx, fill_value=0.0)

        # Compute total
        cat_pivot.insert(0, 'total', cat_pivot.sum(axis=1))

        # Calendar features
        cat_pivot['dow'] = cat_pivot.index.dayofweek / 6.0
        cat_pivot['dom'] = (cat_pivot.index.day - 1) / 30.0
        cat_pivot['month'] = (cat_pivot.index.month - 1) / 11.0

        # Input array of shape (1, 30, 12)
        X_user = cat_pivot[FEATURE_COLUMNS].values.reshape(1, 30, len(FEATURE_COLUMNS))

        # Scale using fitted scaler_X
        X_user_scaled = self.scaler_X.transform(X_user.reshape(-1, X_user.shape[-1])).reshape(X_user.shape)

        # Run genuine TensorFlow inference
        pred_scaled = self.model.predict(X_user_scaled, verbose=0)
        pred_raw = self.scaler_Y.inverse_transform(pred_scaled)[0]
        # Clip negative amounts
        pred_raw = np.maximum(0.0, pred_raw)

        predicted_total = float(round(pred_raw[0], 2))
        category_preds = {}
        for idx, cat in enumerate(EXPENSE_CATEGORIES, start=1):
            category_preds[cat] = float(round(pred_raw[idx], 2))

        # If user categories had aliases (e.g. Food vs Food & Drink), normalize
        # Previous 30-day spending per category to determine highest increase
        prev_spending = {}
        for cat in EXPENSE_CATEGORIES:
            prev_spending[cat] = float(cat_pivot[cat].sum())

        increases = {cat: category_preds[cat] - prev_spending.get(cat, 0.0) for cat in EXPENSE_CATEGORIES}
        highest_increase_cat = max(increases.items(), key=lambda x: x[1])[0]

        overrun = float(round(max(0.0, predicted_total - budget), 2))

        return {
            'status': 'success',
            'engine': 'TensorFlow/Keras LSTM',
            'predicted_total': predicted_total,
            'budget': float(budget),
            'overrun': overrun,
            'highest_increase_category': highest_increase_cat,
            'category_predictions': category_preds,
            'months_of_history': months_count,
            'days_of_history': days_span,
            'model_info': {
                'window_days': 30,
                'horizon_days': 30,
                'model_name': 'ExpenseCast_LSTM',
                'disclaimer': 'Forecasts are estimates generated by a machine-learning model from historical patterns, not financial guarantees.',
            }
        }

# Global singleton service
ml_service = MLService()
