"""
ExpenseCast - Demo Mode Service
Provides an isolated, read-only data access layer for Demo Mode using the
supplied Personal_Finance_Dataset.csv and the genuine TensorFlow/Keras LSTM model.
Does not write to or read from the production database.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from backend.ml_service import ml_service, EXPENSE_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = ROOT / 'machine-learning' / 'data' / 'raw' / 'Personal_Finance_Dataset.csv'

# Standard demo benchmark budget explicitly labeled for demonstration
DEMO_BENCHMARK_BUDGET = 20000.0

class DemoService:
    def __init__(self, raw_csv: Path | None = None):
        self.raw_csv = raw_csv or RAW_CSV_PATH
        self._cached_transactions = None
        self._cached_summary = None
        self._cached_forecast = None

    def _load_raw_transactions(self) -> list[dict]:
        """Loads and formats transactions from the supplied CSV in chronological order."""
        if not self.raw_csv.exists():
            raise FileNotFoundError(f"Demo dataset not found at {self.raw_csv}")

        df = pd.read_csv(self.raw_csv)
        # Sort chronologically descending for display (latest first)
        df['Date_dt'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date_dt', 'Amount'], ascending=[False, False]).reset_index(drop=True)

        transactions = []
        for idx, row in df.iterrows():
            typ = str(row['Type']).strip()
            cat = str(row['Category']).strip()
            # Recurring flag for demonstration: Rent and Utilities naturally recurring
            is_recurring = cat in ('Rent', 'Utilities') and typ == 'Expense'
            transactions.append({
                'id': int(idx + 1),
                'date': str(row['Date']),
                'type': typ,
                'amount': float(row['Amount']),
                'merchant': str(row['Transaction Description']),
                'category': cat,
                'recurring': int(is_recurring),
                'is_demo': True
            })
        return transactions

    def get_transactions(self) -> list[dict]:
        if self._cached_transactions is None:
            self._cached_transactions = self._load_raw_transactions()
        return self._cached_transactions

    def get_summary(self) -> dict:
        """
        Calculates summary statistics from the supplied dataset.
        Uses the latest complete calendar month in the dataset as the reference period.
        """
        txs = self.get_transactions()
        # Find latest month in dataset (e.g. 2024-12)
        dates = [t['date'] for t in txs]
        latest_month = max(dates)[:7]

        month_income = sum(t['amount'] for t in txs if t['type'] == 'Income' and t['date'][:7] == latest_month)
        month_spent = sum(t['amount'] for t in txs if t['type'] == 'Expense' and t['date'][:7] == latest_month)
        all_income = sum(t['amount'] for t in txs if t['type'] == 'Income')
        all_exp = sum(t['amount'] for t in txs if t['type'] == 'Expense')

        cats = {}
        for t in txs:
            if t['type'] == 'Expense' and t['date'][:7] == latest_month:
                cats[t['category']] = cats.get(t['category'], 0.0) + t['amount']

        return {
            'is_demo': True,
            'reference_month': latest_month,
            'income': round(month_income, 2),
            'spent': round(month_spent, 2),
            'balance': round(all_income - all_exp, 2),
            'savings': 0.0,
            'investments': 0.0,
            'budget': DEMO_BENCHMARK_BUDGET,
            'budget_note': 'Demo benchmark budget (₹20,000)',
            'categories': {k: round(v, 2) for k, v in cats.items()},
            'recent': txs[:8]
        }

    def get_forecast(self) -> dict:
        """
        Executes genuine TensorFlow/Keras LSTM inference on the 1,500 demo transactions
        from the supplied dataset, using the benchmark budget.
        """
        txs = self.get_transactions()
        # Predict using actual ML service on the demo transactions
        res = ml_service.predict_for_user(txs, budget=DEMO_BENCHMARK_BUDGET)
        res['is_demo'] = True
        res['dataset_label'] = 'Personal_Finance_Dataset.csv (1,500 records)'
        res['budget_label'] = f'Configured demo benchmark budget ({DEMO_BENCHMARK_BUDGET:.0f})'
        return res

    def get_bills(self) -> list[dict]:
        """Returns recurring payments identified in the demo dataset."""
        txs = self.get_transactions()
        bills = [t for t in txs if t['recurring'] == 1]
        # Return unique or latest recurring commitments
        return bills[:15]

    def get_goals(self) -> list[dict]:
        """Returns sample demonstration goals (clearly labeled as demo)."""
        return [
            {
                'id': 101,
                'name': 'Emergency Reserve (Sample Demo)',
                'target': 50000.0,
                'current': 32500.0,
                'is_demo': True
            },
            {
                'id': 102,
                'name': 'Annual Insurance Fund (Sample Demo)',
                'target': 18000.0,
                'current': 12000.0,
                'is_demo': True
            },
            {
                'id': 103,
                'name': 'Technology Upgrade (Sample Demo)',
                'target': 75000.0,
                'current': 45000.0,
                'is_demo': True
            }
        ]

demo_service = DemoService()
