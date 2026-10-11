"""
ExpenseCast - Preprocessing and Data Pipeline
Loads, inspects, validates, and prepares daily expense series from the supplied dataset.
"""

import os
import json
import hashlib
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

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

def get_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()

def validate_and_profile_data(raw_csv_path: Path) -> dict:
    """
    Profiles raw CSV dataset, verifies schema, types, counts, and checks for inconsistencies.
    """
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_csv_path}")

    file_hash = get_file_hash(raw_csv_path)
    df = pd.read_csv(raw_csv_path)

    expected_cols = ['Date', 'Transaction Description', 'Category', 'Amount', 'Type']
    if list(df.columns) != expected_cols:
        raise ValueError(f"Unexpected columns: {list(df.columns)}, expected: {expected_cols}")

    total_rows = len(df)
    missing_count = int(df.isna().sum().sum())
    type_counts = {str(k): int(v) for k, v in df['Type'].value_counts().items()}
    category_counts = {str(k): int(v) for k, v in df['Category'].value_counts().items()}

    salary_types = {str(k): int(v) for k, v in df[df['Category'] == 'Salary']['Type'].value_counts().items()}
    income_categories = {str(k): int(v) for k, v in df[df['Type'] == 'Income']['Category'].value_counts().items()}

    date_min = str(df['Date'].min())
    date_max = str(df['Date'].max())
    amount_min = float(df['Amount'].min())
    amount_max = float(df['Amount'].max())
    amount_mean = float(df['Amount'].mean())

    report = {
        'dataset_file': str(raw_csv_path.name),
        'file_sha256': file_hash,
        'provenance': 'Supplied archive (Personal_Finance_Dataset.csv); public/Kaggle provenance unverified',
        'currency_declared': 'Unverified (no currency symbol in raw dataset; treated as generic monetary units)',
        'total_rows': total_rows,
        'total_columns': len(df.columns),
        'columns': expected_cols,
        'missing_values': missing_count,
        'date_range': {'min': date_min, 'max': date_max},
        'amounts': {'min': amount_min, 'max': amount_max, 'mean': amount_mean},
        'type_counts': type_counts,
        'category_counts': category_counts,
        'data_quality_concerns': [
            {
                'issue': 'Salary categorized as Expense',
                'detail': f'All {salary_types.get("Expense", 0)} records in category "Salary" are marked Type="Expense". None are marked "Income".',
                'handling': 'In accordance with non-negotiable rules, records are preserved untouched in raw data; filtering strictly uses Type=="Expense" for expenditures.'
            },
            {
                'issue': 'Limited Income categories',
                'detail': f'Only {list(income_categories.keys())} are marked as Income ({income_categories}).',
                'handling': 'Expense forecasting focuses strictly on Type=="Expense".'
            },
            {
                'issue': 'Synthetic descriptions and unverified currency',
                'detail': 'Transaction descriptions appear procedurally generated; no currency code or symbol is embedded.',
                'handling': 'Recorded honestly without inventing currency labels or external provenance.'
            }
        ]
    }
    return report

def preprocess_daily_expenses(raw_csv_path: Path, processed_dir: Path) -> pd.DataFrame:
    """
    Cleans and aggregates raw transactions into a full calendar daily expense time series.
    Outputs:
      - processed_dir / daily_expenses.csv
      - processed_dir / data_quality_report.json
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    report = validate_and_profile_data(raw_csv_path)

    # Save data quality report
    report_path = processed_dir / 'data_quality_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    df = pd.read_csv(raw_csv_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter strictly to Expense
    exp = df[df['Type'] == 'Expense'].copy()

    # Reindex to full calendar daily frequency
    full_idx = pd.date_range(start=exp['Date'].min(), end=exp['Date'].max(), freq='D')
    
    # Pivot category daily spending
    cat_daily = exp.pivot_table(index='Date', columns='Category', values='Amount', aggfunc='sum', fill_value=0.0)
    
    # Ensure all expected expense categories are present
    for cat in EXPENSE_CATEGORIES:
        if cat not in cat_daily.columns:
            cat_daily[cat] = 0.0
            
    cat_daily = cat_daily[EXPENSE_CATEGORIES].reindex(full_idx, fill_value=0.0)
    
    # Daily total
    cat_daily.insert(0, 'total', cat_daily.sum(axis=1))
    
    # Calendar features
    cat_daily['dow'] = cat_daily.index.dayofweek / 6.0
    cat_daily['dom'] = (cat_daily.index.day - 1) / 30.0
    cat_daily['month'] = (cat_daily.index.month - 1) / 11.0
    
    # Index name
    cat_daily.index.name = 'Date'
    
    # Save processed daily dataset
    processed_csv = processed_dir / 'daily_expenses.csv'
    cat_daily.to_csv(processed_csv)
    
    return cat_daily

def build_sequences(data_df: pd.DataFrame, window: int = 30, horizon: int = 30):
    """
    Constructs chronological (X, Y) sequence windows.
    At day t:
      X[i] = features from day (t - window + 1) to day t (shape: (window, n_features))
      Y[i] = sum of forward expenses from day (t + 1) to day (t + horizon) (shape: (n_targets,))
    """
    feature_data = data_df[FEATURE_COLUMNS].values
    total_series = data_df['total'].values
    cat_series = data_df[EXPENSE_CATEGORIES].values

    N = len(data_df)
    t_start = window - 1
    t_end = N - horizon

    X = []
    Y = []
    dates = []

    for t in range(t_start, t_end):
        X.append(feature_data[t - window + 1 : t + 1])
        fwd_total = np.sum(total_series[t + 1 : t + horizon + 1])
        fwd_cats = np.sum(cat_series[t + 1 : t + horizon + 1], axis=0)
        Y.append([fwd_total] + list(fwd_cats))
        dates.append(str(data_df.index[t]))

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), dates

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    raw_path = root / 'machine-learning' / 'data' / 'raw' / 'Personal_Finance_Dataset.csv'
    proc_dir = root / 'machine-learning' / 'data' / 'processed'
    print(f"Preprocessing data from {raw_path}...")
    df_proc = preprocess_daily_expenses(raw_path, proc_dir)
    print(f"Processed daily data created with shape {df_proc.shape}. Report saved.")
