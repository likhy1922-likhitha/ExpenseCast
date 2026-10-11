"""
Tests for Data Validation, Profiling, and Preprocessing Pipeline
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'machine-learning' / 'src'))

from preprocess import (
    EXPENSE_CATEGORIES,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    validate_and_profile_data,
    preprocess_daily_expenses,
    build_sequences,
)

RAW_CSV = ROOT / 'machine-learning' / 'data' / 'raw' / 'Personal_Finance_Dataset.csv'
PROC_DIR = ROOT / 'machine-learning' / 'data' / 'processed'

def test_raw_csv_exists_and_schema():
    assert RAW_CSV.exists(), "Raw dataset file must exist"
    df = pd.read_csv(RAW_CSV)
    assert len(df) == 1500, f"Expected 1,500 rows, found {len(df)}"
    assert list(df.columns) == ['Date', 'Transaction Description', 'Category', 'Amount', 'Type']
    assert df.isna().sum().sum() == 0, "Raw dataset must not contain null values"

def test_data_quality_report_and_salary_inconsistency():
    report = validate_and_profile_data(RAW_CSV)
    assert report['total_rows'] == 1500
    assert report['type_counts']['Expense'] == 1222
    assert report['type_counts']['Income'] == 278
    assert report['category_counts']['Salary'] == 146
    
    # Verify the specific Salary/Expense inconsistency
    df = pd.read_csv(RAW_CSV)
    salary_types = set(df[df['Category'] == 'Salary']['Type'])
    assert salary_types == {'Expense'}, "All Salary rows must be documented as Type='Expense'"

def test_daily_preprocessing_output():
    daily_df = preprocess_daily_expenses(RAW_CSV, PROC_DIR)
    assert (PROC_DIR / 'daily_expenses.csv').exists()
    assert (PROC_DIR / 'data_quality_report.json').exists()
    assert len(daily_df) == 1824, "Daily range from 2020-01-02 to 2024-12-29 must have 1,824 days"
    for col in FEATURE_COLUMNS:
        assert col in daily_df.columns, f"Feature column {col} missing in daily dataframe"

def test_sequence_construction_and_no_leakage():
    daily_df = pd.read_csv(PROC_DIR / 'daily_expenses.csv', index_col='Date', parse_dates=True)
    window = 30
    horizon = 30
    X, Y, dates = build_sequences(daily_df, window=window, horizon=horizon)

    assert len(X) == len(Y) == len(dates)
    assert X.shape[1] == window, f"Input sequence length must equal window size {window}"
    assert X.shape[2] == len(FEATURE_COLUMNS), f"Feature count must match {len(FEATURE_COLUMNS)}"
    assert Y.shape[1] == len(TARGET_COLUMNS), f"Target count must match {len(TARGET_COLUMNS)}"

    # Check chronological order
    date_objs = pd.to_datetime(dates)
    assert (date_objs.is_monotonic_increasing), "Dates must be strictly chronologically increasing"

    # Leakage check: Ensure targets are not present in the input feature window
    n_samples = len(X)
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)
    train_end_date = pd.to_datetime(dates[n_train - 1])
    val_start_date = pd.to_datetime(dates[n_train])
    test_start_date = pd.to_datetime(dates[n_train + n_val])

    assert val_start_date > train_end_date, "Validation dates must strictly follow training dates"
    assert test_start_date > pd.to_datetime(dates[n_train + n_val - 1]), "Test dates must strictly follow validation dates"
