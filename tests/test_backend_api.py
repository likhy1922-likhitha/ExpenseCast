"""
Tests for FastAPI Backend Endpoints, Auth, User Isolation, CRUD, and Forecast
"""

import sys
import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.main import app, db

client = TestClient(app)

def test_health_endpoints():
    r1 = client.get('/health')
    assert r1.status_code == 200
    assert r1.json()['status'] == 'ok'

    r2 = client.get('/api/health')
    assert r2.status_code == 200
    assert r2.json()['status'] == 'ok'

def test_ml_status_endpoint():
    r = client.get('/api/ml/status')
    assert r.status_code == 200
    data = r.json()
    assert data['available'] is True
    assert 'TensorFlow/Keras' in data['engine']

def test_unauthenticated_access_blocked():
    # Calling protected endpoints without cookies must return 401
    assert client.get('/api/auth/me').status_code == 401
    assert client.get('/api/transactions').status_code == 401
    assert client.get('/api/summary').status_code == 401
    assert client.get('/api/forecast').status_code == 401
    assert client.get('/api/goals').status_code == 401
    assert client.get('/api/bills').status_code == 401

def test_auth_flow_and_user_isolation():
    # User A Signup
    email_a = "user_a@example.com"
    r_signup = client.post('/api/auth/signup', json={
        'name': 'User A',
        'email': email_a,
        'password': 'password123'
    })
    # If already exists from prior run, log in
    if r_signup.status_code == 409:
        r_login = client.post('/api/auth/login', json={'email': email_a, 'password': 'password123'})
        assert r_login.status_code == 200
        cookie_a = r_login.cookies.get('expensecast_session')
    else:
        assert r_signup.status_code == 200
        cookie_a = r_signup.cookies.get('expensecast_session')

    headers_a = {'Cookie': f'expensecast_session={cookie_a}'}

    # User A adds a transaction
    tx_resp = client.post('/api/transactions', headers=headers_a, json={
        'date': '2026-09-01',
        'type': 'Expense',
        'amount': 500.0,
        'merchant': 'User A Coffee',
        'category': 'Food & Drink',
        'recurring': False
    })
    assert tx_resp.status_code == 200
    tx_a_id = tx_resp.json()['id']

    # User A adds a goal
    goal_resp = client.post('/api/goals', headers=headers_a, json={
        'name': 'User A Vacation',
        'target': 10000.0,
        'current': 1000.0
    })
    assert goal_resp.status_code == 200
    goal_a_id = goal_resp.json()['id']

    # User B Signup
    email_b = "user_b@example.com"
    r_signup_b = client.post('/api/auth/signup', json={
        'name': 'User B',
        'email': email_b,
        'password': 'password123'
    })
    if r_signup_b.status_code == 409:
        r_login_b = client.post('/api/auth/login', json={'email': email_b, 'password': 'password123'})
        cookie_b = r_login_b.cookies.get('expensecast_session')
    else:
        cookie_b = r_signup_b.cookies.get('expensecast_session')

    headers_b = {'Cookie': f'expensecast_session={cookie_b}'}

    # User B reads transactions: must NOT see User A's transaction
    b_txs = client.get('/api/transactions', headers=headers_b).json()['transactions']
    assert not any(t['id'] == tx_a_id for t in b_txs), "User B must not see User A transactions"

    # User B tries to delete User A's transaction
    client.delete(f'/api/transactions/{tx_a_id}', headers=headers_b)

    # User A still has the transaction
    a_txs = client.get('/api/transactions', headers=headers_a).json()['transactions']
    assert any(t['id'] == tx_a_id for t in a_txs), "User A transaction must remain intact after unauthorized delete attempt"

    # User B tries to delete User A's goal
    client.delete(f'/api/goals/{goal_a_id}', headers=headers_b)
    a_goals = client.get('/api/goals', headers=headers_a).json()['goals']
    assert any(g['id'] == goal_a_id for g in a_goals), "User A goal must remain intact after unauthorized delete attempt"

def test_transactions_crud_and_validation():
    # Login user A
    r_login = client.post('/api/auth/login', json={'email': 'user_a@example.com', 'password': 'password123'})
    cookie = r_login.cookies.get('expensecast_session')
    headers = {'Cookie': f'expensecast_session={cookie}'}

    # Invalid amount <= 0
    bad_amt = client.post('/api/transactions', headers=headers, json={
        'date': '2026-09-02', 'type': 'Expense', 'amount': -50.0, 'merchant': 'Bad', 'category': 'Other'
    })
    assert bad_amt.status_code == 400

    # Invalid type
    bad_type = client.post('/api/transactions', headers=headers, json={
        'date': '2026-09-02', 'type': 'InvalidType', 'amount': 100.0, 'merchant': 'Bad', 'category': 'Other'
    })
    assert bad_type.status_code == 400

    # Valid recurring transaction (bill)
    bill_resp = client.post('/api/transactions', headers=headers, json={
        'date': '2026-09-05', 'type': 'Expense', 'amount': 1200.0, 'merchant': 'Broadband Bill', 'category': 'Utilities', 'recurring': True
    })
    assert bill_resp.status_code == 200
    bill_id = bill_resp.json()['id']

    # Bills endpoint should return it
    bills = client.get('/api/bills', headers=headers).json()['bills']
    assert any(b['id'] == bill_id for b in bills), "Recurring transaction must appear in bills endpoint"

    # Delete transaction
    del_resp = client.delete(f'/api/transactions/{bill_id}', headers=headers)
    assert del_resp.status_code == 200

def test_csv_import_endpoint():
    r_login = client.post('/api/auth/login', json={'email': 'user_a@example.com', 'password': 'password123'})
    cookie = r_login.cookies.get('expensecast_session')
    headers = {'Cookie': f'expensecast_session={cookie}'}

    csv_data = "date,type,amount,merchant,category,recurring\n2026-08-01,Expense,350,Bookstore,Shopping,false\n2026-08-02,Expense,800,Dinner,Food & Drink,false\n"
    files = {'file': ('test.csv', io.BytesIO(csv_data.encode('utf-8')), 'text/csv')}
    r = client.post('/api/transactions/import', headers=headers, files=files)
    assert r.status_code == 200
    assert r.json()['imported'] == 2

def test_budget_and_forecast_endpoints():
    r_login = client.post('/api/auth/login', json={'email': 'user_a@example.com', 'password': 'password123'})
    cookie = r_login.cookies.get('expensecast_session')
    headers = {'Cookie': f'expensecast_session={cookie}'}

    from datetime import datetime
    cur_month = datetime.now().strftime('%Y-%m')
    # Set budget
    r_budget = client.put('/api/budget', headers=headers, json={'month': cur_month, 'amount': 25000.0})
    assert r_budget.status_code == 200

    # Summary
    r_summary = client.get('/api/summary', headers=headers)
    assert r_summary.status_code == 200
    assert r_summary.json()['budget'] == 25000.0

    # Forecast
    r_fc = client.get('/api/forecast', headers=headers)
    assert r_fc.status_code == 200
    fc_data = r_fc.json()
    assert 'engine' in fc_data
    assert 'predicted_total' in fc_data
    assert 'overrun' in fc_data
