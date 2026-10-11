"""
Tests for Demo Mode, Read-Only Enforcement, Data Isolation, and UI Cleanup
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.main import app

client = TestClient(app)

def test_demo_endpoints_accessible_without_auth():
    """Confirms demo endpoints are publicly accessible and read-only without requiring user login."""
    # Status
    r_status = client.get('/api/demo/status')
    assert r_status.status_code == 200
    data_status = r_status.json()
    assert data_status['mode'] == 'demo'
    assert data_status['read_only'] is True
    assert data_status['total_records'] == 1500

    # Transactions
    r_tx = client.get('/api/demo/transactions')
    assert r_tx.status_code == 200
    txs = r_tx.json()['transactions']
    assert len(txs) == 1500
    assert txs[0]['is_demo'] is True

    # Summary
    r_sum = client.get('/api/demo/summary')
    assert r_sum.status_code == 200
    summary = r_sum.json()
    assert summary['is_demo'] is True
    assert summary['budget'] == 20000.0
    assert summary['spent'] > 0
    assert 'categories' in summary

    # Forecast with genuine LSTM
    r_fc = client.get('/api/demo/forecast')
    assert r_fc.status_code == 200
    fc = r_fc.json()
    assert fc['is_demo'] is True
    assert fc['status'] == 'success'
    assert fc['engine'] == 'TensorFlow/Keras LSTM'
    assert fc['predicted_total'] > 0
    assert len(fc['category_predictions']) == 8

    # Bills & Goals
    r_bills = client.get('/api/demo/bills')
    assert r_bills.status_code == 200
    assert len(r_bills.json()['bills']) > 0

    r_goals = client.get('/api/demo/goals')
    assert r_goals.status_code == 200
    assert len(r_goals.json()['goals']) == 3

def test_demo_endpoints_reject_all_mutations():
    """Confirms Demo Mode is strictly read-only and rejects POST/PUT/DELETE with 403 Forbidden."""
    assert client.post('/api/demo/transactions', json={'amount': 100}).status_code == 403
    assert client.delete('/api/demo/transactions/1').status_code == 403
    assert client.put('/api/demo/budget', json={'amount': 25000}).status_code == 403
    assert client.post('/api/demo/goals', json={'name': 'Test'}).status_code == 403
    assert client.delete('/api/demo/goals/101').status_code == 403
    assert client.post('/api/demo/transactions/import').status_code == 403

def test_demo_and_normal_user_isolation():
    """Confirms demo data is never mixed with normal user data and normal user does not fall back to demo."""
    # Create a fresh isolated normal user
    email = "isolated_user@example.com"
    r_signup = client.post('/api/auth/signup', json={
        'name': 'Isolated User',
        'email': email,
        'password': 'password123'
    })
    if r_signup.status_code == 409:
        r_login = client.post('/api/auth/login', json={'email': email, 'password': 'password123'})
        cookie = r_login.cookies.get('expensecast_session')
    else:
        cookie = r_signup.cookies.get('expensecast_session')

    headers = {'Cookie': f'expensecast_session={cookie}'}

    # Normal user's transactions: must NOT contain the 1,500 demo records
    normal_txs = client.get('/api/transactions', headers=headers).json()['transactions']
    assert len(normal_txs) != 1500
    assert not any(t.get('is_demo') is True for t in normal_txs)

    # Normal user with 0 or < 30 days history must NOT get demo forecast
    normal_fc = client.get('/api/forecast', headers=headers).json()
    assert normal_fc.get('is_demo') is not True
    assert normal_fc['status'] == 'insufficient_history'
    assert normal_fc['predicted_total'] == 0.0

def test_removed_forecast_engine_ui_elements():
    """Confirms the two specific visible forecast engine status elements are removed from index.html."""
    html_path = ROOT / 'index.html'
    content = html_path.read_text(encoding='utf-8')

    # 1. Bottom-left sidebar card labelled 'Forecast engine'
    assert 'class="engine"' not in content, "Sidebar 'Forecast engine' card must be removed"
    assert '<b>Forecast engine</b>' not in content, "Sidebar 'Forecast engine' label must be removed"

    # 2. Badge at top-right of Forecast page labelled 'TensorFlow/Keras LSTM'
    assert 'class="model"' not in content, "Forecast page 'model' badge must be removed"
    assert '● <span id="modelName">' not in content, "Model badge span must be removed"

    # 3. Verify Demo Mode triggers are present
    assert 'id="btnTryDemo"' in content, "Demo mode button must be present on login view"
    assert 'id="demoBadge"' in content, "Demo mode badge must be present in header"
    assert 'id="demoBanner"' in content, "Demo mode banner must be present"
