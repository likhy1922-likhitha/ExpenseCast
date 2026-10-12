"""
Integration tests for authentication, database persistence, session lifecycle,
and data persistence across application restarts (simulating process restarts).
"""

import sys
import os
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend.db as db_module
from backend.main import app, hash_pw, check_pw


def test_auth_full_lifecycle_and_persistence(tmp_path):
    """
    Test complete lifecycle:
    1. Register a new account
    2. Confirm account exists in DB
    3. Log out and log in with same credentials
    4. Confirm incorrect passwords rejected
    5. Confirm duplicate registration handled with 409
    6. Add test transaction and confirm saved
    7. Simulate process restart / re-opening DB and verify account & transactions persist
    8. Confirm second user cannot access first user's data
    """
    # Use isolated SQLite file for restart simulation test
    isolated_db_path = tmp_path / "lifecycle_test.db"
    orig_path = db_module.SQLITE_DB_PATH
    db_module.SQLITE_DB_PATH = isolated_db_path

    try:
        # Initialize schema
        db_module.init_db()

        client = TestClient(app)

        # 1. Register test user
        user_email = "persistent_user@example.com"
        user_pass = "SecurePass123!"
        reg_res = client.post("/api/auth/signup", json={
            "name": "Persistent Tester",
            "email": f"  {user_email.upper()}  ",  # test whitespace & uppercase normalization
            "password": user_pass
        })
        assert reg_res.status_code == 200, f"Signup failed: {reg_res.text}"
        user_data = reg_res.json()["user"]
        assert user_data["email"] == user_email
        uid = user_data["id"]

        # 2. Confirm account exists in database directly
        conn = db_module.get_connection()
        row = conn.execute("SELECT * FROM users WHERE email=?", (user_email,)).fetchone()
        conn.close()
        assert row is not None
        assert row["name"] == "Persistent Tester"
        assert check_pw(user_pass, row["password_hash"]) is True

        # 3. Log out and log in with the same credentials (trimmed & lowercased)
        client.post("/api/auth/logout")
        login_res = client.post("/api/auth/login", json={
            "email": f" {user_email} ",
            "password": user_pass
        })
        assert login_res.status_code == 200, f"Login failed: {login_res.text}"
        assert login_res.json()["user"]["id"] == uid
        session_cookie = login_res.cookies.get("expensecast_session")
        assert session_cookie is not None

        # 4. Confirm incorrect passwords rejected
        bad_login = client.post("/api/auth/login", json={
            "email": user_email,
            "password": "WrongPassword999"
        })
        assert bad_login.status_code == 401
        assert "Invalid email or password" in bad_login.json()["detail"]

        # 5. Confirm duplicate registration is rejected with 409
        dup_reg = client.post("/api/auth/signup", json={
            "name": "Duplicate Tester",
            "email": user_email,
            "password": user_pass
        })
        assert dup_reg.status_code == 409
        assert "already exists" in dup_reg.json()["detail"]

        # 6. Add transaction and verify it is saved
        headers = {"Cookie": f"expensecast_session={session_cookie}"}
        tx_res = client.post("/api/transactions", headers=headers, json={
            "date": "2026-10-12",
            "type": "Expense",
            "amount": 1450.50,
            "merchant": "Supermarket Groceries",
            "category": "Groceries",
            "recurring": False
        })
        assert tx_res.status_code == 200
        tx_id = tx_res.json()["id"]

        # 7. Simulate restart: create a new TestClient and new DB connection instance
        client_after_restart = TestClient(app)
        # Login again after restart
        relogin_res = client_after_restart.post("/api/auth/login", json={
            "email": user_email,
            "password": user_pass
        })
        assert relogin_res.status_code == 200
        new_cookie = relogin_res.cookies.get("expensecast_session")
        restart_headers = {"Cookie": f"expensecast_session={new_cookie}"}

        # Check transactions exist after simulated restart
        tx_list = client_after_restart.get("/api/transactions", headers=restart_headers).json()["transactions"]
        assert len(tx_list) >= 1
        assert any(t["id"] == tx_id and t["merchant"] == "Supermarket Groceries" for t in tx_list)

        # 8. Confirm second user cannot access first user's data
        user2_res = client.post("/api/auth/signup", json={
            "name": "Second User",
            "email": "second_user@example.com",
            "password": "Password456!"
        })
        assert user2_res.status_code == 200
        user2_cookie = user2_res.cookies.get("expensecast_session")
        user2_headers = {"Cookie": f"expensecast_session={user2_cookie}"}

        user2_txs = client.get("/api/transactions", headers=user2_headers).json()["transactions"]
        assert not any(t["id"] == tx_id for t in user2_txs), "Second user must not see first user's transactions!"

    finally:
        db_module.SQLITE_DB_PATH = orig_path


def test_postgres_sql_compatibility():
    """Verify that is_unique_violation handles various database error formats."""
    import sqlite3
    sqlite_err = sqlite3.IntegrityError("UNIQUE constraint failed: users.email")
    assert db_module.is_unique_violation(sqlite_err) is True

    generic_pg_err = Exception('duplicate key value violates unique constraint "users_email_key"')
    assert db_module.is_unique_violation(generic_pg_err) is True

    other_err = Exception("Connection timed out")
    assert db_module.is_unique_violation(other_err) is False
