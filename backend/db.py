"""
Database abstraction layer supporting SQLite (development) and PostgreSQL / Supabase (production).
Provides connection management, schema initialization, parameter translation, and helper queries.
"""

import os
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger("expensecast.db")

ROOT = Path(__file__).resolve().parents[1]
SQLITE_DB_PATH = ROOT / "expensecast.db"

# Optional PostgreSQL driver
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2 import IntegrityError as PGIntegrityError
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    PGIntegrityError = sqlite3.IntegrityError


def get_database_url() -> str | None:
    """Return sanitized DATABASE_URL from environment if configured."""
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    url = url.strip()
    # Handle postgres:// legacy prefix from Supabase / Render
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def is_postgres() -> bool:
    """Check if the application is configured to use PostgreSQL."""
    return bool(get_database_url())


def is_unique_violation(e: Exception) -> bool:
    """Check whether an exception represents a unique constraint violation across SQLite and PostgreSQL."""
    if isinstance(e, sqlite3.IntegrityError):
        return True
    if PSYCOPG2_AVAILABLE and isinstance(e, PGIntegrityError):
        return True
    msg = str(e).lower()
    return "unique constraint" in msg or "duplicate key" in msg or "already exists" in msg


class DBConnection:
    """
    Unified database connection wrapper supporting both SQLite and PostgreSQL.
    Provides dict-like row access, unified parameter substitution (? -> %s for PG),
    and consistent lastrowid resolution.
    """

    def __init__(self, raw_conn, is_pg: bool = False):
        self._conn = raw_conn
        self.is_pg = is_pg

    def execute(self, sql: str, params: tuple | list = ()):
        """
        Execute SQL statement. Automatically adapts ? placeholders to %s when connected to PostgreSQL.
        Returns a cursor with dict-like row mapping.
        """
        if self.is_pg:
            cur = self._conn.cursor(cursor_factory=RealDictCursor)
            adapted_sql = sql.replace("?", "%s")
            cur.execute(adapted_sql, params)
        else:
            cur = self._conn.cursor()
            cur.execute(sql, params)
        return cur

    def execute_insert(self, sql: str, params: tuple | list = (), returning_col: str = "id") -> int:
        """
        Execute an INSERT statement and return the inserted primary key ID.
        In PostgreSQL: appends RETURNING id.
        In SQLite: uses cursor.lastrowid.
        """
        if self.is_pg:
            cur = self._conn.cursor(cursor_factory=RealDictCursor)
            adapted_sql = sql.replace("?", "%s")
            if "RETURNING" not in adapted_sql.upper():
                adapted_sql = f"{adapted_sql.rstrip(';')} RETURNING {returning_col}"
            cur.execute(adapted_sql, params)
            row = cur.fetchone()
            inserted_id = row[returning_col] if row else None
            return inserted_id
        else:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            return cur.lastrowid

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()


def get_connection() -> DBConnection:
    """
    Get a new database connection.
    Connects to PostgreSQL if DATABASE_URL is set, otherwise SQLite.
    In production (e.g. Render), if DATABASE_URL is set but connection fails, raises an error.
    """
    db_url = get_database_url()
    is_render = bool(os.getenv("RENDER"))

    if db_url:
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError(
                "DATABASE_URL is set but 'psycopg2-binary' is not installed."
            )
        try:
            conn = psycopg2.connect(db_url)
            return DBConnection(conn, is_pg=True)
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL database: %s", str(e))
            raise RuntimeError(f"PostgreSQL database connection failed: {e}")

    # Fallback to local SQLite only when DATABASE_URL is NOT set
    if is_render:
        logger.warning(
            "Running on Render without DATABASE_URL configured! Local SQLite will be reset on restarts."
        )

    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    conn.row_factory = sqlite3.Row
    return DBConnection(conn, is_pg=False)


def init_db():
    """
    Initialize required tables safely without dropping existing user data.
    """
    c = get_connection()
    try:
        cur = c._conn.cursor()
        if c.is_pg:
            # PostgreSQL Schema
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at VARCHAR(100) NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                date VARCHAR(50) NOT NULL,
                type VARCHAR(50) NOT NULL,
                amount DOUBLE PRECISION NOT NULL,
                merchant VARCHAR(255) NOT NULL,
                category VARCHAR(100) NOT NULL,
                recurring INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS budgets (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                month VARCHAR(20) NOT NULL,
                amount DOUBLE PRECISION NOT NULL,
                UNIQUE(user_id, month)
            );

            CREATE TABLE IF NOT EXISTS goals (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                target DOUBLE PRECISION NOT NULL,
                current DOUBLE PRECISION NOT NULL DEFAULT 0
            );
            """)
        else:
            # SQLite Schema
            cur.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                type TEXT NOT NULL,
                amount REAL NOT NULL,
                merchant TEXT NOT NULL,
                category TEXT NOT NULL,
                recurring INTEGER DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                month TEXT NOT NULL,
                amount REAL NOT NULL,
                UNIQUE(user_id, month)
            );

            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                target REAL NOT NULL,
                current REAL NOT NULL DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """)
        c.commit()
    finally:
        c.close()
