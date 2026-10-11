from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import sqlite3, os, hashlib, hmac, secrets, base64, csv, io
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import numpy as np

from backend.ml_service import ml_service
from backend.demo_service import demo_service

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'expensecast.db'
DATABASE_URL = os.getenv('DATABASE_URL')
SECRET = os.getenv('EXPENSECAST_SECRET', 'expensecast-production-secret-key-default').encode()

app = FastAPI(title='ExpenseCast API', version='1.0.0')

# Configure CORS
origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ['*'] else ['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

def db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = db()
    c.executescript('''
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS transactions(
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
    CREATE TABLE IF NOT EXISTS budgets(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        month TEXT NOT NULL,
        amount REAL NOT NULL,
        UNIQUE(user_id, month)
    );
    CREATE TABLE IF NOT EXISTS goals(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        target REAL NOT NULL,
        current REAL NOT NULL DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    ''')
    c.commit()
    c.close()

init_db()

def hash_pw(pw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt, 180000)
    return base64.urlsafe_b64encode(salt + dk).decode()

def check_pw(pw: str, stored: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(stored.encode())
        salt, old = raw[:16], raw[16:]
        return hmac.compare_digest(hashlib.pbkdf2_hmac('sha256', pw.encode(), salt, 180000), old)
    except Exception:
        return False

def token(uid: int) -> str:
    payload = str(uid).encode()
    sig = hmac.new(SECRET, payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload + b'.' + sig).decode()

def uid_from_request(req: Request) -> dict:
    t = req.cookies.get('expensecast_session')
    if not t:
        raise HTTPException(401, 'Please log in')
    try:
        raw = base64.urlsafe_b64decode(t.encode())
        payload, sig = raw.split(b'.', 1)
        if not hmac.compare_digest(hmac.new(SECRET, payload, hashlib.sha256).digest(), sig):
            raise ValueError()
        uid = int(payload.decode())
        c = db()
        r = c.execute('SELECT id, name, email FROM users WHERE id=?', (uid,)).fetchone()
        c.close()
        if not r:
            raise ValueError()
        return dict(r)
    except Exception:
        raise HTTPException(401, 'Session expired. Please log in again.')

def month_key(d: str) -> str:
    return d[:7]

def get_tx(uid: int) -> list[dict]:
    c = db()
    rows = c.execute(
        'SELECT * FROM transactions WHERE user_id=? ORDER BY date DESC, id DESC',
        (uid,)
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]

# Pydantic Request Models
class Auth(BaseModel):
    name: str | None = None
    email: EmailStr
    password: str

class Tx(BaseModel):
    date: str
    type: str
    amount: float
    merchant: str
    category: str
    recurring: bool = False

class Budget(BaseModel):
    month: str
    amount: float

class Goal(BaseModel):
    name: str
    target: float
    current: float = 0.0

# Health endpoints
@app.get('/health')
def health_root():
    return {'status': 'ok', 'service': 'ExpenseCast API', 'version': '1.0.0'}

@app.get('/api/health')
def health():
    return {'status': 'ok', 'service': 'ExpenseCast API'}

# ML Status endpoint
@app.get('/api/ml/status')
def ml_status():
    return ml_service.get_status()

# Auth Endpoints
@app.post('/api/auth/signup')
def signup(a: Auth):
    if not a.name or len(a.password) < 6:
        raise HTTPException(400, 'Name and password of at least 6 characters are required')
    c = db()
    try:
        cur = c.execute(
            'INSERT INTO users(name, email, password_hash, created_at) VALUES(?, ?, ?, ?)',
            (a.name, a.email.lower(), hash_pw(a.password), datetime.now(timezone.utc).isoformat())
        )
        c.commit()
        uid = cur.lastrowid
    except sqlite3.IntegrityError:
        c.close()
        raise HTTPException(409, 'An account with this email already exists')
    c.close()
    resp = JSONResponse({'user': {'id': uid, 'name': a.name, 'email': a.email.lower()}})
    resp.set_cookie(
        'expensecast_session',
        token(uid),
        httponly=True,
        samesite='lax',
        secure=bool(os.getenv('RENDER')),
        max_age=604800
    )
    return resp

@app.post('/api/auth/login')
def login(a: Auth):
    c = db()
    r = c.execute('SELECT * FROM users WHERE email=?', (a.email.lower(),)).fetchone()
    c.close()
    if not r or not check_pw(a.password, r['password_hash']):
        raise HTTPException(401, 'Invalid email or password')
    resp = JSONResponse({'user': {'id': r['id'], 'name': r['name'], 'email': r['email']}})
    resp.set_cookie(
        'expensecast_session',
        token(r['id']),
        httponly=True,
        samesite='lax',
        secure=bool(os.getenv('RENDER')),
        max_age=604800
    )
    return resp

@app.post('/api/auth/logout')
def logout():
    resp = JSONResponse({'ok': True})
    resp.delete_cookie('expensecast_session')
    return resp

@app.get('/api/auth/me')
def me(request: Request):
    user = uid_from_request(request)
    return {'user': user}

# Transactions Endpoints
@app.get('/api/transactions')
def transactions(request: Request):
    u = uid_from_request(request)['id']
    return {'transactions': get_tx(u)}

@app.post('/api/transactions')
def add_tx(request: Request, t: Tx):
    u = uid_from_request(request)['id']
    if t.type not in ['Expense', 'Income', 'Savings', 'Investment']:
        raise HTTPException(400, 'Invalid transaction type')
    if t.amount <= 0:
        raise HTTPException(400, 'Amount must be greater than zero')
    c = db()
    cur = c.execute(
        'INSERT INTO transactions(user_id, date, type, amount, merchant, category, recurring) VALUES(?, ?, ?, ?, ?, ?, ?)',
        (u, t.date, t.type, t.amount, t.merchant, t.category, int(t.recurring))
    )
    c.commit()
    rid = cur.lastrowid
    c.close()
    return {'id': rid}

@app.delete('/api/transactions/{tid}')
def delete_tx(request: Request, tid: int):
    u = uid_from_request(request)['id']
    c = db()
    c.execute('DELETE FROM transactions WHERE id=? AND user_id=?', (tid, u))
    c.commit()
    c.close()
    return {'ok': True}

@app.post('/api/transactions/import')
def import_csv(request: Request, file: UploadFile = File(...)):
    u = uid_from_request(request)['id']
    try:
        content = file.file.read().decode('utf-8-sig')
    except Exception:
        raise HTTPException(400, 'Failed to read uploaded CSV file')

    reader = csv.DictReader(io.StringIO(content))
    imported_count = 0
    skipped_count = 0
    c = db()

    for row in reader:
        try:
            d = row.get('date') or row.get('Date')
            if not d:
                skipped_count += 1
                continue
            datetime.strptime(d.strip(), '%Y-%m-%d')

            typ = row.get('type') or row.get('Type') or 'Expense'
            typ = typ.strip()
            if typ not in ['Expense', 'Income', 'Savings', 'Investment']:
                typ = 'Expense'

            amt_raw = row.get('amount') or row.get('Amount')
            amount = float(amt_raw)
            if amount <= 0:
                skipped_count += 1
                continue

            merchant = (
                row.get('merchant')
                or row.get('Merchant')
                or row.get('Transaction Description')
                or 'Imported'
            ).strip()

            cat = (row.get('category') or row.get('Category') or 'Other').strip()
            rec = str(row.get('recurring', '')).lower() in ('1', 'true', 'yes')

            c.execute(
                'INSERT INTO transactions(user_id, date, type, amount, merchant, category, recurring) VALUES(?, ?, ?, ?, ?, ?, ?)',
                (u, d.strip(), typ, amount, merchant, cat, int(rec))
            )
            imported_count += 1
        except Exception:
            skipped_count += 1
            continue

    c.commit()
    c.close()
    return {'imported': imported_count, 'skipped': skipped_count}

# Summary Endpoint
@app.get('/api/summary')
def summary(request: Request):
    u = uid_from_request(request)['id']
    rows = get_tx(u)
    now = datetime.now()
    ym = now.strftime('%Y-%m')

    income = sum(r['amount'] for r in rows if r['type'] == 'Income' and month_key(r['date']) == ym)
    spent = sum(r['amount'] for r in rows if r['type'] == 'Expense' and month_key(r['date']) == ym)
    all_income = sum(r['amount'] for r in rows if r['type'] == 'Income')
    all_exp = sum(r['amount'] for r in rows if r['type'] == 'Expense')
    savings = sum(r['amount'] for r in rows if r['type'] == 'Savings')
    investments = sum(r['amount'] for r in rows if r['type'] == 'Investment')

    cats = {}
    for r in rows:
        if r['type'] == 'Expense' and month_key(r['date']) == ym:
            cats[r['category']] = cats.get(r['category'], 0.0) + r['amount']

    c = db()
    b = c.execute('SELECT amount FROM budgets WHERE user_id=? AND month=?', (u, ym)).fetchone()
    c.close()
    budget = float(b['amount']) if b else max(30000.0, income * 0.67 if income else 30000.0)

    return {
        'income': income,
        'spent': spent,
        'balance': all_income - all_exp - savings - investments,
        'savings': savings,
        'investments': investments,
        'budget': budget,
        'categories': cats,
        'recent': rows[:8]
    }

# Genuine Forecast Endpoints
@app.get('/api/forecast')
@app.get('/api/ml/forecast')
def forecast(request: Request):
    u = uid_from_request(request)['id']
    rows = get_tx(u)
    now = datetime.now()
    ym = now.strftime('%Y-%m')

    c = db()
    b = c.execute('SELECT amount FROM budgets WHERE user_id=? AND month=?', (u, ym)).fetchone()
    c.close()
    budget = float(b['amount']) if b else max(30000.0, sum(r['amount'] for r in rows if r['type'] == 'Income' and month_key(r['date']) == ym) * 0.67 or 30000.0)

    return ml_service.predict_for_user(rows, budget=budget)

# Budget Endpoints
@app.put('/api/budget')
def set_budget(request: Request, b: Budget):
    u = uid_from_request(request)['id']
    c = db()
    c.execute(
        'INSERT INTO budgets(user_id, month, amount) VALUES(?, ?, ?) ON CONFLICT(user_id, month) DO UPDATE SET amount=excluded.amount',
        (u, b.month, b.amount)
    )
    c.commit()
    c.close()
    return {'ok': True}

# Goals Endpoints
@app.get('/api/goals')
def goals(request: Request):
    u = uid_from_request(request)['id']
    c = db()
    rows = c.execute('SELECT * FROM goals WHERE user_id=? ORDER BY id DESC', (u,)).fetchall()
    c.close()
    return {'goals': [dict(r) for r in rows]}

@app.post('/api/goals')
def add_goal(request: Request, g: Goal):
    u = uid_from_request(request)['id']
    c = db()
    cur = c.execute(
        'INSERT INTO goals(user_id, name, target, current) VALUES(?, ?, ?, ?)',
        (u, g.name, g.target, g.current)
    )
    c.commit()
    rid = cur.lastrowid
    c.close()
    return {'id': rid}

@app.delete('/api/goals/{gid}')
def delete_goal(request: Request, gid: int):
    u = uid_from_request(request)['id']
    c = db()
    c.execute('DELETE FROM goals WHERE id=? AND user_id=?', (gid, u))
    c.commit()
    c.close()
    return {'ok': True}

# Bills Endpoint (Recurring transactions)
@app.get('/api/bills')
def bills(request: Request):
    u = uid_from_request(request)['id']
    c = db()
    rows = c.execute(
        'SELECT * FROM transactions WHERE user_id=? AND recurring=1 ORDER BY date DESC',
        (u,)
    ).fetchall()
    c.close()
    return {'bills': [dict(r) for r in rows]}

# ==========================================
# Isolated Read-Only Demo Mode Endpoints
# ==========================================
@app.get('/api/demo/status')
def demo_status():
    return {
        'mode': 'demo',
        'read_only': True,
        'dataset': 'Personal_Finance_Dataset.csv',
        'total_records': len(demo_service.get_transactions()),
        'budget': 20000.0,
        'description': 'Isolated Demo Mode reading supplied Personal_Finance_Dataset.csv with genuine TensorFlow LSTM forecasts.'
    }

@app.get('/api/demo/transactions')
def demo_transactions():
    return {'transactions': demo_service.get_transactions()}

@app.get('/api/demo/summary')
def demo_summary():
    return demo_service.get_summary()

@app.get('/api/demo/forecast')
def demo_forecast():
    return demo_service.get_forecast()

@app.get('/api/demo/bills')
def demo_bills():
    return {'bills': demo_service.get_bills()}

@app.get('/api/demo/goals')
def demo_goals():
    return {'goals': demo_service.get_goals()}

# Explicit Read-Only Guards for Demo Mode (Rejecting all mutations with 403)
@app.post('/api/demo/transactions')
@app.delete('/api/demo/transactions/{tid}')
@app.post('/api/demo/transactions/import')
@app.put('/api/demo/budget')
@app.post('/api/demo/goals')
@app.delete('/api/demo/goals/{gid}')
def demo_readonly_block():
    raise HTTPException(
        status_code=403,
        detail='Demo Mode is strictly read-only. Modifying records is disabled in Demo Mode.'
    )

# Serve existing frontend static files
@app.get('/')
def home():
    return FileResponse(ROOT / 'index.html')

app.mount('/assets', StaticFiles(directory=ROOT), name='assets')
