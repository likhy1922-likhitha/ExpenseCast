# ExpenseCast — Full Stack

ExpenseCast is now a functional full-stack personal finance application with authentication, persistent transactions, CSV import, budgets, goals, and a forecast API.

## Stack
- Frontend: HTML, CSS, JavaScript, Chart.js
- Backend: FastAPI
- Database: SQLite
- Authentication: PBKDF2 password hashing + HTTP-only signed session cookie
- Forecast: NumPy LSTM sequence model for monthly expense history, with a moving-average fallback when there is not enough history

## Run locally
Requires Python 3.10+.

```bash
cd ExpenseCast
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`.

## CSV import
Use columns:

```text
date,type,amount,merchant,category,recurring
2026-09-01,Expense,450,Swiggy,Food,false
2026-09-02,Income,45000,Salary,Salary,false
```

Dates must be `YYYY-MM-DD` for CSV import.

## Render deployment
Create a **Web Service** (not Static Site) from this repository. The included `render.yaml` uses:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

Set an environment variable in Render:

`EXPENSECAST_SECRET` = a long random secret string.

The database is SQLite and is suitable for a college/demo deployment. For production multi-user deployment, move the database to managed PostgreSQL and use secure HTTPS cookies.

## API
- `GET /api/health`
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/me`
- `GET/POST /api/transactions`
- `DELETE /api/transactions/{id}`
- `POST /api/transactions/import`
- `GET /api/summary`
- `GET /api/forecast`
- `PUT /api/budget`
- `GET/POST /api/goals`
