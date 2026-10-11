# ExpenseCast — Personal Finance & Expense Forecasting

ExpenseCast is a full-stack personal finance web application featuring secure authentication, transaction tracking, CSV batch import, monthly budget management, savings goals, recurring bills tracking, and future expense forecasting powered by a genuine **TensorFlow/Keras Sequential LSTM**.

---

## 1. Stack & Architecture

- **Frontend**: Single-Page Application (HTML5, Modern CSS3, Vanilla ES6 JavaScript, Chart.js 4.4.9, Google Fonts `DM Sans` & `Manrope`).
- **Backend**: FastAPI 0.115+ (Python 3.10+ / Python 3.12).
- **Database**:
  - Development / Local: SQLite (`expensecast.db`) with zero external setup required.
  - Production / Supabase: PostgreSQL via `DATABASE_URL` connection string.
- **Authentication**: Salted PBKDF2-SHA256 password hashing (180,000 iterations) + HMAC-SHA256 signed HTTP-only cookies (`expensecast_session`).
- **Machine Learning**: Genuine Sequential LSTM built with TensorFlow 2.x and Keras 3.x trained offline on 5 years of chronological spending data (`Personal_Finance_Dataset.csv`).
- **Demo Mode**: Instant one-click demonstration for evaluators and guides using an isolated read-only data layer reading `Personal_Finance_Dataset.csv` with genuine LSTM model forecasts and benchmark budgets. Zero database writes.

---

## 2. Machine Learning Pipeline

- **Training Data**: Located at `machine-learning/data/raw/Personal_Finance_Dataset.csv` (1,500 transactions spanning 2020 to 2024).
- **Architecture**:
  - Input Window: 30 days $\times$ 12 features (daily total expense, 8 categories, normalized calendar signals).
  - Layers: `LSTM(64, return_sequences=True)` $\to$ `Dropout(0.2)` $\to$ `LSTM(32)` $\to$ `Dropout(0.2)` $\to$ `Dense(16, relu)` $\to$ `Dense(9)`.
  - Forecast Horizon: 30 days ahead forward total expense and category-level spending.
- **Evaluation on Held-Out Test Set (266 samples)**:
  - LSTM Model MAE: **4,180.08** (16.0% lower error than the naive last-window baseline of 4,974.67).
  - LSTM Model RMSE: **4,968.16** (21.5% lower error than naive baseline of 6,328.85).
- **Inference**: Loaded in FastAPI via `backend/ml_service.py` with zero request-time training overhead. Returns honest warnings when user history is below the required 30-day window.

---

## 3. Local Setup & Running

Requires Python 3.10+.

```bash
# Clone and enter workspace
cd ExpenseCast

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Re-train LSTM model if needed:
python machine-learning/src/train.py

# Run backend with uvicorn
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000` in your web browser.

---

## 4. Running Automated Tests

Run the complete test suite (17 tests covering data validation, preprocessing, leakage prevention, ML smoke training, model inference, authentication, user isolation, and CRUD operations):

```bash
pytest tests/ -v
```

---

## 5. API Reference

### Health & ML Status
- `GET /health` — Root health check.
- `GET /api/health` — API health check.
- `GET /api/ml/status` — Model loaded status, architecture, window size, and test evaluation metrics.

### Authentication
- `POST /api/auth/signup` — Create user account (`name`, `email`, `password`).
- `POST /api/auth/login` — Sign in and receive signed HTTP-only session cookie.
- `POST /api/auth/logout` — Clear session cookie.
- `GET /api/auth/me` — Return current authenticated user profile.

### Transactions
- `GET /api/transactions` — Get all transactions for authenticated user.
- `POST /api/transactions` — Add transaction (`date`, `type`, `amount`, `merchant`, `category`, `recurring`).
- `DELETE /api/transactions/{id}` — Delete transaction scoped to authenticated user.
- `POST /api/transactions/import` — Upload and import CSV file (supports standard format and supplied dataset schema).

### Planning & Analytics
- `GET /api/summary` — Current month income, spent, balance, budget, and category spending.
- `GET /api/forecast` — Run TensorFlow LSTM forecast on user transaction history.
- `PUT /api/budget` — Update budget for a specified month (`month`, `amount`).
- `GET /api/goals` — List user savings goals.
- `POST /api/goals` — Create a new goal (`name`, `target`, `current`).
- `DELETE /api/goals/{id}` — Delete a goal.
- `GET /api/bills` — List active recurring transactions / bills.

### Demo Mode (Read-Only)
- `GET /api/demo/status` — Status of Demo Mode and sample dataset.
- `GET /api/demo/transactions` — 1,500 demo transactions from `Personal_Finance_Dataset.csv`.
- `GET /api/demo/summary` — Reference period income, spent, and benchmark budget.
- `GET /api/demo/forecast` — Genuine TensorFlow LSTM forecast on supplied dataset.
- `GET /api/demo/bills` — Sample recurring commitments.
- `GET /api/demo/goals` — Sample demonstration goals.

---

## 6. Deployment (Render & Supabase)

The service is preconfigured for Render deployment via `render.yaml`.

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `EXPENSECAST_SECRET` = A strong random secret key for HMAC signatures.
  - `DATABASE_URL` = (Optional) Supabase / PostgreSQL database connection URI. Defaults to local SQLite if omitted.
  - `ALLOWED_ORIGINS` = Comma-separated list of allowed frontend origins for CORS.
