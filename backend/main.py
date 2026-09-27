from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
import sqlite3, os, hashlib, hmac, secrets, base64, csv, io
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/'expensecast.db'
SECRET=os.getenv('EXPENSECAST_SECRET','change-this-secret-in-production').encode()

app=FastAPI(title='ExpenseCast API', version='1.0.0')

def db():
    c=sqlite3.connect(DB)
    c.row_factory=sqlite3.Row
    return c

def init_db():
    c=db()
    c.executescript('''
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,email TEXT UNIQUE NOT NULL,password_hash TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS transactions(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,date TEXT NOT NULL,type TEXT NOT NULL,amount REAL NOT NULL,merchant TEXT NOT NULL,category TEXT NOT NULL,recurring INTEGER DEFAULT 0,FOREIGN KEY(user_id) REFERENCES users(id));
    CREATE TABLE IF NOT EXISTS budgets(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,month TEXT NOT NULL,amount REAL NOT NULL,UNIQUE(user_id,month));
    CREATE TABLE IF NOT EXISTS goals(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,name TEXT NOT NULL,target REAL NOT NULL,current REAL NOT NULL,FOREIGN KEY(user_id) REFERENCES users(id));
    ''')
    c.commit(); c.close()
init_db()

def hash_pw(pw):
    salt=secrets.token_bytes(16); dk=hashlib.pbkdf2_hmac('sha256',pw.encode(),salt,180000)
    return base64.urlsafe_b64encode(salt+dk).decode()
def check_pw(pw,stored):
    raw=base64.urlsafe_b64decode(stored.encode()); salt,old=raw[:16],raw[16:]
    return hmac.compare_digest(hashlib.pbkdf2_hmac('sha256',pw.encode(),salt,180000),old)
def token(uid):
    payload=str(uid).encode(); sig=hmac.new(SECRET,payload,hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload+b'.'+sig).decode()
def uid_from_request(req):
    t=req.cookies.get('expensecast_session')
    if not t: raise HTTPException(401,'Please log in')
    try:
        raw=base64.urlsafe_b64decode(t.encode()); payload,sig=raw.split(b'.',1)
        if not hmac.compare_digest(hmac.new(SECRET,payload,hashlib.sha256).digest(),sig): raise ValueError()
        uid=int(payload.decode());
        c=db(); r=c.execute('SELECT id,name,email FROM users WHERE id=?',(uid,)).fetchone(); c.close()
        if not r: raise ValueError()
        return r
    except Exception: raise HTTPException(401,'Session expired. Please log in again.')

def month_key(d): return d[:7]

def get_tx(uid):
    c=db(); rows=c.execute('SELECT * FROM transactions WHERE user_id=? ORDER BY date DESC,id DESC',(uid,)).fetchall(); c.close(); return [dict(r) for r in rows]

def monthly_expenses(rows):
    out={}
    for r in rows:
        if r['type']!='Expense': continue
        m=month_key(r['date']); out[m]=out.get(m,0)+float(r['amount'])
    return dict(sorted(out.items()))

# Tiny NumPy LSTM trained from scratch for monthly total forecasting.
def lstm_predict(values):
    vals=np.array(values,dtype=float)
    if len(vals)<6:
        if len(vals)==0:return 0.0,'insufficient history'
        return float(np.mean(vals[-3:] if len(vals)>=3 else vals)),'moving-average fallback'
    lo=max(1, vals.min()); hi=max(lo+1, vals.max()); scale=hi
    x=vals/scale
    seq=4; X=[]; Y=[]
    for i in range(len(x)-seq): X.append(x[i:i+seq]); Y.append(x[i+seq])
    X=np.array(X); Y=np.array(Y)
    rng=np.random.default_rng(42); h=8
    Wxi=rng.normal(0,.25,(h,1)); Whi=rng.normal(0,.25,(h,h)); bi=np.zeros(h)
    Wxf=rng.normal(0,.25,(h,1)); Whf=rng.normal(0,.25,(h,h)); bf=np.zeros(h)
    Wxo=rng.normal(0,.25,(h,1)); Who=rng.normal(0,.25,(h,h)); bo=np.zeros(h)
    Wxg=rng.normal(0,.25,(h,1)); Whg=rng.normal(0,.25,(h,h)); bg=np.zeros(h)
    Wy=rng.normal(0,.25,(1,h)); by=np.zeros(1)
    def sig(a): return 1/(1+np.exp(-np.clip(a,-30,30)))
    lr=.03
    for _ in range(180):
        grads=[np.zeros_like(z) for z in [Wxi,Whi,bi,Wxf,Whf,bf,Wxo,Who,bo,Wxg,Whg,bg,Wy,by]]
        for xx,yy in zip(X,Y):
            hs=[np.zeros(h)]; cs=[np.zeros(h)]; cache=[]
            for xt in xx:
                inp=np.array([xt])
                i=sig(Wxi@inp+Whi@hs[-1]+bi); f=sig(Wxf@inp+Whf@hs[-1]+bf); o=sig(Wxo@inp+Who@hs[-1]+bo); g=np.tanh(Wxg@inp+Whg@hs[-1]+bg)
                c=f*cs[-1]+i*g; hv=o*np.tanh(c); hs.append(hv); cs.append(c); cache.append((inp,i,f,o,g,c,hs[-2],cs[-2]))
            pred=float((Wy@hs[-1]+by)[0]); dy=pred-yy
            grads[12]+=dy*hs[-1]; grads[13]+=dy
            dh=Wy.T.flatten()*dy; dc=np.zeros(h)
            for t in reversed(range(seq)):
                inp,i,f,o,g,c,hprev,cprev=cache[t]; tanh_c=np.tanh(c)
                do=dh*tanh_c; dc=dc+dh*o*(1-tanh_c*tanh_c); di=dc*g; dg=dc*i; df=dc*cprev
                dai=di*i*(1-i); daf=df*f*(1-f); dao=do*o*(1-o); dag=dg*(1-g*g)
                grads[0]+=np.outer(dai,inp); grads[1]+=np.outer(dai,hprev); grads[2]+=dai
                grads[3]+=np.outer(daf,inp); grads[4]+=np.outer(daf,hprev); grads[5]+=daf
                grads[6]+=np.outer(dao,inp); grads[7]+=np.outer(dao,hprev); grads[8]+=dao
                grads[9]+=np.outer(dag,inp); grads[10]+=np.outer(dag,hprev); grads[11]+=dag
                dh=Whi.T@dai+Whf.T@daf+Who.T@dao+Whg.T@dag; dc=dc*f
        for p,g in zip([Wxi,Whi,bi,Wxf,Whf,bf,Wxo,Who,bo,Wxg,Whg,bg,Wy,by],grads): p-=lr*np.clip(g,-1,1)/max(1,len(X))
    # inference using last seq values
    hs=np.zeros(h); cs=np.zeros(h)
    for xt in x[-seq:]:
        inp=np.array([xt]); i=sig(Wxi@inp+Whi@hs+bi); f=sig(Wxf@inp+Whf@hs+bf); o=sig(Wxo@inp+Who@hs+bo); g=np.tanh(Wxg@inp+Whg@hs+bg); cs=f*cs+i*g; hs=o*np.tanh(cs)
    pred=float((Wy@hs+by)[0])*scale
    return max(0,pred),'LSTM sequence model'

class Auth(BaseModel): name:str|None=None; email:str; password:str
class Tx(BaseModel): date:str; type:str; amount:float; merchant:str; category:str; recurring:bool=False
class Budget(BaseModel): month:str; amount:float
class Goal(BaseModel): name:str; target:float; current:float=0

@app.get('/api/health')
def health(): return {'status':'ok','service':'ExpenseCast API'}
@app.post('/api/auth/signup')
def signup(a:Auth):
    if not a.name or len(a.password)<6: raise HTTPException(400,'Name and password of at least 6 characters are required')
    c=db()
    try:
        cur=c.execute('INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)',(a.name,a.email.lower(),hash_pw(a.password),datetime.utcnow().isoformat())); c.commit(); uid=cur.lastrowid
    except sqlite3.IntegrityError: c.close(); raise HTTPException(409,'An account with this email already exists')
    c.close(); resp=JSONResponse({'user':{'id':uid,'name':a.name,'email':a.email.lower()}}); resp.set_cookie('expensecast_session',token(uid),httponly=True,samesite='lax',secure=bool(os.getenv('RENDER')),max_age=604800); return resp
@app.post('/api/auth/login')
def login(a:Auth):
    c=db(); r=c.execute('SELECT * FROM users WHERE email=?',(a.email.lower(),)).fetchone(); c.close()
    if not r or not check_pw(a.password,r['password_hash']): raise HTTPException(401,'Invalid email or password')
    resp=JSONResponse({'user':{'id':r['id'],'name':r['name'],'email':r['email']}}); resp.set_cookie('expensecast_session',token(r['id']),httponly=True,samesite='lax',secure=bool(os.getenv('RENDER')),max_age=604800); return resp
@app.post('/api/auth/logout')
def logout():
    resp=JSONResponse({'ok':True}); resp.delete_cookie('expensecast_session'); return resp
@app.get('/api/auth/me')
def me(request:Request):
    r=uid_from_request(request); return {'user':dict(r)}

@app.get('/api/transactions')
def transactions(request:Request): return {'transactions':get_tx(uid_from_request(request)['id'])}
@app.post('/api/transactions')
def add_tx(request:Request,t:Tx):
    u=uid_from_request(request)['id']
    if t.type not in ['Expense','Income','Savings','Investment']: raise HTTPException(400,'Invalid transaction type')
    if t.amount<=0: raise HTTPException(400,'Amount must be greater than zero')
    c=db(); cur=c.execute('INSERT INTO transactions(user_id,date,type,amount,merchant,category,recurring) VALUES(?,?,?,?,?,?,?)',(u,t.date,t.type,t.amount,t.merchant,t.category,int(t.recurring))); c.commit(); rid=cur.lastrowid; c.close(); return {'id':rid}
@app.delete('/api/transactions/{tid}')
def delete_tx(request:Request,tid:int):
    u=uid_from_request(request)['id']; c=db(); c.execute('DELETE FROM transactions WHERE id=? AND user_id=?',(tid,u)); c.commit(); c.close(); return {'ok':True}

@app.post('/api/transactions/import')
def import_csv(request:Request,file:UploadFile=File(...)):
    u=uid_from_request(request)['id']; data=(file.file.read()).decode('utf-8-sig'); reader=csv.DictReader(io.StringIO(data)); count=0; c=db()
    for row in reader:
        try:
            d=row.get('date') or row.get('Date'); typ=row.get('type') or row.get('Type') or 'Expense'; amount=float(row.get('amount') or row.get('Amount')); merchant=row.get('merchant') or row.get('Merchant') or 'Imported'; cat=row.get('category') or row.get('Category') or 'Other'; rec=str(row.get('recurring','')).lower() in ('1','true','yes')
            datetime.strptime(d,'%Y-%m-%d')
            c.execute('INSERT INTO transactions(user_id,date,type,amount,merchant,category,recurring) VALUES(?,?,?,?,?,?,?)',(u,d,typ,amount,merchant,cat,int(rec))); count+=1
        except Exception: continue
    c.commit(); c.close(); return {'imported':count}

@app.get('/api/summary')
def summary(request:Request):
    u=uid_from_request(request)['id']; rows=get_tx(u); now=datetime.now(); ym=now.strftime('%Y-%m')
    income=sum(r['amount'] for r in rows if r['type']=='Income' and month_key(r['date'])==ym); spent=sum(r['amount'] for r in rows if r['type']=='Expense' and month_key(r['date'])==ym)
    all_income=sum(r['amount'] for r in rows if r['type']=='Income'); all_exp=sum(r['amount'] for r in rows if r['type']=='Expense'); savings=sum(r['amount'] for r in rows if r['type']=='Savings'); investments=sum(r['amount'] for r in rows if r['type']=='Investment')
    cats={}
    for r in rows:
        if r['type']=='Expense' and month_key(r['date'])==ym: cats[r['category']]=cats.get(r['category'],0)+r['amount']
    c=db(); b=c.execute('SELECT amount FROM budgets WHERE user_id=? AND month=?',(u,ym)).fetchone(); c.close(); budget=float(b['amount']) if b else max(30000,income*.67 if income else 30000)
    return {'income':income,'spent':spent,'balance':all_income-all_exp-savings-investments,'savings':savings,'investments':investments,'budget':budget,'categories':cats,'recent':rows[:8]}

@app.get('/api/forecast')
def forecast(request:Request):
    u=uid_from_request(request)['id']; rows=get_tx(u); monthly=monthly_expenses(rows); vals=list(monthly.values()); pred,engine=lstm_predict(vals)
    now=datetime.now(); ym=now.strftime('%Y-%m'); c=db(); b=c.execute('SELECT amount FROM budgets WHERE user_id=? AND month=?',(u,ym)).fetchone(); c.close(); budget=float(b['amount']) if b else max(30000, sum(r['amount'] for r in rows if r['type']=='Income' and month_key(r['date'])==ym)*.67)
    cats={}
    for r in rows:
        if r['type']=='Expense': cats[r['category']]=cats.get(r['category'],[])+[r['amount']]
    cp={k:round(float(np.mean(v[-3:] if len(v)>=3 else v))*1.04,2) for k,v in cats.items()}
    total_cat=sum(cp.values())
    if cp and total_cat>0:
        factor=pred/total_cat
        cp={k:round(v*factor,2) for k,v in cp.items()}
    highest=max(cp.items(),key=lambda x:x[1])[0] if cp else 'Food'
    return {'predicted_total':round(pred,2),'budget':budget,'overrun':round(max(0,pred-budget),2),'highest_increase_category':highest,'category_predictions':cp,'engine':engine,'months_of_history':len(vals)}

@app.put('/api/budget')
def set_budget(request:Request,b:Budget):
    u=uid_from_request(request)['id']; c=db(); c.execute('INSERT INTO budgets(user_id,month,amount) VALUES(?,?,?) ON CONFLICT(user_id,month) DO UPDATE SET amount=excluded.amount',(u,b.month,b.amount)); c.commit(); c.close(); return {'ok':True}
@app.get('/api/goals')
def goals(request:Request):
    u=uid_from_request(request)['id']; c=db(); rows=c.execute('SELECT * FROM goals WHERE user_id=? ORDER BY id DESC',(u,)).fetchall(); c.close(); return {'goals':[dict(r) for r in rows]}
@app.post('/api/goals')
def add_goal(request:Request,g:Goal):
    u=uid_from_request(request)['id']; c=db(); cur=c.execute('INSERT INTO goals(user_id,name,target,current) VALUES(?,?,?,?)',(u,g.name,g.target,g.current)); c.commit(); rid=cur.lastrowid; c.close(); return {'id':rid}

# Serve the existing frontend at the root.
@app.get('/')
def home(): return FileResponse(ROOT/'index.html')
app.mount('/assets',StaticFiles(directory=ROOT),name='assets')
