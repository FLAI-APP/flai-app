# app.py — fix definitivo per movements.importo → movements.amount
import os, time
from datetime import datetime, timedelta
from decimal import Decimal

from fastapi import FastAPI, Request, HTTPException, Query, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openai import OpenAI

APP = FastAPI()

# ================
# ENV & CORS
# ================
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS","").split(",") if o.strip()]
APP.add_middleware(
    CORSMiddleware,
    allow_origins=origins or [],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_APP       = os.getenv("API_KEY_APP", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN","")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY","")
DATABASE_URL      = (os.getenv("DATABASE_URL") or "").strip()
if DATABASE_URL and "sslmode=" not in DATABASE_URL:
    sep = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = DATABASE_URL + f"{sep}sslmode=require"

client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ================
# Bootstrap schema
# ================
def _bootstrap_schema():
    with engine.begin() as conn:
        # messages
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS messages (
          id SERIAL PRIMARY KEY,
          content   TEXT NOT NULL,
          reply     TEXT,
          created_at TIMESTAMP DEFAULT NOW()
        );
        """))

        # movements base
        conn.execute(text("CREATE TABLE IF NOT EXISTS movements (id SERIAL PRIMARY KEY);"))

        # fix colonna legacy `importo`
        cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name='movements'
        """)).scalars().all()

        if "importo" in cols:
            if "amount" not in cols:
                conn.execute(text("ALTER TABLE movements RENAME COLUMN importo TO amount;"))
            else:
                conn.execute(text("ALTER TABLE movements DROP COLUMN importo;"))

        # colonne standard
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS type VARCHAR(10);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS amount NUMERIC(14,2);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS currency VARCHAR(8);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS category VARCHAR(50);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS note TEXT;"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS voce VARCHAR(50) DEFAULT 'generale';"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();"))

        # default & not null
        conn.execute(text("UPDATE movements SET currency='CHF' WHERE currency IS NULL;"))
        conn.execute(text("UPDATE movements SET created_at=NOW() WHERE created_at IS NULL;"))
        conn.execute(text("UPDATE movements SET voce='generale' WHERE voce IS NULL;"))
        conn.execute(text("UPDATE movements SET type='in' WHERE type IS NULL;"))
        conn.execute(text("UPDATE movements SET amount=0 WHERE amount IS NULL;"))

        conn.execute(text("ALTER TABLE movements ALTER COLUMN type SET NOT NULL;"))
        conn.execute(text("ALTER TABLE movements ALTER COLUMN amount SET NOT NULL;"))
        conn.execute(text("ALTER TABLE movements ALTER COLUMN currency SET NOT NULL;"))
        conn.execute(text("ALTER TABLE movements ALTER COLUMN voce SET NOT NULL;"))

try:
    _bootstrap_schema()
except Exception as e:
    print("SCHEMA_BOOTSTRAP_ERROR:", e, flush=True)

# ================
# Middleware
# ================
WINDOW_SECONDS = 60
MAX_REQ = 60
_bucket = {}
def _client_key(req: Request) -> str:
    xf = req.headers.get("x-forwarded-for")
    ip = xf.split(",")[0].strip() if xf else (req.client.host if req.client else "0.0.0.0")
    return ip + "|" + (req.headers.get("x-api-key") or "-")

@APP.middleware("http")
async def security_and_rate_limit(request: Request, call_next):
    path = request.url.path
    open_paths = {"/","/healthz","/webhook","/debug"}
    if path not in open_paths:
        if request.headers.get("x-api-key") != API_KEY_APP:
            raise HTTPException(status_code=401, detail="invalid api key")
    return await call_next(request)

# ================
# Routes base
# ================
@APP.get("/") def root(): return {"status":"ok"}
@APP.get("/healthz") def healthz(): return "ok"
@APP.get("/debug")
def debug():
    try:
        with engine.begin() as conn: conn.execute(text("SELECT 1"))
        db_ok=True
    except Exception as e:
        db_ok=False
        return {"db_ok":db_ok,"err":str(e)}
    return {"db_ok":db_ok}

# ================
# Movements
# ================
@APP.post("/movements")
def create_movement(item: dict = Body(...)):
    try:
        t = (item.get("type") or "").lower()
        amt = Decimal(str(item.get("amount",0)))
        cur = (item.get("currency") or "CHF").upper()
        cat = item.get("category")
        note= item.get("note")
        voce= item.get("voce") or "generale"
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO movements(type,amount,currency,category,note,voce,created_at)
                VALUES (:t,:a,:c,:cat,:n,:v,:ts)
            """), {"t":t,"a":amt,"c":cur,"cat":cat,"n":note,"v":voce,"ts":datetime.utcnow()})
        return {"status":"ok"}
    except Exception as e:
        return {"error":"db_failed_insert","detail":str(e)}

@APP.get("/movements")
def list_movements():
    with engine.begin() as conn:
        rows=conn.execute(text("SELECT * FROM movements ORDER BY created_at DESC LIMIT 50")).mappings().all()
    return {"items":[dict(r) for r in rows]}

