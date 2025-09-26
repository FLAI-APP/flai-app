# app.py — versione minimale stabile (DB fix movements + API)
import os
from datetime import datetime
from decimal import Decimal

from fastapi import FastAPI, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text

APP = FastAPI()

# -------- CORS --------
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
APP.add_middleware(
    CORSMiddleware,
    allow_origins=origins or [],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- ENV & DB --------
API_KEY_APP  = os.getenv("API_KEY_APP", "")
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
if DATABASE_URL and "sslmode=" not in DATABASE_URL:
    sep = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = DATABASE_URL + f"{sep}sslmode=require"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# -------- Bootstrap schema (idempotente) --------
def _bootstrap_schema():
    with engine.begin() as conn:
        # crea tabella base se manca
        conn.execute(text("CREATE TABLE IF NOT EXISTS movements (id SERIAL PRIMARY KEY);"))

        # colonne presenti
        cols = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='movements'
        """)).scalars().all()

        # rinomina 'importo' -> 'amount' oppure drop se amount esiste già
        if "importo" in cols:
            if "amount" not in cols:
                conn.execute(text("ALTER TABLE movements RENAME COLUMN importo TO amount;"))
                cols = [("amount" if c == "importo" else c) for c in cols]
            else:
                conn.execute(text("ALTER TABLE movements DROP COLUMN importo;"))
                cols = [c for c in cols if c != "importo"]

        # aggiungi colonne standard se mancano
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS type VARCHAR(10);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS amount NUMERIC(14,2);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS currency VARCHAR(8);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS category VARCHAR(50);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS note TEXT;"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS voce VARCHAR(50);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;"))

        # default + valori di sicurezza
        conn.execute(text("UPDATE movements SET currency='CHF' WHERE currency IS NULL;"))
        conn.execute(text("UPDATE movements SET created_at=NOW() WHERE created_at IS NULL;"))
        conn.execute(text("UPDATE movements SET voce='generale' WHERE voce IS NULL;"))
        conn.execute(text("UPDATE movements SET type='in' WHERE type IS NULL;"))
        conn.execute(text("UPDATE movements SET amount=0 WHERE amount IS NULL;"))

        # vincoli coerenti
        conn.execute(text("ALTER TABLE movements ALTER COLUMN currency SET NOT NULL;"))
        conn.execute(text("ALTER TABLE movements ALTER COLUMN type SET NOT NULL;"))
        conn.execute(text("ALTER TABLE movements ALTER COLUMN amount SET NOT NULL;"))
        conn.execute(text("ALTER TABLE movements ALTER COLUMN voce SET NOT NULL;"))

try:
    _bootstrap_schema()
except Exception as e:
    print("SCHEMA_BOOTSTRAP_ERROR:", e, flush=True)

# -------- Middleware API key (solo per percorsi protetti) --------
@APP.middleware("http")
async def api_key_guard(request: Request, call_next):
    open_paths = {"/", "/healthz", "/debug"}  # liberi
    if request.url.path not in open_paths:
        if request.headers.get("x-api-key") != API_KEY_APP:
            raise HTTPException(status_code=401, detail="invalid api key")
    return await call_next(request)

# -------- Routes base --------
@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

@APP.get("/debug")
def debug():
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        return {"db_ok": True}
    except Exception as e:
        return {"db_ok": False, "err": str(e)}

# -------- Movements --------
@APP.post("/movements")
def create_movement(item: dict = Body(...)):
    try:
        t = (item.get("type") or "").strip().lower()
        if t not in ("in", "out"):
            raise HTTPException(422, "type must be 'in' or 'out'")
        amt = Decimal(str(item.get("amount", 0)))
        cur = (item.get("currency") or "CHF").upper()
        cat = item.get("category")
        note = item.get("note")
        voce = item.get("voce") or "generale"

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO movements(type, amount, currency, category, note, voce, created_at)
                VALUES (:t, :a, :c, :cat, :n, :v, :ts)
            """), {"t": t, "a": amt, "c": cur, "cat": cat, "n": note, "v": voce, "ts": datetime.utcnow()})
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": "db_failed_insert", "detail": str(e)}

@APP.get("/movements")
def list_movements(_from: str = Query(None, alias="from"), to: str = None, limit: int = 200):
    try:
        q = "SELECT id, type, amount, currency, category, note, voce, created_at FROM movements WHERE 1=1"
        params = {}
        if _from:
            q += " AND created_at >= :f"; params["f"] = _from
        if to:
            q += " AND created_at < :t"; params["t"] = to
        q += " ORDER BY created_at DESC LIMIT :lim"
        params["lim"] = limit

        with engine.begin() as conn:
            rows = conn.execute(text(q), params).mappings().all()
        return {"items": [dict(r) for r in rows]}
    except Exception as e:
        return {"error": "db_failed_query", "detail": str(e)}

# -------- Summaries --------
@APP.get("/summaries/overview")
def summaries_overview():
    try:
        with engine.begin() as conn:
            row = conn.execute(text("""
                SELECT
                  COALESCE(SUM(CASE WHEN type='in' THEN amount END),0) AS total_in,
                  COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                FROM movements
            """)).mappings().first()
        total_in = float(row["total_in"])
        total_out = float(row["total_out"])
        saldo = total_in - total_out
        return {"in": total_in, "out": total_out, "saldo": saldo}
    except Exception as e:
        return {"error": "db_failed_summary", "detail": str(e)}

