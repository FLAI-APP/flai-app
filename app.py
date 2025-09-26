# app.py — base stabile + analytics giornalieri + bulk insert
import os
from datetime import datetime, timedelta
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

# -------- Movements (CRUD essenziale) --------
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

# -------- Movements: bulk insert --------
@APP.post("/movements/bulk")
def bulk_movements(items: list[dict] = Body(...)):
    """
    Esempio body:
    [
      {"type":"in","amount":1200,"currency":"CHF","category":"sales","note":"giorno 1"},
      {"type":"out","amount":300,"currency":"CHF","category":"fornitori","note":"pane"}
    ]
    """
    try:
        if not isinstance(items, list) or not items:
            raise HTTPException(422, "body must be a non-empty JSON array")
        now = datetime.utcnow()
        to_insert = []
        for it in items:
            t = (it.get("type") or "").strip().lower()
            if t not in ("in","out"):
                raise HTTPException(422, "each item.type must be 'in' or 'out'")
            amt = Decimal(str(it.get("amount",0)))
            cur = (it.get("currency") or "CHF").upper()
            cat = it.get("category")
            note = it.get("note")
            voce = it.get("voce") or "generale"
            to_insert.append({"t":t,"a":amt,"c":cur,"cat":cat,"n":note,"v":voce,"ts":now})

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO movements(type, amount, currency, category, note, voce, created_at)
                SELECT :t, :a, :c, :cat, :n, :v, :ts
            """), to_insert)
        return {"status":"ok","inserted":len(to_insert)}
    except HTTPException:
        raise
    except Exception as e:
        return {"error":"db_failed_bulk","detail":str(e)}

# -------- Analytics: overview per giorno --------
@APP.get("/analytics/overview")
def analytics_overview(days: int = 30):
    """
    Restituisce per gli ultimi N giorni:
    - per_day: elenco con {date, in, out, net}
    - totals: somma complessiva in/out/net del periodo
    """
    if days <= 0 or days > 365:
        raise HTTPException(422, "days must be between 1 and 365")

    try:
        with engine.begin() as conn:
            # serie date
            per_day = conn.execute(text("""
                WITH days AS (
                  SELECT generate_series(
                    (CURRENT_DATE - (:d::int - 1) * INTERVAL '1 day')::date,
                    CURRENT_DATE::date,
                    INTERVAL '1 day'
                  )::date AS d
                ),
                agg AS (
                  SELECT
                    DATE(created_at) AS d,
                    COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS in_amt,
                    COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS out_amt
                  FROM movements
                  WHERE created_at >= CURRENT_DATE - (:d::int - 1) * INTERVAL '1 day'
                  GROUP BY DATE(created_at)
                )
                SELECT
                  days.d AS date,
                  COALESCE(agg.in_amt,0)  AS in,
                  COALESCE(agg.out_amt,0) AS out,
                  COALESCE(agg.in_amt,0) - COALESCE(agg.out_amt,0) AS net
                FROM days
                LEFT JOIN agg ON agg.d = days.d
                ORDER BY days.d ASC
            """), {"d": days}).mappings().all()

            totals = conn.execute(text("""
                SELECT
                  COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
                  COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                FROM movements
                WHERE created_at >= CURRENT_DATE - (:d::int - 1) * INTERVAL '1 day'
            """), {"d": days}).mappings().first()

        tin  = float(totals["total_in"]  or 0)
        tout = float(totals["total_out"] or 0)
        return {
            "days": days,
            "per_day": [dict(r) for r in per_day],
            "totals": {"in": tin, "out": tout, "net": tin - tout}
        }
    except Exception as e:
        return {"error":"db_failed_analytics","detail":str(e)}

