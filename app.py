import os
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import psycopg2
import psycopg2.extras

from fastapi import FastAPI, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------- CONFIG --------------------

APP = FastAPI(title="FLAI APP")

DATABASE_URL = os.getenv("DATABASE_URL")  # contiene già sslmode=require
API_KEY_APP = os.getenv("API_KEY_APP", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")

# CORS stretti (nessun wildcard)
APP.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or [],
    allow_methods=["*"],
    allow_headers=["*"],
)

# rate limit soft per API-key/IP
_WINDOW = 60
_LIMIT = 90
_bucket: Dict[str, Any] = {}


def _client_key(req: Request) -> str:
    xf = req.headers.get("x-forwarded-for")
    ip = (xf.split(",")[0].strip() if xf else req.client.host) if req.client else "unknown"
    return f"{ip}|{req.headers.get('x-api-key','-')}"


@APP.middleware("http")
async def guard(request: Request, call_next):
    open_paths = {"/", "/healthz", "/webhook"}  # webhook GET deve restare aperto per Meta
    if request.url.path not in open_paths:
        if not API_KEY_APP or request.headers.get("x-api-key") != API_KEY_APP:
            raise HTTPException(status_code=401, detail="unauthorized")
        # rate soft
        now = int(time.time())
        ck = _client_key(request)
        wstart, cnt = _bucket.get(ck, (now, 0))
        if now - wstart >= _WINDOW:
            wstart, cnt = now, 0
        cnt += 1
        _bucket[ck] = (wstart, cnt)
        if cnt > _LIMIT:
            return JSONResponse({"error": "rate_limited", "limit_per_min": _LIMIT}, status_code=429)
    return await call_next(request)


# -------------------- DB HELPERS --------------------

def db_conn():
    """
    Connessione semplice a Postgres. Nessun 'with cursor' (evita l'errore
    'cursor object does not support the context manager').
    """
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing")
    return psycopg2.connect(DATABASE_URL)  # sslmode è nel DSN


def ensure_tables():
    """Crea tabelle se non esistono (safe). Usa un sottoinsieme compatibile con lo schema attuale."""
    conn = db_conn()
    try:
        cur = conn.cursor()
        # messages
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                reply   TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        # movements (solo colonne che usiamo)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS movements (
                id SERIAL PRIMARY KEY,
                type VARCHAR(10) NOT NULL,
                amount NUMERIC(14,2) NOT NULL,
                currency VARCHAR(8) NOT NULL DEFAULT 'CHF',
                category VARCHAR(50),
                note TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


@APP.on_event("startup")
def _startup():
    ensure_tables()


# -------------------- ROUTES CORE --------------------

@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    try:
        conn = db_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        _ = cur.fetchone()
        cur.close()
        conn.close()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": "db_bootstrap_failed", "detail": str(e)}, status_code=500)


@APP.post("/messages")
def post_message(payload: Dict[str, Any] = Body(...)):
    msg = (payload or {}).get("message", "")
    if not msg:
        raise HTTPException(400, "missing 'message'")
    reply = f"Hai scritto: {msg}"
    conn = db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages(content, reply, created_at) VALUES (%s,%s,NOW()) RETURNING id",
            (msg, reply),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        return {"id": new_id, "reply": reply}
    except Exception as e:
        conn.rollback()
        return JSONResponse({"error": "db_failed_insert_message", "detail": str(e)}, status_code=500)
    finally:
        conn.close()


@APP.post("/movements")
def create_movement(item: Dict[str, Any] = Body(...)):
    t = item.get("type")
    if t not in ("in", "out"):
        raise HTTPException(422, "type must be 'in' or 'out'")
    a = item.get("amount")
    if a is None:
        raise HTTPException(422, "amount required")
    cur_ = item.get("currency", "CHF")
    cat = item.get("category")
    note = item.get("note")

    conn = db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO movements(type, amount, currency, category, note, created_at)
               VALUES (%s,%s,%s,%s,%s,NOW()) RETURNING id""",
            (t, a, cur_, cat, note),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        return {"ok": True, "id": new_id}
    except Exception as e:
        conn.rollback()
        return JSONResponse({"error": "db_failed_insert", "detail": str(e)}, status_code=500)
    finally:
        conn.close()


@APP.get("/movements")
def list_movements(
    _from: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    limit: int = 50,
):
    if limit < 1 or limit > 500:
        limit = 50
    where = []
    params: List[Any] = []
    if _from:
        where.append("created_at >= %s")
        params.append(_from)
    if to:
        where.append("created_at < %s")
        params.append(to)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""SELECT id, type, amount, currency, category, note, created_at
              FROM movements {where_sql}
              ORDER BY created_at DESC
              LIMIT %s"""
    params.append(limit)
    conn = db_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows
    except Exception as e:
        return JSONResponse({"error": "db_failed_query", "detail": str(e)}, status_code=500)
    finally:
        conn.close()


# -------------------- ANALYTICS --------------------

@APP.get("/analytics/overview")
def analytics_overview(days: int = 7):
    """Ritorna per giorno: in, out, net nell'intervallo [today-days+1, today]"""
    if days < 1 or days > 90:
        days = 7
    end = datetime.utcnow().date() + timedelta(days=1)  # esclusivo
    start = end - timedelta(days=days)

    conn = db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DATE(created_at) AS d,
                   COALESCE(SUM(CASE WHEN type='in'  THEN amount END), 0) AS in_amt,
                   COALESCE(SUM(CASE WHEN type='out' THEN amount END), 0) AS out_amt
            FROM movements
            WHERE created_at >= %s AND created_at < %s
            GROUP BY 1
            ORDER BY 1
            """,
            (start, end),
        )
        agg = {row[0].isoformat(): {"in": float(row[1]), "out": float(row[2])} for row in cur.fetchall()}
        # riempi giorni mancanti
        out = []
        d = start
        while d < end:
            k = d.isoformat()
            vals = agg.get(k, {"in": 0.0, "out": 0.0})
            out.append({"date": k, "in": vals["in"], "out": vals["out"], "net": vals["in"] - vals["out"]})
            d += timedelta(days=1)
        cur.close()
        return {"window_days": days, "series": out}
    except Exception as e:
        return JSONResponse({"error": "db_failed_analytics", "detail": str(e)}, status_code=500)
    finally:
        conn.close()


@APP.post("/summaries/generate")
def summaries_generate(days: int = 30):
    """Calcola KPI di base e genera un riassunto in linguaggio naturale (senza usare OpenAI)."""
    if days < 1 or days > 180:
        days = 30
    end = datetime.utcnow().date() + timedelta(days=1)
    start = end - timedelta(days=days)

    conn = db_conn()
    try:
        cur = conn.cursor()
        # movimenti totali
        cur.execute(
            """
            SELECT
              COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
              COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
            FROM movements
            WHERE created_at >= %s AND created_at < %s
            """,
            (start, end),
        )
        row = cur.fetchone()
        total_in = float(row[0] or 0.0)
        total_out = float(row[1] or 0.0)
        net = total_in - total_out

        # volume transazioni
        cur.execute(
            """
            SELECT COUNT(*) FILTER (WHERE type='in'),
                   COUNT(*) FILTER (WHERE type='out')
            FROM movements
            WHERE created_at >= %s AND created_at < %s
            """,
            (start, end),
        )
        c_in, c_out = cur.fetchone()

        cur.close()

        summary = (
            f"Ultimi {days} giorni — Incassi: {total_in:.2f} CHF, Uscite: {total_out:.2f} CHF, "
            f"Netto: {net:.2f} CHF. Operazioni: {c_in} entrate, {c_out} uscite."
        )
        return {
            "window_days": days,
            "totals": {"in": total_in, "out": total_out, "net": net},
            "counts": {"in": c_in, "out": c_out},
            "summary_text": summary,
        }
    except Exception as e:
        return JSONResponse({"error": "db_failed_summary", "detail": str(e)}, status_code=500)
    finally:
        conn.close()


# -------------------- WEBHOOK (Meta verify + demo) --------------------

@APP.get("/webhook")
def whatsapp_verify(hub_mode: str = "", hub_challenge: str = "", hub_verify_token: str = ""):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        # nessuna auth qui (deve essere pubblico per Meta)
        return PlainTextResponse(hub_challenge or "", media_type="text/plain")
    raise HTTPException(status_code=403, detail="verification_failed")


@APP.post("/webhook")
def webhook_demo(payload: Dict[str, Any] = Body(...)):
    """
    Per ora: se arriva {"message":"..."} risponde come /messages (eco + salva).
    Quando WA sarà sbloccato potremo parsare il formato Meta.
    """
    msg = (payload or {}).get("message")
    if not msg:
        return {"ok": True}  # altri eventi, ignora
    # riusa la logica di /messages
    return post_message({"message": msg})
