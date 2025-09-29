import os
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

import psycopg  # psycopg v3
from psycopg.rows import dict_row

APP = FastAPI()

# ==== ENV ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_KEY_APP    = os.getenv("API_KEY_APP", "flai_Chiasso13241")
DATABASE_URL   = os.getenv("DATABASE_URL", "") or os.getenv("POSTGRES_URL", "")

# ==== SECURITY (API KEY SU TUTTO tranne "/" e "/healthz") ====
@APP.middleware("http")
async def api_key_guard(request: Request, call_next):
    if request.url.path not in {"/", "/healthz"}:
        if request.headers.get("x-api-key") != API_KEY_APP:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)

# ==== ROOT/HEALTH ====
@APP.get("/")
def root():
    return {"ok": True, "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    # opzionale: probe DB veloce
    try:
        if DATABASE_URL:
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": "db_bootstrap_failed", "detail": str(e)}, status_code=500)

# ==== DB HELPER (psycopg v3) ====
def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    # open/close ad ogni richiesta: semplice e sicuro
    return psycopg.connect(DATABASE_URL)

# ==== MESSAGES (demo semplice AI OFF finché non serve) ====
@APP.post("/messages")
def create_message(payload: dict = Body(...)):
    """
    Body: { "message": "..." }
    Salva la domanda + una risposta finta e ritorna la reply.
    """
    msg = (payload or {}).get("message", "").strip()
    if not msg:
        raise HTTPException(422, "message is required")

    # risposta finta per test infrastruttura
    reply = f"Hai scritto: {msg}"

    # salva su DB se esiste tabella messages (id serial, content text, reply text, created_at timestamp)
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        reply   TEXT,
                        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    "INSERT INTO messages(content, reply, created_at) VALUES (%s,%s,%s) RETURNING id;",
                    (msg, reply, datetime.now(timezone.utc).replace(tzinfo=None)),
                )
                new_id = cur.fetchone()[0]
        return {"id": new_id, "reply": reply}
    except Exception as e:
        return JSONResponse({"error": "db_failed_insert_message", "detail": str(e)}, status_code=500)

# ==== MOVEMENTS ====
@APP.post("/movements")
def add_movement(payload: dict = Body(...)):
    """
    Body:
      {
        "type": "in" | "out",
        "amount": 123.45,
        "currency": "CHF",
        "category": "sales",
        "note": "opzionale"
      }
    """
    t = (payload or {}).get("type")
    if t not in ("in", "out"):
        raise HTTPException(422, "type must be 'in' or 'out'")

    try:
        amt = Decimal(str((payload or {}).get("amount")))
    except Exception:
        raise HTTPException(422, "amount must be a number")

    cur_ = (payload or {}).get("currency", "CHF") or "CHF"
    cat = (payload or {}).get("category")
    note = (payload or {}).get("note")

    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                # bootstrap tabella se manca (schema semplice e coerente)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS movements (
                        id SERIAL PRIMARY KEY,
                        type VARCHAR(10) NOT NULL,
                        amount NUMERIC(14,2) NOT NULL,
                        currency VARCHAR(8) NOT NULL DEFAULT 'CHF',
                        category VARCHAR(50),
                        note TEXT,
                        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO movements(type, amount, currency, category, note, created_at)
                    VALUES (%s,%s,%s,%s,%s,NOW())
                    RETURNING id;
                    """,
                    (t, amt, cur_, cat, note),
                )
                new_id = cur.fetchone()[0]
        return {"ok": True, "id": new_id}
    except Exception as e:
        return JSONResponse({"error": "db_failed_insert", "detail": str(e)}, status_code=500)

@APP.get("/movements")
def list_movements():
    try:
        with db_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, type, amount, currency, category, note, created_at
                    FROM movements
                    ORDER BY created_at DESC
                    LIMIT 200;
                    """
                )
                rows = cur.fetchall()
        return {"items": rows}
    except Exception as e:
        return JSONResponse({"error": "db_failed_query", "detail": str(e)}, status_code=500)

# ==== WEBHOOK META (verifica) ====
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "flai-verify-123")

@APP.get("/webhook")
def whatsapp_verify(
    hub_mode: str = "",
    hub_challenge: str = "",
    hub_verify_token: str = "",
):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "", media_type="text/plain")
    raise HTTPException(status_code=403, detail="verification_failed")

