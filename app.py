import os
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

import psycopg
from psycopg.rows import dict_row

APP = FastAPI()

# API Key per i tuoi endpoint
API_KEY_APP = os.getenv("API_KEY_APP", "flai_Chiasso13241")

# Normalizza DATABASE_URL
def normalize_db_url(url: str) -> str:
    if not url:
        return url
    clean = url.strip().strip("'").strip('"').replace("\n", "").replace("\r", "").strip()
    if not clean:
        return clean
    if "sslmode=" in clean:
        return clean
    p = urlparse(clean)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q["sslmode"] = "require"
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), p.fragment))

DATABASE_URL = normalize_db_url(os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or "")

def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    return psycopg.connect(DATABASE_URL)

def bootstrap_tables():
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    content   TEXT NOT NULL,
                    reply     TEXT,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS movements (
                    id SERIAL PRIMARY KEY,
                    type     VARCHAR(10) NOT NULL,
                    amount   NUMERIC(14,2) NOT NULL,
                    currency VARCHAR(8) NOT NULL DEFAULT 'CHF',
                    category VARCHAR(50),
                    note     TEXT,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
                );
            """)

@APP.middleware("http")
async def api_key_guard(request: Request, call_next):
    if request.url.path not in {"/", "/healthz"}:
        if request.headers.get("x-api-key") != API_KEY_APP:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)

@APP.get("/")
def root():
    return {"ok": True, "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    try:
        if not DATABASE_URL:
            return JSONResponse({"error": "db_url_missing"}, status_code=500)
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": "db_bootstrap_failed", "detail": str(e)}, status_code=500)

@APP.post("/messages")
def create_message(payload: dict = Body(...)):
    text = (payload or {}).get("message", "").strip()
    if not text:
        raise HTTPException(422, "message is required")
    reply = f"Hai scritto: {text}"
    try:
        bootstrap_tables()
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages(content, reply, created_at) VALUES (%s,%s,%s) RETURNING id;",
                    (text, reply, datetime.now(timezone.utc).replace(tzinfo=None)),
                )
                new_id = cur.fetchone()[0]
        return {"id": new_id, "reply": reply}
    except Exception as e:
        return JSONResponse({"error": "db_failed_insert_message", "detail": str(e)}, status_code=500)

@APP.post("/movements")
def add_movement(payload: dict = Body(...)):
    t = (payload or {}).get("type")
    if t not in ("in", "out"):
        raise HTTPException(422, "type must be 'in' or 'out'")
    try:
        amt = Decimal(str((payload or {}).get("amount")))
    except Exception:
        raise HTTPException(422, "amount must be a number")
    cur_ = (payload or {}).get("currency", "CHF") or "CHF"
    cat  = (payload or {}).get("category")
    note = (payload or {}).get("note")
    try:
        bootstrap_tables()
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO movements(type, amount, currency, category, note, created_at)
                    VALUES (%s,%s,%s,%s,%s,NOW()) RETURNING id;
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
                cur.execute("""
                    SELECT id, type, amount, currency, category, note, created_at
                    FROM movements
                    ORDER BY created_at DESC LIMIT 200;
                """)
                rows = cur.fetchall()
        return {"items": rows}
    except Exception as e:
        return JSONResponse({"error": "db_failed_query", "detail": str(e)}, status_code=500)

META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "flai-verify-123")

@APP.get("/webhook")
def whatsapp_verify(hub_mode: str = "", hub_challenge: str = "", hub_verify_token: str = ""):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "", media_type="text/plain")
    raise HTTPException(status_code=403, detail="verification_failed")
