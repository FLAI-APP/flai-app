import os
import time
import json
from decimal import Decimal
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import psycopg
from psycopg.rows import dict_row

from openai import OpenAI

# ------------------ Config ------------------

API_KEY_APP = os.getenv("API_KEY_APP", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# OpenAI client (usa OPENAI_API_KEY da env)
OA_CLIENT = OpenAI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Helpers ------------------

def _need_api_key(path: str) -> bool:
    # Endpoints "pubblici"
    if path in ("/", "/healthz", "/webhook"):  # GET verify di WhatsApp deve restare pubblico
        return False
    return True

@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if _need_api_key(request.url.path):
        if not API_KEY_APP or request.headers.get("x-api-key") != API_KEY_APP:
            raise HTTPException(status_code=401, detail="unauthorized")
    return await call_next(request)

def get_conn():
    """
    Connessione psycopg v3.
    Usa DATABASE_URL come viene da Render (va bene anche con sslmode=require).
    row_factory=dict_row per avere dict dalle SELECT.
    """
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def ensure_schema():
    """
    Crea tabelle se non esistono.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages(
                  id SERIAL PRIMARY KEY,
                  content TEXT NOT NULL,
                  reply   TEXT,
                  created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS movements(
                  id SERIAL PRIMARY KEY,
                  type VARCHAR(10) NOT NULL,      -- "in" | "out"
                  amount NUMERIC(14,2) NOT NULL,
                  currency VARCHAR(8) NOT NULL DEFAULT 'CHF',
                  category VARCHAR(50),
                  note TEXT,
                  created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)
            conn.commit()

@app.on_event("startup")
def _startup():
    ensure_schema()

# ------------------ Routes ------------------

@app.get("/")
def root():
    return {"ok": True}

@app.get("/healthz")
def healthz():
    # verifica rapida DB
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok;")
            one = cur.fetchone()["ok"]
        return {"ok": bool(one)}
    except Exception as e:
        return JSONResponse({"error": "db_bootstrap_failed", "detail": str(e)}, status_code=500)

# ----- WhatsApp verify (GET) + events (POST) -----

@app.get("/webhook")
def whatsapp_verify(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "", media_type="text/plain")
    raise HTTPException(status_code=403, detail="forbidden")

@app.post("/webhook")
async def whatsapp_events(payload: dict = Body(...)):
    """
    Per ora: supporta la demo {"message": "..."} che usiamo da curl.
    Quando WhatsApp sarà sbloccato, qui parseremo entry/changes.
    """
    if "message" in payload:
        text = str(payload["message"]).strip()
        reply = _ai_reply(text)
        # salva su DB
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages(content, reply) VALUES (%s, %s) RETURNING id;",
                    (text, reply),
                )
                new_id = cur.fetchone()["id"]
                conn.commit()
            return {"id": new_id, "reply": reply}
        except Exception as e:
            return JSONResponse({"error": "db_failed_insert_message", "detail": str(e)}, status_code=500)

    return {"ok": True}

# ----- Chat semplice (/messages) -----

@app.post("/messages")
def post_message(body: dict = Body(...)):
    """
    Body: { "message": "..." }
    Risponde con AI e salva su DB.
    """
    text = str(body.get("message", "")).strip()
    if not text:
        raise HTTPException(400, "message is required")

    reply = _ai_reply(text)

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages(content, reply) VALUES (%s, %s) RETURNING id;",
                (text, reply),
            )
            new_id = cur.fetchone()["id"]
            conn.commit()
        return {"id": new_id, "reply": reply}
    except Exception as e:
        return JSONResponse({"error": "db_failed_insert_message", "detail": str(e)}, status_code=500)

# ----- Movements (/movements) -----

@app.post("/movements")
def create_movement(item: dict = Body(...)):
    """
    Body minimo:
    { "type": "in"|"out", "amount": 2500, "currency": "CHF", "category": "sales", "note": "..." }
    """
    t = str(item.get("type", "")).strip().lower()
    if t not in ("in", "out"):
        raise HTTPException(422, "type must be 'in' or 'out'")

    try:
        amt = Decimal(str(item.get("amount")))
    except Exception:
        raise HTTPException(422, "amount must be a number")

    cur = (item.get("currency") or "CHF").upper()
    cat = item.get("category")
    note = item.get("note")

    try:
        with get_conn() as conn, conn.cursor() as c:
            c.execute(
                """
                INSERT INTO movements(type, amount, currency, category, note)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (t, amt, cur, cat, note),
            )
            new_id = c.fetchone()["id"]
            conn.commit()
        return {"ok": True, "id": new_id}
    except Exception as e:
        return JSONResponse({"error": "db_failed_insert", "detail": str(e)}, status_code=500)

@app.get("/movements")
def list_movements(limit: int = 100, _from: str | None = Query(None, alias="from"), to: str | None = None):
    """
    Lista movimenti (max 100 di default) con filtri opzionali di data (YYYY-MM-DD).
    """
    limit = max(1, min(limit, 500))
    q = "SELECT id, type, amount, currency, category, note, created_at FROM movements WHERE 1=1"
    params: list = []
    if _from:
        q += " AND created_at >= %s"
        params.append(_from)
    if to:
        q += " AND created_at < %s"
        params.append(to)
    q += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    try:
        with get_conn() as conn, conn.cursor() as c:
            c.execute(q, params)
            rows = c.fetchall()
        return rows
    except Exception as e:
        return JSONResponse({"error": "db_failed_query", "detail": str(e)}, status_code=500)

# ------------------ AI ------------------

def _ai_reply(user_text: str) -> str:
    """
    Risposta breve e sicura. Se OpenAI fallisce, fallback statico.
    """
    try:
        # modello leggero ed economico
        resp = OA_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Rispondi in modo breve, chiaro e utile."},
                {"role": "user", "content": user_text},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        return (resp.choices[0].message.content or "Ok.").strip()
    except Exception:
        # fallback sicuro
        return f"Hai scritto: {user_text}"
