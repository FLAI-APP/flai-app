import os
import json
import decimal
from datetime import datetime

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, PlainTextResponse

# OpenAI SDK (>=1.40) – niente proxies custom
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as _e:
    _openai_client = None

APP = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL")

# -----------------------------
# Helpers DB
# -----------------------------

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)

def bootstrap_db():
    """
    Crea tabelle se non esistono (schema compatibile con il tuo DB attuale).
    """
    conn = get_conn()
    cur = conn.cursor()

    # messages
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            reply   TEXT,
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
        );
    """)

    # movements (schema allineato alla foto: id, message_id, type, voce, valuta, note, created_at, amount, currency, category)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS movements (
            id SERIAL PRIMARY KEY,
            message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
            type VARCHAR(10) NOT NULL,
            voce TEXT NOT NULL DEFAULT 'generale',
            valuta VARCHAR(8) NOT NULL DEFAULT 'CHF',
            note  TEXT,
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
            amount NUMERIC(14,2) NOT NULL,
            currency VARCHAR(8) NOT NULL DEFAULT 'CHF',
            category VARCHAR(50)
        );
    """)

    conn.commit()
    cur.close()
    conn.close()

# Esegui bootstrap all'avvio
try:
    bootstrap_db()
except Exception as e:
    # non bloccare l'avvio: lo segnaliamo su /healthz
    print("DB bootstrap error:", e, flush=True)

# -----------------------------
# Utils
# -----------------------------

def to_decimal(x):
    if isinstance(x, (int, float, decimal.Decimal)):
        return decimal.Decimal(str(x))
    # stringhe tipo "123.45"
    return decimal.Decimal(x)

def ai_answer(prompt: str) -> str:
    """
    Chiama OpenAI; se non configurato, torna una risposta di fallback.
    """
    if _openai_client is None:
        return "AI non configurata (manca OPENAI_API_KEY); risposta di fallback."

    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sei un assistente aziendale pratico e conciso."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=180
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI fallback] Errore modello: {e}"

# -----------------------------
# Endpoints base
# -----------------------------

@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    """
    Verifica connessione DB (senza usare 'with cursor').
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return {"status": "ok", "message": "FLAI API attiva 🚀"}
    except Exception as e:
        return {"error": "db_bootstrap_failed", "detail": str(e)}

# -----------------------------
# Chat AI + persistenza
# -----------------------------

@APP.post("/messages")
def create_message(payload: dict = Body(...)):
    """
    Body: { "message": "testo" }
    Salva domanda/risposta su 'messages'.
    """
    text = (payload or {}).get("message", "").strip()
    if not text:
        return JSONResponse({"error": "missing_message"}, status_code=400)

    reply = ai_answer(text)

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (content, reply, created_at) VALUES (%s,%s,%s) RETURNING id",
            (text, reply, datetime.utcnow())
        )
        mid = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return JSONResponse({"error": "db_insert_failed", "detail": str(e)}, status_code=500)

    return {"id": mid, "message": text, "reply": reply}

# -----------------------------
# Movements (entrate/uscite)
# -----------------------------

@APP.post("/movements")
def add_movement(payload: dict = Body(...)):
    """
    Body minimi:
      {
        "type": "in" | "out",
        "amount": 123.45,
        "currency": "CHF",       # opzionale (default DB = CHF)
        "category": "sales",     # opzionale
        "note": "incasso giorno" # opzionale
      }
    Colonne DB esistenti: (id, message_id, type, voce, valuta, note, created_at, amount, currency, category)
    Settiamo automaticamente: voce='generale', valuta='CHF' (lasciamo i default).
    """
    t = (payload or {}).get("type", "").strip().lower()
    if t not in ("in", "out"):
        return JSONResponse({"error": "invalid_type", "detail": "type deve essere 'in' o 'out'."}, status_code=400)

    try:
        amt = to_decimal((payload or {}).get("amount"))
    except Exception:
        return JSONResponse({"error": "invalid_amount"}, status_code=400)

    cur_code = (payload or {}).get("currency") or None
    cat = (payload or {}).get("category") or None
    note = (payload or {}).get("note") or None

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO movements (type, amount, currency, category, note)
            VALUES (%s, %s, COALESCE(%s,'CHF'), %s, %s)
            RETURNING id, created_at
            """,
            (t, amt, cur_code, cat, note)
        )
        rid = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "ok", "id": rid[0], "created_at": rid[1].isoformat()}
    except Exception as e:
        return JSONResponse({"error": "db_failed_insert", "detail": str(e)}, status_code=500)

@APP.get("/movements")
def list_movements(limit: int = 200):
    """
    Lista ultimi movimenti + totali in/out/netto.
    """
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            """
            SELECT id, type, amount, currency, category, note, created_at
            FROM movements
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,)
        )
        rows = cur.fetchall()

        cur.execute(
            """
            SELECT
              COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
              COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
            FROM movements
            """
        )
        totals = cur.fetchone()

        cur.close()
        conn.close()

        # serializza Decimal -> float
        for r in rows:
            if isinstance(r.get("amount"), decimal.Decimal):
                r["amount"] = float(r["amount"])
        if isinstance(totals.get("total_in"), decimal.Decimal):
            totals["total_in"] = float(totals["total_in"])
        if isinstance(totals.get("total_out"), decimal.Decimal):
            totals["total_out"] = float(totals["total_out"])
        totals["net"] = totals["total_in"] - totals["total_out"]

        return {"items": rows, "totals": totals}

    except Exception as e:
        return JSONResponse({"error": "db_failed_query", "detail": str(e)}, status_code=500)
