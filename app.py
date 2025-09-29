import os
import json
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse

from fastapi import FastAPI, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import pg8000.dbapi as pg
from decimal import Decimal

# =========================
# CONFIG / ENV
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL   = os.getenv("DATABASE_URL", "")
API_KEY_APP    = os.getenv("API_KEY_APP", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "flai-verify-123")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# =========================
# FASTAPI
# =========================
app = FastAPI(title="flai-app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DB helper (pg8000)
# =========================
def _pg_params_from_url(db_url: str) -> dict:
    """
    Converte postgresql://user:pass@host:port/dbname in kwargs per pg8000.dbapi.connect
    """
    u = urlparse(db_url)
    if u.scheme not in ("postgresql", "postgres"):
        raise RuntimeError("DATABASE_URL must start with postgresql://")
    return {
        "user":     u.username,
        "password": u.password,
        "host":     u.hostname,
        "port":     u.port or 5432,
        "database": u.path.lstrip("/"),
        "ssl_context": True,  # Render usa SSL
    }

def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    return pg.connect(**_pg_params_from_url(DATABASE_URL))

def _rows_to_dicts(cur):
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]

def _exec(sql: str, params: dict | tuple | None = None, fetch: bool = False):
    """
    Esegue SQL con commit automatico. Se fetch=True ritorna list[dict]
    """
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or {})
            if fetch and cur.description:
                return _rows_to_dicts(cur)
            conn.commit()
    return None

# =========================
# BOOTSTRAP SCHEMA
# =========================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  reply   TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS movements (
  id SERIAL PRIMARY KEY,
  message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
  type VARCHAR(10) NOT NULL,             -- 'in' | 'out'
  voce TEXT NOT NULL DEFAULT 'generale',
  valuta VARCHAR(8) NOT NULL DEFAULT 'CHF',
  note  TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  amount NUMERIC(14,2) NOT NULL,
  currency VARCHAR(8) NOT NULL DEFAULT 'CHF',
  category VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_movements_created ON movements(created_at);
"""

def bootstrap_schema():
    for statement in SCHEMA_SQL.strip().split(";\n\n"):
        s = (statement or "").strip()
        if s:
            _exec(s)

# =========================
# SECURITY (API KEY + rate limit soft)
# =========================
WINDOW_SECONDS = 60
MAX_REQ = 120  # per chiave/min
_bucket: dict[str, tuple[int, int]] = {}

@app.middleware("http")
async def sec_and_rate(request: Request, call_next):
    path = request.url.path
    open_paths = {"/", "/healthz", "/webhook"}  # /webhook deve restare aperto per la verifica Meta
    if path not in open_paths:
        if not API_KEY_APP:
            return JSONResponse({"error":"server_misconfigured_no_api_key"}, status_code=500)
        if request.headers.get("x-api-key") != API_KEY_APP:
            raise HTTPException(status_code=401, detail="invalid api key")

        # rate limit (chiave + ip)
        ip = (request.headers.get("x-forwarded-for") or request.client.host or "-").split(",")[0].strip()
        key = f"{ip}|{request.headers.get('x-api-key') or '-'}"
        now = int(time.time())
        w, c = _bucket.get(key, (now, 0))
        if now - w >= WINDOW_SECONDS:
            w, c = now, 0
        c += 1
        _bucket[key] = (w, c)
        if c > MAX_REQ:
            return JSONResponse({"error":"rate_limited","limit_per_min":MAX_REQ}, status_code=429)

    return await call_next(request)

# =========================
# ROUTES BASE
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@app.get("/healthz")
def healthz():
    try:
        bootstrap_schema()
        return "ok"
    except Exception as e:
        return JSONResponse({"error":"db_bootstrap_failed","detail":str(e)}, status_code=500)

# =========================
# /messages  (AI demo + persistenza)
# =========================
from openai import OpenAI
_openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@app.post("/messages")
def post_message(payload: dict = Body(...)):
    """
    Body: { "message": "..." }
    """
    bootstrap_schema()
    msg = (payload or {}).get("message", "")
    if not msg:
        raise HTTPException(422, "message is required")

    # Chiamata AI (se configurata), altrimenti eco
    if _openai:
        try:
            res = _openai.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content": msg}],
                temperature=0.3,
            )
            reply = res.choices[0].message.content.strip()
        except Exception as e:
            reply = f"(AI temporarily unavailable) {e}"
    else:
        reply = f"Hai scritto: {msg}"

    # Persistenza
    _exec(
        "INSERT INTO messages(content, reply, created_at) VALUES (%(c)s,%(r)s,NOW())",
        {"c": msg, "r": reply},
        fetch=False
    )
    row = _exec(
        "SELECT id, content, reply, created_at FROM messages ORDER BY id DESC LIMIT 1",
        fetch=True
    )[0]
    return row

@app.get("/messages/recent")
def recent_messages(limit: int = 20):
    bootstrap_schema()
    rows = _exec(
        "SELECT id, content, reply, created_at FROM messages ORDER BY created_at DESC LIMIT %(lim)s",
        {"lim": limit},
        fetch=True
    )
    return rows

# =========================
# MOVEMENTS
# =========================
@app.post("/movements")
def create_movement(item: dict = Body(...)):
    """
    Body: { "type":"in|out", "amount":123.45, "currency":"CHF", "category":"...", "note":"...", "voce":"...", "message_id":1 }
    """
    bootstrap_schema()
    t = item.get("type")
    if t not in ("in", "out"):
        raise HTTPException(422, "type must be 'in' or 'out'")

    amt = Decimal(str(item.get("amount", 0)))
    if amt <= 0:
        raise HTTPException(422, "amount must be > 0")

    params = {
        "mid": item.get("message_id"),
        "t": t,
        "v": item.get("voce", "generale"),
        "val": item.get("valuta", "CHF"),
        "n": item.get("note"),
        "a": amt,
        "c": item.get("currency", "CHF"),
        "cat": item.get("category"),
    }

    _exec(
        """
        INSERT INTO movements(message_id,type,voce,valuta,note,created_at,amount,currency,category)
        VALUES (%(mid)s,%(t)s,%(v)s,%(val)s,%(n)s,NOW(),%(a)s,%(c)s,%(cat)s)
        """,
        params
    )
    return {"status":"ok"}

@app.get("/movements")
def list_movements(_from: str | None = Query(None, alias="from"), to: str | None = None, limit: int = 200):
    bootstrap_schema()
    where = ["1=1"]
    p = {"lim": limit}
    if _from:
        where.append("created_at >= %(f)s")
        p["f"] = _from
    if to:
        where.append("created_at < %(t)s")
        p["t"] = to
    sql = f"""
        SELECT id, message_id, type, voce, valuta, note, created_at, amount, currency, category
        FROM movements
        WHERE {' AND '.join(where)}
        ORDER BY created_at DESC
        LIMIT %(lim)s
    """
    rows = _exec(sql, p, fetch=True)
    # Totali
    totals = _exec(
        f"""
        SELECT
          COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0)  AS total_in,
          COALESCE(SUM(CASE WHEN type='out' THEN amount END),0)  AS total_out
        FROM movements
        WHERE {' AND '.join(where)}
        """,
        p,
        fetch=True
    )[0]
    totals["net"] = Decimal(totals["total_in"]) - Decimal(totals["total_out"])
    return {"items": rows, "totals": totals}

# =========================
# ANALYTICS
# =========================
@app.get("/analytics/overview")
def analytics_overview(days: int = 30):
    """
    Ritorna kpi base per ultimi N giorni
    """
    bootstrap_schema()
    days = max(1, min(days, 90))
    # Serie giorni
    start = (datetime.utcnow().date() - timedelta(days=days-1)).isoformat()

    # messaggi
    msgs = _exec(
        """
        SELECT DATE(created_at) AS d, COUNT(*) AS n
        FROM messages
        WHERE created_at >= %(start)s
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at)
        """,
        {"start": start},
        fetch=True
    )

    # movimenti
    mov = _exec(
        """
        SELECT DATE(created_at) AS d,
               COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS in_amt,
               COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS out_amt
        FROM movements
        WHERE created_at >= %(start)s
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at)
        """,
        {"start": start},
        fetch=True
    )

    return {"messages_by_day": msgs, "movements_by_day": mov}

# =========================
# SUMMARIES (testo naturale)
# =========================
@app.post("/summaries/generate")
def generate_summary(days: int = 30):
    bootstrap_schema()
    days = max(1, min(days, 90))
    # recupera totali
    totals = _exec(
        """
        SELECT
          COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
          COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
        FROM movements
        WHERE created_at >= NOW() - %(days)s::interval
        """,
        {"days": f"{days} days"},
        fetch=True
    )[0]
    totals["net"] = Decimal(totals["total_in"]) - Decimal(totals["total_out"])

    # genera testo con AI se disponibile
    if _openai:
        prompt = f"""Sei un assistente che riassume KPI aziendali in modo chiaro.
Ultimi {days} giorni:
- Entrate (in): {totals['total_in']}
- Uscite (out): {totals['total_out']}
- Netto: {totals['net']}
Scrivi 2 frasi di insight in italiano, tono professionale ma semplice."""
        try:
            res = _openai.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content": prompt}],
                temperature=0.2,
            )
            text = res.choices[0].message.content.strip()
        except Exception as e:
            text = f"Riassunto semplice: netto {totals['net']} (in {totals['total_in']} / out {totals['total_out']}). [AI error: {e}]"
    else:
        text = f"Netto {totals['net']} negli ultimi {days} giorni (Entrate {totals['total_in']} / Uscite {totals['total_out']})."

    return {"totals": totals, "summary": text}

# =========================
# WEBHOOK META (verify + eventi)
# =========================
@app.get("/webhook")
def whatsapp_verify(hub_mode: str = "", hub_challenge: str = "", hub_verify_token: str = ""):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "", media_type="text/plain")
    raise HTTPException(403, "verification_failed")

@app.post("/webhook")
async def whatsapp_events(payload: dict = Body(...)):
    """
    Accetta sia test interni {message:"..."} sia notifiche Meta (semplificato)
    """
    if "message" in payload:
        # riusa /messages
        return post_message({"message": payload["message"]})

    # parse minimale formato Meta
    try:
        entry = payload.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value  = change.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return {"status":"ok"}
        text = messages[0].get("text", {}).get("body") or "[no text]"
        return post_message({"message": text})
    except Exception as e:
        if DEBUG:
            return {"status":"ok","note":f"unparsed payload: {str(e)}"}
        return {"status":"ok"}
