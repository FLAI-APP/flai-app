import os
import time
import json
from datetime import date, datetime, timedelta
from typing import Optional, List

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# CONFIG DA ENV
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
API_KEY_APP = os.getenv("API_KEY_APP", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# -----------------------------
# FASTAPI APP + CORS
# -----------------------------
app = FastAPI(title="flai-app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# API KEY + RATE LIMIT (semplice)
# -----------------------------
WINDOW_SECONDS = 60
MAX_REQ = 60
_bucket = {}  # key -> (window_start_ts, count)

def _client_key(req: Request) -> str:
    xf = req.headers.get("x-forwarded-for")
    ip = xf.split(",")[0].strip() if xf else (req.client.host if req.client else "unknown")
    api = req.headers.get("x-api-key", "-")
    return f"{ip}|{api}"

@app.middleware("http")
async def _security(request: Request, call_next):
    path = request.url.path
    # endpoint pubblici
    open_paths = {"/", "/healthz", "/webhook"}  # /webhook GET deve restare pubblico per Meta
    if path not in open_paths:
        if not API_KEY_APP or request.headers.get("x-api-key") != API_KEY_APP:
            raise HTTPException(status_code=401, detail="invalid api key")

        # rate-limit soft
        now = int(time.time())
        key = _client_key(request)
        wstart, cnt = _bucket.get(key, (now, 0))
        if now - wstart >= WINDOW_SECONDS:
            wstart, cnt = now, 0
        cnt += 1
        _bucket[key] = (wstart, cnt)
        if cnt > MAX_REQ:
            return JSONResponse({"error": "rate_limited", "limit_per_min": MAX_REQ}, status_code=429)

    try:
        resp = await call_next(request)
        return resp
    except HTTPException:
        raise
    except Exception as e:
        if DEBUG:
            return JSONResponse({"error": "internal_error", "detail": str(e)}, status_code=500)
        return JSONResponse({"error": "internal_error"}, status_code=500)

# -----------------------------
# DB UTILS
# -----------------------------
def _conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    # connessione con dict cursor
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)

def _bootstrap_db():
    """
    Crea tabelle se mancano. Non tocca colonne extra legacy.
    """
    ddl_messages = """
    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        reply   TEXT,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
    );
    """
    ddl_movements = """
    CREATE TABLE IF NOT EXISTS movements (
        id SERIAL PRIMARY KEY,
        type TEXT NOT NULL,                 -- 'in' | 'out'
        amount NUMERIC(12,2) NOT NULL,
        currency TEXT NOT NULL DEFAULT 'CHF',
        category TEXT,
        note TEXT,
        created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
    );
    """
    with _conn() as c, c.cursor() as cur:
        cur.execute(ddl_messages)
        cur.execute(ddl_movements)
        c.commit()

_bootstrap_db()

# -----------------------------
# OPENAI
# -----------------------------
from openai import OpenAI
_openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMP = float(os.getenv("OPENAI_TEMP", "0.3"))

def ai_reply(prompt: str) -> str:
    if not _openai:
        return "(OpenAI non configurato) " + prompt[:200]
    try:
        rsp = _openai.chat.completions.create(
            model=MODEL,
            temperature=TEMP,
            messages=[
                {"role": "system", "content": "Sei un assistente aziendale conciso e pratico."},
                {"role": "user", "content": prompt}
            ],
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        if DEBUG:
            return f"(AI error: {e})"
        return "(AI non disponibile)"

# -----------------------------
# BASICS
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@app.get("/healthz")
def healthz():
    return "ok"

# -----------------------------
# WHATSAPP VERIFY + DEMO POST
# -----------------------------
@app.get("/webhook")
def whatsapp_verify(
    hub_mode: str = Query("", alias="hub.mode"),
    hub_challenge: str = Query("", alias="hub.challenge"),
    hub_verify_token: str = Query("", alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "", media_type="text/plain")
    # Se uno chiama /webhook senza i parametri, non è errore (evita 500 in produzione)
    return PlainTextResponse("forbidden", status_code=403)

@app.post("/webhook")
def webhook_demo(payload: dict = Body(...)):
    """
    DEMO: se arriva {"message": "..."} rispondiamo con AI e salviamo su messages.
    (Quando WhatsApp sarà sbloccato, qui si gestirà anche il formato Meta.)
    """
    message = (payload or {}).get("message")
    if not message:
        return {"ok": True, "note": "no-demo-message"}
    reply = ai_reply(message)

    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO messages(content, reply) VALUES (%s,%s) RETURNING id, created_at",
            (message, reply),
        )
        row = cur.fetchone()
        c.commit()

    return {"reply": reply, "id": row["id"], "created_at": row["created_at"].isoformat()}

# -----------------------------
# MESSAGES
# -----------------------------
@app.get("/messages")
def messages_list(limit: int = 20):
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT id, content, reply, created_at FROM messages ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    return {"items": rows}

# -----------------------------
# MOVEMENTS
# -----------------------------
@app.post("/movements")
def movement_create(item: dict = Body(...)):
    """
    Body minimo:
    {
      "type":"in"|"out",
      "amount": 123.45,
      "currency": "CHF",
      "category": "sales",
      "note": "..."
    }
    """
    t = (item.get("type") or "").lower()
    if t not in ("in", "out"):
        raise HTTPException(422, "type must be 'in' or 'out'")
    try:
        amt = float(item.get("amount"))
    except Exception:
        raise HTTPException(422, "amount must be a number")

    curcy = item.get("currency", "CHF")
    cat = item.get("category")
    note = item.get("note")

    with _conn() as c, c.cursor() as cur:
        cur.execute(
            """
            INSERT INTO movements(type, amount, currency, category, note, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            RETURNING id, created_at
            """,
            (t, amt, curcy, cat, note),
        )
        row = cur.fetchone()
        c.commit()

    return {"status": "ok", "id": row["id"], "created_at": row["created_at"].isoformat()}

@app.get("/movements")
def movements_list(
    _from: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    limit: int = 200,
):
    q = """SELECT id, type, amount, currency, category, note, created_at
           FROM movements WHERE 1=1"""
    params: List = []
    if _from:
        q += " AND created_at >= %s"
        params.append(_from)
    if to:
        q += " AND created_at < %s"
        params.append(to)
    q += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    with _conn() as c, c.cursor() as cur:
        cur.execute(q, tuple(params))
        items = cur.fetchall()

        # totali nel periodo richiesto
        tq = """
            SELECT
              COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
              COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
            FROM movements
            WHERE (%s IS NULL OR created_at >= %s)
              AND (%s IS NULL OR created_at < %s)
        """
        cur.execute(tq, (_from, _from, to, to))
        totals = cur.fetchone()

    totals["net"] = float(totals["total_in"] or 0) - float(totals["total_out"] or 0)
    return {"items": items, "totals": totals}

# -----------------------------
# ANALYTICS (serie giornaliera)
# -----------------------------
@app.get("/analytics/overview")
def analytics_overview(days: int = 7):
    """
    Ritorna serie degli ultimi N giorni: in, out, net per giorno.
    Calcolo fatto in SQL usando generate_series, parametrizzando le date
    per evitare i problemi di cast.
    """
    if days < 1:
        days = 1
    start_date = (date.today() - timedelta(days=days - 1)).isoformat()

    sql = """
        WITH days AS (
          SELECT d::date
          FROM generate_series(%s::date, CURRENT_DATE, INTERVAL '1 day') AS g(d)
        ),
        agg AS (
          SELECT
            DATE(created_at) AS d,
            COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS in_amt,
            COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS out_amt
          FROM movements
          WHERE created_at >= %s::date
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
    """
    with _conn() as c, c.cursor() as cur:
        cur.execute(sql, (start_date, start_date))
        rows = cur.fetchall()

        # messaggi totali nel periodo (semplice)
        cur.execute(
            "SELECT COUNT(*) AS total_messages FROM messages WHERE created_at >= %s::date",
            (start_date,),
        )
        m = cur.fetchone()

    return {"days": rows, "messages": m["total_messages"]}

# -----------------------------
# SUMMARIES (KPI + AI)
# -----------------------------
@app.post("/summaries/generate")
def summaries_generate(days: int = 30):
    if days < 1:
        days = 1
    start_date = (date.today() - timedelta(days=days - 1)).isoformat()

    with _conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT
              COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
              COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
            FROM movements
            WHERE created_at >= %s::date
            """,
            (start_date,),
        )
        t = cur.fetchone()

        cur.execute(
            "SELECT COUNT(*) AS total_messages FROM messages WHERE created_at >= %s::date",
            (start_date,),
        )
        m = cur.fetchone()

    net = float(t["total_in"] or 0) - float(t["total_out"] or 0)
    kpi = {
        "window_days": days,
        "total_in": float(t["total_in"] or 0),
        "total_out": float(t["total_out"] or 0),
        "net": net,
        "messages": int(m["total_messages"] or 0),
        "since": start_date,
        "until": date.today().isoformat(),
    }

    # Testo naturale via OpenAI (best effort)
    prompt = (
        "Genera un breve riassunto in italiano dei KPI:\n"
        + json.dumps(kpi, ensure_ascii=False)
        + "\nStile: pratico, positivo, con 1 consiglio operativo."
    )
    text = ai_reply(prompt)

    return {"kpi": kpi, "summary": text}
