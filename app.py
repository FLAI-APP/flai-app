import os
import time
import json
import decimal
from typing import Optional, Dict, Any

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ----------------- CONFIG -----------------
app = FastAPI(title="flai-app")

API_KEY_APP   = os.getenv("API_KEY_APP", "").strip()
DATABASE_URL  = os.getenv("DATABASE_URL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# CORS (domini separati da virgola)
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS","").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- SECURITY + RATE LIMIT -----------------
WINDOW_SECONDS = 60
MAX_REQ = 120
_bucket: Dict[str, Any] = {}

def _client_key(req: Request) -> str:
    xf = req.headers.get("x-forwarded-for")
    ip = xf.split(",")[0].strip() if xf else (req.client.host if req.client else "unknown")
    return ip + "|" + (req.headers.get("x-api-key") or "-")

@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    path = request.url.path
    if path not in {"/", "/healthz"}:
        if not API_KEY_APP:
            return JSONResponse({"error":"server_no_api_key"}, status_code=500)
        if request.headers.get("x-api-key") != API_KEY_APP:
            raise HTTPException(status_code=401, detail="invalid api key")

        now = int(time.time())
        key = _client_key(request)
        wstart, cnt = _bucket.get(key, (now, 0))
        if now - wstart >= WINDOW_SECONDS:
            wstart, cnt = now, 0
        cnt += 1
        _bucket[key] = (wstart, cnt)
        if cnt > MAX_REQ:
            return JSONResponse({"error":"rate_limited","limit_per_min":MAX_REQ}, status_code=429)

    return await call_next(request)

# ----------------- DB HELPERS (psycopg v3) -----------------
def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    # row_factory=dict_row => ritorna dict
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def _d(v):
    return float(v) if isinstance(v, decimal.Decimal) else v

# ----------------- ROOT/HEALTH -----------------
@app.get("/")
def root():
    return {"status":"ok","service":"flai-app"}

@app.get("/healthz")
def healthz():
    try:
        with db_conn() as con, con.cursor() as cur:
            cur.execute("SELECT 1 AS ok;")
            row = cur.fetchone()
        return {"ok": True, "db": row["ok"] == 1}
    except Exception as e:
        return {"ok": False, "db_error": str(e)}

# ----------------- MESSAGES -----------------
@app.get("/messages")
def list_messages(limit: int = 20):
    sql = """
        SELECT id, content, reply, created_at
        FROM messages
        ORDER BY created_at DESC
        LIMIT %s
    """
    try:
        with db_conn() as con, con.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        rows = [{k:_d(v) for k,v in r.items()} for r in rows]
        return {"items": rows}
    except Exception as e:
        return JSONResponse({"error":"db_failed_query","detail":str(e)}, status_code=500)

# ----------------- WEBHOOK (AI demo) -----------------
@app.post("/webhook")
def webhook(payload: Dict[str, Any] = Body(...)):
    msg = (payload or {}).get("message", "").strip()
    if not msg:
        raise HTTPException(422, "missing message")

    # genera reply (OpenAI se disponibile, altrimenti eco)
    reply_text = f"Hai scritto: {msg}"
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            oai = OpenAI(api_key=OPENAI_API_KEY)
            resp = oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"Rispondi in modo utile e sintetico."},
                    {"role":"user","content": msg}
                ],
                temperature=0.3,
                max_tokens=200
            )
            reply_text = resp.choices[0].message.content.strip()
        except Exception:
            pass

    ins = """
        INSERT INTO messages (content, reply, created_at)
        VALUES (%s, %s, NOW())
        RETURNING id
    """
    try:
        with db_conn() as con, con.cursor() as cur:
            cur.execute(ins, (msg, reply_text))
            saved_id = cur.fetchone()["id"]
            con.commit()
        return {"id": saved_id, "reply": reply_text}
    except Exception as e:
        return JSONResponse({"error":"db_failed_insert","detail":str(e)}, status_code=500)

# ----------------- MOVEMENTS (schema della tua tabella) -----------------
@app.post("/movements")
def create_movement(item: Dict[str, Any] = Body(...)):
    t = (item.get("type") or "").strip()
    if t not in ("in","out"):
        raise HTTPException(422, "type must be 'in' or 'out'")

    try:
        amount = decimal.Decimal(str(item.get("amount")))
    except Exception:
        raise HTTPException(422, "amount must be a number")

    message_id = item.get("message_id")
    voce   = (item.get("voce") or "generale").strip()
    valuta = (item.get("valuta") or "CHF").strip()      # colonna 'valuta'
    note   = (item.get("note") or "").strip()
    currency = (item.get("currency") or "CHF").strip()  # colonna 'currency'
    category = (item.get("category") or "").strip()

    sql = """
        INSERT INTO movements (message_id, type, voce, valuta, note, created_at, amount, currency, category)
        VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s)
        RETURNING id
    """
    try:
        with db_conn() as con, con.cursor() as cur:
            cur.execute(sql, (message_id, t, voce, valuta, note, amount, currency, category))
            rid = cur.fetchone()["id"]
            con.commit()
        return {"status":"ok","id":rid}
    except Exception as e:
        return JSONResponse({"error":"db_failed_insert","detail":str(e)}, status_code=500)

@app.get("/movements")
def list_movements(
    _from: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    limit: int = 200
):
    base = """
        SELECT id, message_id, type, voce, valuta, note, created_at, amount, currency, category
        FROM movements
        WHERE 1=1
    """
    params = []
    if _from:
        base += " AND created_at >= %s"; params.append(_from)
    if to:
        base += " AND created_at < %s"; params.append(to)
    base += " ORDER BY created_at DESC LIMIT %s"; params.append(limit)

    try:
        with db_conn() as con, con.cursor() as cur:
            cur.execute(base, tuple(params))
            items = cur.fetchall()

            qsum = """
                SELECT
                  COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
                  COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                FROM movements
                WHERE 1=1
            """
            p2 = []
            if _from:
                qsum += " AND created_at >= %s"; p2.append(_from)
            if to:
                qsum += " AND created_at < %s"; p2.append(to)
            cur.execute(qsum, tuple(p2))
            totals = cur.fetchone()

        items  = [{k:_d(v) for k,v in r.items()} for r in items]
        totals = {k:_d(v) for k,v in totals.items()}
        totals["net"] = _d(decimal.Decimal(str(totals["total_in"])) - decimal.Decimal(str(totals["total_out"])))
        return {"items": items, "totals": totals}
    except Exception as e:
        return JSONResponse({"error":"db_failed_query","detail":str(e)}, status_code=500)

# ----------------- ANALYTICS -----------------
@app.get("/analytics/overview")
def analytics_overview(days: int = 30):
    if days < 1 or days > 365:
        raise HTTPException(422, "days must be between 1 and 365")
    try:
        with db_conn() as con, con.cursor() as cur:
            cur.execute(
                """
                WITH days AS (
                  SELECT generate_series(
                    (CURRENT_DATE - (%s - 1) * INTERVAL '1 day')::date,
                    CURRENT_DATE::date,
                    INTERVAL '1 day'
                  )::date AS d
                ),
                agg AS (
                  SELECT DATE(created_at) AS d,
                    COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS in_amt,
                    COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS out_amt
                  FROM movements
                  WHERE created_at >= CURRENT_DATE - (%s - 1) * INTERVAL '1 day'
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
                """,
                (days, days)
            )
            rows = cur.fetchall()

            cur.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
                  COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                FROM movements
                WHERE created_at >= CURRENT_DATE - (%s - 1) * INTERVAL '1 day'
                """,
                (days,)
            )
            totals = cur.fetchone()

        rows = [{k:_d(v) for k,v in r.items()} for r in rows]
        totals = {k:_d(v) for k,v in totals.items()}
        totals["net"] = _d(decimal.Decimal(str(totals["total_in"])) - decimal.Decimal(str(totals["total_out"])))
        return {"days": rows, "totals": totals}
    except Exception as e:
        return JSONResponse({"error":"db_failed_analytics","detail":str(e)}, status_code=500)

# ----------------- SUMMARIES -----------------
@app.post("/summaries/generate")
def generate_summary(days: int = 30):
    try:
        with db_conn() as con, con.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS messages_count FROM messages WHERE created_at >= CURRENT_DATE - (%s - 1) * INTERVAL '1 day'",
                (days,)
            )
            m = cur.fetchone()
            cur.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
                  COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                FROM movements
                WHERE created_at >= CURRENT_DATE - (%s - 1) * INTERVAL '1 day'
                """,
                (days,)
            )
            mv = cur.fetchone()
    except Exception as e:
        return JSONResponse({"error":"db_failed_kpi","detail":str(e)}, status_code=500)

    total_in = _d(mv["total_in"])
    total_out = _d(mv["total_out"])
    net = _d(decimal.Decimal(str(total_in)) - decimal.Decimal(str(total_out)))
    kpi = {
        "window_days": days,
        "messages": int(m["messages_count"]),
        "revenue_in": total_in,
        "expenses_out": total_out,
        "net": net
    }

    summary_text = (
        f"Ultimi {days} giorni: messaggi {kpi['messages']}, "
        f"entrate {kpi['revenue_in']}, uscite {kpi['expenses_out']}, "
        f"netto {kpi['net']}."
    )
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            oai = OpenAI(api_key=OPENAI_API_KEY)
            prompt = "Scrivi un breve riepilogo manageriale (5-8 frasi) dei KPI:\n" + json.dumps(kpi, ensure_ascii=False)
            resp = oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content": prompt}],
                temperature=0.4,
                max_tokens=300
            )
            summary_text = resp.choices[0].message.content.strip()
        except Exception:
            pass

    return {"kpi": kpi, "summary": summary_text}
