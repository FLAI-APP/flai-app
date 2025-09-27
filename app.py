import os
import time
from fastapi import FastAPI, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from datetime import datetime

# --- INIT ---
APP = FastAPI()
DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY_APP = os.getenv("API_KEY_APP", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

engine = create_engine(DATABASE_URL)

# --- CORS ---
origins = [o for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
APP.add_middleware(
    CORSMiddleware,
    allow_origins=origins or [],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API KEY + Rate limit ---
WINDOW_SECONDS = 60
MAX_REQ = 60
_bucket = {}

def client_key(req: Request) -> str:
    xf = req.headers.get("x-forwarded-for")
    return (xf.split(",")[0].strip() if xf else req.client.host) + "|" + (req.headers.get("x-api-key") or "-")

@APP.middleware("http")
async def security_and_rate_limit(request: Request, call_next):
    if request.url.path not in {"/", "/healthz"}:
        if not API_KEY_APP:
            return JSONResponse({"error": "server_misconfigured_no_api_key"}, status_code=500)
        if request.headers.get("x-api-key") != API_KEY_APP:
            raise HTTPException(status_code=401, detail="invalid api key")
        now = int(time.time())
        key = client_key(request)
        wstart, cnt = _bucket.get(key, (now, 0))
        if now - wstart >= WINDOW_SECONDS:
            wstart, cnt = now, 0
        cnt += 1
        _bucket[key] = (wstart, cnt)
        if cnt > MAX_REQ:
            return JSONResponse({"error": "rate_limited", "limit_per_min": MAX_REQ}, status_code=429)
    return await call_next(request)

# --- ROOT + HEALTH ---
@APP.get("/")
def root():
    return {"status": "ok"}

@APP.get("/healthz")
def healthz():
    return "ok"

# --- ANALYTICS ---
@APP.get("/analytics/overview")
def analytics_overview(days: int = Query(7, ge=1, le=90)):
    try:
        with engine.begin() as conn:
            rows = conn.execute(text("""
                WITH days AS (
                  SELECT generate_series(
                    (CURRENT_DATE - (:d - 1) * INTERVAL '1 day')::date,
                    CURRENT_DATE::date,
                    INTERVAL '1 day'
                  )::date AS d
                ), agg AS (
                  SELECT
                    DATE(created_at) AS d,
                    COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS in_amt,
                    COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS out_amt
                  FROM movements
                  WHERE created_at >= CURRENT_DATE - (:d - 1) * INTERVAL '1 day'
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
        return {"days": rows}
    except Exception as e:
        return {"error": "db_failed_analytics", "detail": str(e)}

@APP.get("/analytics/messages")
def analytics_messages():
    try:
        with engine.begin() as conn:
            row = conn.execute(text("SELECT COUNT(*) AS total FROM messages")).mappings().first()
        return {"messages_total": row["total"]}
    except Exception as e:
        return {"error": "db_failed_messages", "detail": str(e)}

@APP.get("/analytics/movements")
def analytics_movements():
    try:
        with engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT id, type, amount, currency, category, note, created_at
                FROM movements
                ORDER BY created_at DESC
                LIMIT 50
            """)).mappings().all()
            totals = conn.execute(text("""
                SELECT
                  COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
                  COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                FROM movements
            """)).mappings().first()
        return {"items": rows, "totals": totals}
    except Exception as e:
        return {"error": "db_failed_movements", "detail": str(e)}
