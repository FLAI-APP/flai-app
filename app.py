import os
import time
import json
import datetime
import psycopg2
from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# --- CONFIG ---
DATABASE_URL = os.getenv("DATABASE_URL", "")
API_KEY_APP = os.getenv("API_KEY_APP", "")

# --- DB helper ---
def get_conn():
    return psycopg2.connect(DATABASE_URL)

# --- Middleware sicurezza base ---
@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if request.url.path not in ["/", "/healthz"]:
        if request.headers.get("x-api-key") != API_KEY_APP:
            return JSONResponse({"error": "invalid api key"}, status_code=401)
    return await call_next(request)

@app.get("/")
async def root():
    return {"ok": True}

@app.get("/healthz")
async def healthz():
    return {"ok": True}

# --- Movements demo ---
@app.post("/movements")
async def create_movement(item: dict = Body(...)):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO movements(type, amount, currency, category, note, created_at)
                    VALUES (%s,%s,%s,%s,%s,NOW())
                """, (
                    item.get("type"),
                    item.get("amount"),
                    item.get("currency","CHF"),
                    item.get("category"),
                    item.get("note")
                ))
        return {"status": "ok"}
    except Exception as e:
        return {"error": "db_failed_insert", "detail": str(e)}

@app.get("/movements")
async def list_movements():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, type, amount, currency, category, note, created_at
                    FROM movements ORDER BY created_at DESC LIMIT 50
                """)
                rows = cur.fetchall()
        return {"items": rows}
    except Exception as e:
        return {"error": "db_failed_query", "detail": str(e)}

# --- Summaries molto semplice ---
@app.post("/summaries/generate")
async def generate_summary(days: int = 30):
    """
    Calcola entrate, uscite e netto ultimi N giorni.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                      COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
                      COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                    FROM movements
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                """, (days,))
                res = cur.fetchone()
        total_in, total_out = res
        return {
            "window_days": days,
            "in": float(total_in),
            "out": float(total_out),
            "net": float(total_in - total_out)
        }
    except Exception as e:
        return {"error": "db_failed_summary", "detail": str(e)}

