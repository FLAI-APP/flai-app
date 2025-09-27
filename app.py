import os
import time
import json
from fastapi import FastAPI, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# -------------------------
# Setup DB connection
# -------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session():
    return SessionLocal()

# -------------------------
# Setup FastAPI
# -------------------------
app = FastAPI()

# CORS
origins = [o for o in os.getenv("ALLOWED_ORIGINS","").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],   # in prod meglio senza *
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key
API_KEY_APP = os.getenv("API_KEY_APP", "")
if not API_KEY_APP:
    print("⚠️ Warning: API_KEY_APP not set!")

# -------------------------
# Middleware sicurezza
# -------------------------
WINDOW_SECONDS = 60
MAX_REQ = 60
_bucket = {}

def client_key(req: Request) -> str:
    xf = req.headers.get("x-forwarded-for")
    return (xf.split(",")[0].strip() if xf else req.client.host) + "|" + (req.headers.get("x-api-key") or "-")

@app.middleware("http")
async def security_and_rate_limit(request: Request, call_next):
    path = request.url.path
    open_paths = {"/", "/healthz"}

    if path not in open_paths:
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
            return JSONResponse({"error":"rate_limited","limit_per_min":MAX_REQ}, status_code=429)

    response = await call_next(request)
    return response

# -------------------------
# Endpoints
# -------------------------

@app.get("/")
async def root():
    return {"ok": True}

@app.get("/healthz")
async def healthz():
    try:
        with get_session() as s:
            s.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -------------------------
# Movements
# -------------------------
@app.post("/movements")
async def create_movement(item: dict = Body(...)):
    try:
        t = item.get("type")
        if t not in ("in","out"):
            raise HTTPException(422, "type must be 'in' or 'out'")
        amt = item.get("amount")
        cur = item.get("currency","CHF")
        cat = item.get("category")
        note = item.get("note")
        voce = item.get("voce","generale")

        with get_session() as s:
            s.execute(text("""
                INSERT INTO movements(type, amount, currency, category, note, voce, created_at)
                VALUES (:t,:a,:c,:cat,:n,:v, NOW())
            """), {"t":t,"a":amt,"c":cur,"cat":cat,"n":note,"v":voce})
            s.commit()
        return {"status":"ok"}
    except Exception as e:
        return {"error":"db_failed_insert","detail":str(e)}

@app.get("/movements")
async def list_movements(_from: str = Query(None, alias="from"), to: str = None):
    try:
        q = "SELECT id,type,amount,currency,category,note,voce,created_at FROM movements WHERE 1=1"
        params = {}
        if _from:
            q += " AND created_at >= :f"; params["f"] = _from
        if to:
            q += " AND created_at < :t"; params["t"] = to
        q += " ORDER BY created_at DESC LIMIT 200"

        with get_session() as s:
            rows = s.execute(text(q), params).mappings().all()
            totals = s.execute(text("""
                SELECT
                  COALESCE(SUM(CASE WHEN type='in' THEN amount END),0) AS total_in,
                  COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                FROM movements
                WHERE (:f IS NULL OR created_at >= :f)
                  AND (:t IS NULL OR created_at < :t)
            """), {"f":_from, "t":to}).mappings().first()

        return {"items": rows, "totals": totals}
    except Exception as e:
        return {"error":"db_failed_query","detail":str(e)}

# -------------------------
# Summaries
# -------------------------
@app.post("/summaries/generate")
async def generate_summary(days: int = Query(30, ge=1, le=365)):
    """
    Genera un riassunto degli ultimi X giorni.
    """
    try:
        with get_session() as s:
            rows = s.execute(text("""
                WITH days AS (
                  SELECT generate_series(
                    CURRENT_DATE - (:d - 1) * INTERVAL '1 day',
                    CURRENT_DATE,
                    INTERVAL '1 day'
                  )::date AS d
                ),
                agg AS (
                  SELECT
                    DATE(created_at) AS d,
                    COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0)  AS in_amt,
                    COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS out_amt
                  FROM movements
                  WHERE created_at >= CURRENT_DATE - (:d - 1) * INTERVAL '1 day'
                  GROUP BY DATE(created_at)
                )
                SELECT
                  days.d AS date,
                  COALESCE(agg.in_amt,0)  AS in_amount,
                  COALESCE(agg.out_amt,0) AS out_amount,
                  COALESCE(agg.in_amt,0) - COALESCE(agg.out_amt,0) AS net
                FROM days
                LEFT JOIN agg ON agg.d = days.d
                ORDER BY days.d ASC
            """), {"d": days}).mappings().all()

        return {"days": days, "data": list(rows)}

    except Exception as e:
        return {"error": "db_failed_summary", "detail": str(e)}
