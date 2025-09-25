import os, time
from datetime import datetime, timedelta
from decimal import Decimal

from fastapi import FastAPI, Request, HTTPException, Query, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openai import OpenAI

# =========================
# App & CORS
# =========================
APP = FastAPI()

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS","").split(",") if o.strip()]
APP.add_middleware(
    CORSMiddleware,
    allow_origins=origins or [],   # niente wildcard in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ENV
# =========================
API_KEY_APP       = os.getenv("API_KEY_APP", "")
DEBUG             = os.getenv("DEBUG","false").lower() == "true"
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN","")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY","")
DATABASE_URL      = (os.getenv("DATABASE_URL") or "").strip()
if DATABASE_URL and "sslmode=" not in DATABASE_URL:
    sep = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = DATABASE_URL + f"{sep}sslmode=require"

# =========================
# Clients
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# =========================
# Schema DB (auto-create se non esiste)
# =========================
with engine.begin() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS messages (
      id SERIAL PRIMARY KEY,
      content   TEXT NOT NULL,
      reply     TEXT,
      created_at TIMESTAMP DEFAULT NOW()
    );
    """))
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS movements (
      id SERIAL PRIMARY KEY,
      type VARCHAR(10) NOT NULL,          -- 'in' | 'out'
      amount NUMERIC(14,2) NOT NULL,
      currency VARCHAR(8) NOT NULL DEFAULT 'CHF',
      category VARCHAR(50),
      note TEXT,
      created_at TIMESTAMP DEFAULT NOW()
    );
    """))

# =========================
# Rate limit soft (per pod)
# =========================
WINDOW_SECONDS = 60
MAX_REQ = 60
_bucket = {}  # {key: (window_start_ts, count)}

def _client_key(req: Request) -> str:
    xf = req.headers.get("x-forwarded-for")
    ip = xf.split(",")[0].strip() if xf else (req.client.host if req.client else "0.0.0.0")
    return ip + "|" + (req.headers.get("x-api-key") or "-")

@APP.middleware("http")
async def security_and_rate_limit(request: Request, call_next):
    path = request.url.path

    # endpoint aperti (Meta non manda header custom)
        open_paths = {"/", "/healthz", "/webhook", "/debug", "/echo"}


    if path not in open_paths:
        if not API_KEY_APP:
            return JSONResponse({"error":"server_misconfigured_no_api_key"}, status_code=500)
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

    response = await call_next(request)
    return response

# =========================
# Schemi pydantic
# =========================
class IncomingMessage(BaseModel):
    message: str

# =========================
# Routes base
# =========================
@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

# =========================
# WhatsApp verify + events (unificato su /webhook)
# =========================
@APP.get("/webhook")
async def whatsapp_verify(hub_mode: str = "", hub_challenge: str = "", hub_verify_token: str = ""):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "", media_type="text/plain")
    raise HTTPException(status_code=403, detail="verification_failed")

@APP.post("/webhook")
async def whatsapp_events(payload: dict = Body(...)):
    # Demo: {"message": "..."}  (nostro test manuale)
    if "message" in payload:
        user_msg = (payload.get("message") or "").strip()
        if not user_msg:
            return {"error":"no message"}
        # AI
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"Sei un assistente aziendale. Rispondi chiaro, concreto e conciso."},
                    {"role":"user","content": user_msg}
                ],
                temperature=0.2
            )
            reply_text = resp.choices[0].message.content.strip()
        except Exception as e:
            return {"error":"openai_failed","detail":str(e)}
        # Save
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO messages (content, reply, created_at) VALUES (:c,:r,:t)"),
                    {"c": user_msg, "r": reply_text, "t": datetime.utcnow()}
                )
        except Exception as e:
            return {"error":"db_failed","detail":str(e)}
        return {"reply": reply_text}

    # Meta payload (semplificato)
    try:
        entry = payload.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return {"status":"ok"}
        msg = messages[0]
        text_in = msg.get("text", {}).get("body", "") or "[unsupported message type]"
        # AI (preview)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"Sei un assistente aziendale. Rispondi chiaro, concreto e conciso."},
                    {"role":"user","content": text_in}
                ],
                temperature=0.2
            )
            reply_text = resp.choices[0].message.content.strip()
        except Exception as e:
            reply_text = f"[openai_failed: {e}]"
        # Save
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO messages (content, reply, created_at) VALUES (:c,:r,:t)"),
                    {"c": text_in, "r": reply_text, "t": datetime.utcnow()}
                )
        except Exception:
            pass
        return {"status":"ok","preview_reply": reply_text[:200]}
    except Exception as e:
        return JSONResponse({"error":"wa_parse_error","detail":str(e)}, status_code=200)

# =========================
# Messages (protetto da API key)
# =========================
@APP.get("/messages")
def list_messages(limit: int = 20):
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text("SELECT id, content, reply, created_at FROM messages ORDER BY created_at DESC LIMIT :lim"),
                {"lim": limit}
            ).mappings().all()
        return {"items": [dict(r) for r in rows]}
    except Exception as e:
        return {"error":"db_failed", "detail": str(e)}

# =========================
# Movements + Summary (protetti)
# =========================
@APP.post("/movements")
def create_movement(item: dict = Body(...)):
    t = (item.get("type") or "").strip().lower()
    if t not in ("in","out"):
        raise HTTPException(422, "type must be 'in' or 'out'")
    amt = item.get("amount")
    try:
        amt = float(amt)
    except:
        raise HTTPException(422, "amount must be numeric")
    cur = (item.get("currency") or "CHF").upper()
    cat = (item.get("category") or None)
    note = (item.get("note") or None)
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO movements(type,amount,currency,category,note,created_at)
                    VALUES (:t,:a,:c,:cat,:n,:ts)"""),
            {"t":t, "a":Decimal(str(amt)), "c":cur, "cat":cat, "n":note, "ts": datetime.utcnow()}
        )
    return {"status":"ok"}

@APP.get("/movements")
def list_movements(_from: str = Query(None, alias="from"), to: str = None, limit: int = 200):
    q = "SELECT id,type,amount,currency,category,note,created_at FROM movements WHERE 1=1"
    params = {}
    if _from:
        q += " AND created_at >= :f"; params["f"] = _from
    if to:
        q += " AND created_at < :t"; params["t"] = to
    q += " ORDER BY created_at DESC LIMIT :lim"
    params["lim"] = limit
    with engine.begin() as conn:
        rows = conn.execute(text(q), params).mappings().all()
        totals = conn.execute(text("""
            SELECT
              COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
              COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
            FROM movements
            WHERE 1=1
              AND (:f IS NULL OR created_at >= :f)
              AND (:t IS NULL OR created_at < :t)
        """), {"f": _from, "t": to}).mappings().first()
    total_in  = float(totals["total_in"]) if totals and totals["total_in"]  is not None else 0.0
    total_out = float(totals["total_out"]) if totals and totals["total_out"] is not None else 0.0
    return {"items":[dict(r) for r in rows], "totals": {"total_in": total_in, "total_out": total_out, "net": total_in - total_out}}

@APP.get("/summary")
def summary(days: int = 30):
    since = datetime.utcnow() - timedelta(days=days)
    with engine.begin() as conn:
        totals = conn.execute(text("""
            SELECT
              COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
              COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
            FROM movements
            WHERE created_at >= :since
        """), {"since": since}).mappings().first()
    total_in  = float(totals["total_in"]) if totals and totals["total_in"]  is not None else 0.0
    total_out = float(totals["total_out"]) if totals and totals["total_out"] is not None else 0.0
    return {"period_days": days, "entrate": total_in, "uscite": total_out, "saldo": total_in - total_out}

# =========================
# Debug (puoi tenerli finché vuoi)
# =========================
@APP.get("/debug")
def debug():
    db_ok = False
    db_err = ""
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_ok = False
        db_err = str(e)
    return {
        "has_openai_key": bool(OPENAI_API_KEY),
        "db_ok": db_ok,
        "db_error": db_err[:400],
        "allowed_origins": origins,
        "api_key_set": bool(API_KEY_APP),
    }

@APP.post("/dbwrite")
def dbwrite():
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("INSERT INTO messages (content, reply, created_at) VALUES ('SMOKE','OK',:t) RETURNING id"),
                {"t": datetime.utcnow()}
            ).mappings().first()
        return {"ok": True, "id": row["id"]}
    except Exception as e:
@APP.get("/echo")
def echo(request: Request):
    # NON stampa la tua API key reale; mostra solo se l'header è presente e quanti char ha
    hdr = request.headers.get("x-api-key")
    return {
        "x_api_key_present": bool(hdr),
        "x_api_key_len": len(hdr) if hdr else 0,
        "path": str(request.url.path)
    }

@APP.get("/debug")
def debug():
    # controlla DB e chiave OpenAI senza alzare 500
    db_ok = False
    db_err = ""
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_ok = False
        db_err = str(e)
    return {
        "has_openai_key": bool(OPENAI_API_KEY),
        "db_ok": db_ok,
        "db_error": db_err[:400],
        "allowed_origins": origins,
        "api_key_set": bool(API_KEY_APP),
        "api_key_length": len(API_KEY_APP) if API_KEY_APP else 0
    }

        return {"ok": False, "error": str(e)}

