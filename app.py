import os
import time
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import JSONResponse, PlainTextResponse

# --- DB (psycopg 3) ---
import psycopg

# =========================
# Config & helpers
# =========================
def _clean(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip()
    # normalizza eventuali virgolette accidentali su sslmode
    s = s.replace('sslmode="require"', "sslmode=require")
    return s

DATABASE_URL = _clean(os.getenv("DATABASE_URL"))
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "flai-verify-123")
API_KEY_APP = os.getenv("API_KEY_APP", "flai_Chiasso13241")

# =========================
# App
# =========================
app = FastAPI()

# =========================
# DB connection
# =========================
def get_db():
    """
    Connessione veloce al DB.
    NB: la URL deve essere PULITA (senza \n finali).
    """
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    # Nel tuo caso la URL già contiene ?sslmode=require
    return psycopg.connect(DATABASE_URL)

# =========================
# Security middleware
# =========================
@app.middleware("http")
async def security_mw(request: Request, call_next):
    # Endpoints pubblici (no API key)
    open_paths = {"/", "/healthz", "/webhook"}
    if request.url.path not in open_paths:
        api_key = request.headers.get("X-API-Key")
        if api_key != API_KEY_APP:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        resp = await call_next(request)
        return resp
    except Exception as e:
        # fallback di sicurezza
        return JSONResponse({"error": "internal_error", "detail": str(e)}, status_code=500)

# =========================
# Endpoints base
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "flai-app", "ts": int(time.time())}

@app.get("/healthz")
def healthz():
    # Mini-ping DB per essere sicuri (non blocca in caso di errore)
    db_ok = True
    try:
        with get_db() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
    except Exception:
        db_ok = False
    return {"ok": True, "db": db_ok, "time": int(time.time())}

# =========================
# Messages (demo)
# =========================
@app.post("/messages")
def create_message(req: Dict[str, Any] = Body(...)):
    txt = (req.get("message") or "").strip()
    if not txt:
        return JSONResponse({"error": "empty_message"}, status_code=400)
    # Per ora echo: più avanti colleghiamo OpenAI e salvataggio DB messaggi
    return {"id": int(time.time()), "reply": f"Ho ricevuto: {txt}"}

# =========================
# Movements
# Schema usato: movements(id, type, amount, currency, category, note, voce, created_at)
# =========================
@app.post("/movements")
def create_movement(req: Dict[str, Any] = Body(...)):
    mtype = req.get("type")
    amount = req.get("amount")
    currency = (req.get("currency") or "CHF").upper()
    category = req.get("category")
    note = req.get("note")
    voce = req.get("voce") or "generale"

    if mtype not in ("in", "out"):
        return JSONResponse({"error": "invalid_type", "detail": "type must be 'in' or 'out'"},
                            status_code=422)
    try:
        amount = float(amount)
    except Exception:
        return JSONResponse({"error": "invalid_amount"}, status_code=422)

    try:
        with get_db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO movements(type, amount, currency, category, note, voce, created_at)
                VALUES (%s,%s,%s,%s,%s,%s, NOW())
                RETURNING id
                """,
                (mtype, amount, currency, category, note, voce),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
        return {"ok": True, "id": new_id}
    except Exception as e:
        return JSONResponse({"error": "db_failed_insert", "detail": str(e)}, status_code=500)

@app.get("/movements")
def list_movements(limit: int = 50):
    limit = max(1, min(limit, 200))
    try:
        with get_db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type, amount, currency, category, note, voce, created_at
                FROM movements
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r[0],
                    "type": r[1],
                    "amount": float(r[2]) if r[2] is not None else None,
                    "currency": r[3],
                    "category": r[4],
                    "note": r[5],
                    "voce": r[6],
                    "created_at": r[7].isoformat() if isinstance(r[7], datetime) else str(r[7]),
                }
            )
        return out
    except Exception as e:
        return JSONResponse({"error": "db_failed_query", "detail": str(e)}, status_code=500)

# =========================
# WhatsApp Webhook
# =========================
@app.get("/webhook")
def wa_verify(hub_mode: str = "", hub_challenge: str = "", hub_verify_token: str = ""):
    """
    Verifica di Meta: se token combacia, restituire ESATTAMENTE hub_challenge in chiaro.
    """
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "", media_type="text/plain")
    raise HTTPException(status_code=403, detail="verification_failed")

@app.post("/webhook")
def wa_inbound(payload: Dict[str, Any] = Body(...)):
    """
    Gestisce:
      - Test interni: {"message":"..."} -> echo
      - Meta WhatsApp (formato semplificato): estrae testo e ritorna preview
    """
    # Test interno nostro semplice
    if "message" in payload:
        text = str(payload.get("message") or "").strip()
        if not text:
            return JSONResponse({"error": "empty_message"}, status_code=400)
        return {"reply": f"Echo: {text}"}

    # Payload stile Meta (semplificato)
    try:
        entries = payload.get("entry") or []
        if not entries:
            return {"status": "ok"}
        changes = entries[0].get("changes") or []
        if not changes:
            return {"status": "ok"}
        value = changes[0].get("value") or {}
        msgs = value.get("messages") or []
        if not msgs:
            return {"status": "ok"}

        msg = msgs[0]
        from_wa = msg.get("from")
        text = ""
        if "text" in msg:
            text = (msg["text"].get("body") or "").strip()
        elif msg.get("type") == "interactive":
            interactive = msg.get("interactive") or {}
            text = interactive.get("title") or interactive.get("text") or "[interactive]"
        else:
            text = "[unsupported message type]"

        preview = f"Echo to {from_wa}: {text}"
        return {"status": "ok", "preview_reply": preview}
    except Exception as e:
        return JSONResponse({"error": "wa_parse_error", "detail": str(e)}, status_code=200)
