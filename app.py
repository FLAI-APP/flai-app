import os
from datetime import datetime

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openai import OpenAI

# =========================
# FastAPI + CORS
# =========================
APP = FastAPI()
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # per test; poi limita al tuo dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ENV
# =========================
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")     # es: flai-verify-123
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")        # chiave OpenAI valida
DATABASE_URL      = (os.getenv("DATABASE_URL") or "").strip()

# forza sslmode=require se manca (Render Postgres lo richiede)
if DATABASE_URL and "sslmode=" not in DATABASE_URL:
    sep = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = DATABASE_URL + f"{sep}sslmode=require"

# =========================
# OpenAI
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# DB
# =========================
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
with engine.begin() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS messages (
      id SERIAL PRIMARY KEY,
      content   TEXT NOT NULL,
      reply     TEXT,
      created_at TIMESTAMP DEFAULT NOW()
    );
    """))

# =========================
# Schemi
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

# Verifica webhook (Meta: hub.*)
@APP.get("/webhook")
def verify(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    # accetta anche varianti senza punto (per test manuali)
    hub_mode_alt: str | None = Query(None, alias="hub_mode"),
    hub_challenge_alt: str | None = Query(None, alias="hub_challenge"),
    hub_verify_token_alt: str | None = Query(None, alias="hub_verify_token"),
):
    mode = hub_mode or hub_mode_alt
    challenge = hub_challenge or hub_challenge_alt
    token = hub_verify_token or hub_verify_token_alt
    if mode == "subscribe" and token == META_VERIFY_TOKEN and challenge:
        try:
            return int(challenge)
        except:
            return challenge
    return "forbidden"

# =========================
# Webhook: AI + salva su DB (versione stabile)
# =========================
@APP.post("/webhook")
async def incoming(msg: IncomingMessage):
    # 1) risposta AI (gestione errori esplicita)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"Sei un assistente aziendale. Rispondi chiaro, concreto e conciso."},
                {"role":"user","content": msg.message.strip()}
            ],
            temperature=0.2
        )
        reply_text = resp.choices[0].message.content.strip()
    except Exception as e:
        return {"error":"openai_failed", "detail": str(e)}

    # 2) salva su DB (gestione errori esplicita)
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO messages (content, reply, created_at) VALUES (:c,:r,:t)"),
                {"c": msg.message, "r": reply_text, "t": datetime.utcnow()}
            )
    except Exception as e:
        return {"error":"db_failed", "detail": str(e)}

    return {"reply": reply_text}

# =========================
# Lista messaggi (sicura)
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
# Diagnostica
# =========================
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
        "db_error": db_err[:400]
    }

@APP.post("/dbwrite")
def dbwrite():
    # smoke test: scrive una riga fissa nella tabella messages
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("INSERT INTO messages (content, reply, created_at) VALUES ('SMOKE','OK',:t) RETURNING id"),
                {"t": datetime.utcnow()}
            ).mappings().first()
        return {"ok": True, "id": row["id"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

