import os
from datetime import datetime

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openai import OpenAI

# ===== FastAPI =====
APP = FastAPI()

# CORS "aperto" per test (poi potrai restringere al tuo dominio)
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENV =====
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")      # es. flai-verify-123
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")         # la tua chiave OpenAI
DATABASE_URL      = (os.getenv("DATABASE_URL") or "").strip()

# Forza sslmode=require sul DB (Render/Postgres)
if DATABASE_URL and "sslmode=" not in DATABASE_URL:
    sep = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = DATABASE_URL + f"{sep}sslmode=require"

# ===== OpenAI =====
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== DB =====
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
with engine.begin() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS messages (
      id SERIAL PRIMARY KEY,
      content TEXT NOT NULL,
      reply   TEXT,
      created_at TIMESTAMP DEFAULT NOW()
    );
    """))

# ===== Schemi =====
class IncomingMessage(BaseModel):
    message: str

# ===== Routes =====
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
):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN and hub_challenge:
        try:
            return int(hub_challenge)
        except:
            return hub_challenge
    return "forbidden"

# Webhook semplice: AI + salva su DB
@APP.post("/webhook")
async def incoming(msg: IncomingMessage):
    # 1) risposta AI
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Sei un assistente aziendale. Rispondi chiaro, concreto e conciso."},
            {"role":"user","content": msg.message.strip()}
        ],
        temperature=0.2
    )
    reply_text = resp.choices[0].message.content.strip()

    # 2) salva su DB
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO messages (content, reply, created_at) VALUES (:c,:r,:t)"),
            {"c": msg.message, "r": reply_text, "t": datetime.utcnow()}
        )

    return {"reply": reply_text}

# Lista ultimi messaggi
@APP.get("/messages")
def list_messages(limit: int = 20):
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, content, reply, created_at FROM messages ORDER BY created_at DESC LIMIT :lim"),
            {"lim": limit}
        ).mappings().all()
    return {"items": [dict(r) for r in rows]}

@APP.get("/debug")
def debug():
    # controlla DB e chiave OpenAI senza far esplodere 500
    from sqlalchemy import text
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
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "db_ok": db_ok,
        "db_error": db_err[:400]
    }

