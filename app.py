import os
from datetime import datetime
from fastapi import FastAPI, Request, Query
from sqlalchemy import create_engine, text
from openai import OpenAI

APP = FastAPI()

# --- ENV ---
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

print(">>> META_VERIFY_TOKEN:", repr(META_VERIFY_TOKEN), flush=True)
print(">>> HAS OPENAI KEY:", bool(OPENAI_API_KEY), flush=True)
print(">>> HAS DATABASE_URL:", bool(DATABASE_URL), flush=True)

# --- OpenAI ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- DB engine + tabella ---
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

with engine.begin() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        reply   TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    )
    """))

@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

# --- GET /webhook: verifica Meta (accetta hub.* e hub_*) ---
@APP.get("/webhook")
def verify(
    request: Request,
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
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

# --- POST /webhook: chiede risposta a GPT e salva in DB ---
@APP.post("/webhook")
async def incoming(req: Request):
    body = await req.json()
    msg = (body.get("message") or "").strip()

    if not msg:
        return {"error": "no message"}

    # 1) GPT
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Sei un assistente aziendale. Rispondi in modo chiaro e conciso."},
            {"role":"user","content": msg}
        ],
        temperature=0.2
    )
    reply = resp.choices[0].message.content.strip()

    # 2) Salva in DB
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO messages (content, reply, created_at) VALUES (:c, :r, :t)"),
            {"c": msg, "r": reply, "t": datetime.utcnow()}
        )

    return {"reply": reply}

# --- GET /messages: ultimi 10 messaggi salvati ---
@APP.get("/messages")
def last_messages():
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, content, reply, created_at FROM messages ORDER BY created_at DESC LIMIT 10")
        ).mappings().all()
    return {"items": [dict(r) for r in rows]}

