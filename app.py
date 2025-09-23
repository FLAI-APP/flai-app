import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from openai import OpenAI

# === FastAPI ===
APP = FastAPI()

# === ENV ===
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL non trovato.")

# Forza sslmode=require se manca
if "sslmode=" not in DATABASE_URL:
    sep = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL += f"{sep}sslmode=require"

# === DB ===
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"
    id    = Column(Integer, primary_key=True, index=True)
    text  = Column(Text, nullable=False)
    reply = Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)

# === OpenAI ===
client = OpenAI(api_key=OPENAI_API_KEY)

# === Schemi ===
class IncomingMessage(BaseModel):
    message: str

# === Routes ===
@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

# Verifica webhook WhatsApp (supporta hub.*)
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

# Webhook: chiama OpenAI e salva su DB
@APP.post("/webhook")
async def incoming(msg: IncomingMessage):
    db = SessionLocal()
    try:
        # 1) Chiedi risposta a GPT
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sei un assistente aziendale. Rispondi breve, chiaro e utile."},
                {"role": "user", "content": msg.message.strip()}
            ],
            temperature=0.2
        )
        reply_text = resp.choices[0].message.content.strip()

        # 2) Salva
        rec = Message(text=msg.message, reply=reply_text)
        db.add(rec)
        db.commit()
        db.refresh(rec)

        return {"id": rec.id, "reply": reply_text}
    finally:
        db.close()

@APP.get("/messages")
def list_messages():
    db = SessionLocal()
    try:
        msgs = db.query(Message).order_by(Message.id.desc()).limit(20).all()
        return [{"id": m.id, "text": m.text, "reply": m.reply} for m in msgs]
    finally:
        db.close()

