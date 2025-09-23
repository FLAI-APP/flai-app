import os
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# === FastAPI ===
APP = FastAPI()

# === Database ===
DATABASE_URL = os.getenv("DATABASE_URL")  # lo prende da Render
if not DATABASE_URL:
    raise ValueError("DATABASE_URL non trovato. Aggiungilo su Render > Environment Variables.")

# crea engine con sslmode=require
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# === Modello tabella ===
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    reply = Column(Text, nullable=True)

# crea le tabelle se non esistono
Base.metadata.create_all(bind=engine)

# === Schemi Pydantic ===
class IncomingMessage(BaseModel):
    message: str

# === Rotte API ===

@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

# Verifica Meta (come prima)
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")

@APP.get("/webhook")
def verify(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return int(hub_challenge)
    return "forbidden"

# POST /webhook: salva nel DB
@APP.post("/webhook")
async def incoming(msg: IncomingMessage):
    db = SessionLocal()
    reply_text = f"Hai scritto: {msg.message}"  # qui più avanti mettiamo OpenAI
    new_msg = Message(text=msg.message, reply=reply_text)
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)
    db.close()
    return {"reply": reply_text}

# GET /messages: lista di messaggi salvati
@APP.get("/messages")
def list_messages():
    db = SessionLocal()
    msgs = db.query(Message).all()
    result = [{"id": m.id, "text": m.text, "reply": m.reply} for m in msgs]
    db.close()
    return result

