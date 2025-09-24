import os
import json
from datetime import datetime, timedelta
from decimal import Decimal

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from openai import OpenAI

# =========================
#   FASTAPI + CORS
# =========================
APP = FastAPI()
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # per test: poi limita al tuo dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
#   ENV
# =========================
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
if DATABASE_URL and "sslmode=" not in DATABASE_URL:
    sep = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = DATABASE_URL + f"{sep}sslmode=require"

print(">>> HAS OPENAI KEY:", bool(OPENAI_API_KEY), flush=True)
print(">>> DATABASE_URL:", repr(DATABASE_URL), flush=True)

# =========================
#   OpenAI
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

EXTRACT_SYSTEM_PROMPT = (
    "Sei un assistente contabile. Dal messaggio dell'utente estrai le voci economiche in formato JSON, "
    "con schema: {\"movimenti\":[{\"tipo\":\"entrata|uscita\",\"voce\":\"string\",\"importo\":number,\"valuta\":\"CHF\",\"note\":\"string\"}]}.\n"
    "Regole:\n"
    "- Se non trovi movimenti, restituisci {\"movimenti\":[]}.\n"
    "- importo è numerico (punto come decimale).\n"
    "- valuta: CHF, salvo diversa indicazione.\n"
    "- tipo: 'entrata' per ricavi/incassi, 'uscita' per costi/spese.\n"
)

def extract_movements(text: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role":"user",   "content": text}
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        start = raw.find("{"); end = raw.rfind("}")
        raw_json = raw[start:end+1] if (start!=-1 and end!=-1 and end>start) else raw
        data = json.loads(raw_json)
        if not isinstance(data, dict) or "movimenti" not in data:
            return {"movimenti":[]}
        norm = []
        for m in data.get("movimenti", []):
            try:
                tipo = (m.get("tipo") or "").lower().strip()
                if tipo not in ("entrata","uscita"): continue
                voce = (m.get("voce") or "").strip() or "Voce"
                imp  = m.get("importo")
                imp  = float(imp) if isinstance(imp, (int,float,str)) else None
                if imp is None: continue
                valuta = (m.get("valuta") or "CHF").upper().strip()
                note = (m.get("note") or "").strip()
                norm.append({"tipo":tipo,"voce":voce,"importo":imp,"valuta":valuta,"note":note})
            except:
                continue
        return {"movimenti": norm}
    except Exception as e:
        print(">>> extract_movements error:", str(e), flush=True)
        return {"movimenti":[]}

# =========================
#   DB
# =========================
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
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS movements (
        id SERIAL PRIMARY KEY,
        message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
        type  VARCHAR(10) NOT NULL,
        voce  TEXT NOT NULL,
        importo NUMERIC(14,2) NOT NULL,
        valuta VARCHAR(8) NOT NULL DEFAULT 'CHF',
        note  TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """))

# =========================
#   ROUTES
# =========================
@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

@APP.get("/webhook")
def verify(
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

@APP.post("/webhook")
async def incoming(req: Request):
    body = await req.json()
    msg = (body.get("message") or "").strip()
    if not msg:
        return {"error":"no message"}

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Sei un assistente aziendale. Rispondi in modo chiaro e conciso."},
            {"role":"user","content": msg}
        ],
        temperature=0.2
    )
    reply = resp.choices[0].message.content.strip()

    extracted = extract_movements(msg)

    with engine.begin() as conn:
        row = conn.execute(
            text("INSERT INTO messages (content, reply, created_at) VALUES (:c,:r,:t) RETURNING id"),
            {"c": msg, "r": reply, "t": datetime.utcnow()}
        ).mappings().first()
        message_id = row["id"]

        for m in extracted.get("movimenti", []):
            conn.execute(
                text("""INSERT INTO movements
                        (message_id, type, voce, importo, valuta, note, created_at)
                        VALUES (:mid, :type, :voce, :imp, :val, :note, :t)"""),
                {
                    "mid":  message_id,
                    "type": m["tipo"],
                    "voce": m["voce"],
                    "imp":  Decimal(str(m["importo"])),
                    "val":  m.get("valuta","CHF"),
                    "note": m.get("note",""),
                    "t":    datetime.utcnow()
                }
            )

    return {"reply": reply, "parsed": extracted}

@APP.get("/messages")
def last_messages(limit: int = 20):
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, content, reply, created_at FROM messages ORDER BY created_at DESC LIMIT :lim"),
            {"lim": limit}
        ).mappings().all()
    return {"items": [dict(r) for r in rows]}

@APP.get("/movements")
def last_movements(limit: int = 50):
    with engine.begin() as conn:
        rows = conn.execute(
            text("""SELECT id, message_id, type, voce, importo, valuta, note, created_at
                    FROM movements
                    ORDER BY created_at DESC
                    LIMIT :lim"""),
            {"lim": limit}
        ).mappings().all()
    items = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("importo"), Decimal):
            d["importo"] = float(d["importo"])
        items.append(d)
    return {"items": items}

@APP.get("/summary")
def summary(days: int = 30):
    since = datetime.utcnow() - timedelta(days=days)
    with engine.begin() as conn:
        rows = conn.execute(
            text("""SELECT type, SUM(importo) AS tot
                    FROM movements
                    WHERE created_at >= :since
                    GROUP BY type"""),
            {"since": since}
        ).mappings().all()
    tot_entrate = 0.0
    tot_uscite  = 0.0
    for r in rows:
        t = r["type"]
        v = float(r["tot"]) if r["tot"] is not None else 0.0
        if t == "entrata": tot_entrate = v
        elif t == "uscita": tot_uscite = v
    saldo = tot_entrate - tot_uscite
    return {"period_days": days, "entrate": tot_entrate, "uscite": tot_uscite, "saldo": saldo}

