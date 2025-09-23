import os
from fastapi import FastAPI, Request, Query
from openai import OpenAI

APP = FastAPI()

# --- Config ---
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")
print(">>> TOKEN LETTO DA RENDER:", repr(META_VERIFY_TOKEN), flush=True)

# OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- ROUTES ---

@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

# --- GET /webhook: verifica iniziale di Meta ---
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

# --- POST /webhook: gestisce messaggi in ingresso ---
@APP.post("/webhook")
async def incoming(req: Request):
    body = await req.json()
    print(">>> WEBHOOK BODY:", body, flush=True)

    # Prendiamo il campo "message" (simuliamo WhatsApp/Meta)
    text = body.get("message", "").strip()

    if text:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Sei un assistente aziendale. Rispondi in modo chiaro e conciso."},
                    {"role": "user", "content": text}
                ],
                temperature=0.2
            )
            reply = resp.choices[0].message.content.strip()
            print(">>> RISPOSTA GPT:", reply, flush=True)
            return {"reply": reply}
        except Exception as e:
            print(">>> ERRORE GPT:", str(e), flush=True)
            return {"error": str(e)}

    return {"ok": True}

