import os
from fastapi import FastAPI, Request, Query

APP = FastAPI()

# Legge il token dall'env (Render → Environment)
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")
print(">>> TOKEN LETTO DA RENDER:", repr(META_VERIFY_TOKEN), flush=True)

@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

# --- GET /webhook: accetta sia hub.* (Meta) sia hub_* (test manuale) ---
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

# --- POST /webhook: per ora logga il body ---
@APP.post("/webhook")
async def incoming(req: Request):
    body = await req.json()
    print(">>> WEBHOOK BODY:", body, flush=True)
    return {"ok": True}

