import os
from fastapi import FastAPI, Request

APP = FastAPI()

META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")
print(">>> TOKEN LETTO DA RENDER:", repr(META_VERIFY_TOKEN), flush=True)

@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

# --- GET /webhook: verifica di Meta ---
@APP.get("/webhook")
def verify(hub_mode: str | None = None,
          hub_challenge: str | None = None,
          hub_verify_token: str | None = None):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN and hub_challenge:
        # Meta si aspetta il challenge puro
        try:
            return int(hub_challenge)
        except:
            return hub_challenge
    return "forbidden"

# --- POST /webhook: per ora logga il body e risponde ok ---
@APP.post("/webhook")
async def incoming(req: Request):
    body = await req.json()
    # log semplice su stdout (visibile nei Logs di Render)
    print(">>> WEBHOOK BODY:", body, flush=True)
    return {"ok": True}

