from fastapi import FastAPI

APP = FastAPI()

@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

# Verifica webhook (per Meta): se riceve hub.challenge, lo rimanda
@APP.get("/webhook")
def verify(hub_mode: str = None, hub_challenge: str = None, hub_verify_token: str = None):
    if hub_mode == "subscribe" and hub_challenge:
        try:
            return int(hub_challenge)
        except:
            return hub_challenge
    return "ok"

