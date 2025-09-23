from fastapi import FastAPI
APP = FastAPI()

@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def health():
    return "ok"

