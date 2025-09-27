from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import asyncpg
import os

# Inizializza FastAPI
app = FastAPI()

# Connessione al database (Render usa DATABASE_URL)
DATABASE_URL = os.getenv("DATABASE_URL")

# Middleware per autenticazione
@app.middleware("http")
async def check_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    valid_key = os.getenv("API_KEY", "flai_Chiasso13241")
    if api_key != valid_key:
        return JSONResponse(status_code=401, content={"error": "invalid_api_key"})
    return await call_next(request)

# Endpoint di test base
@app.get("/")
async def root():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# Endpoint esempio: riepilogo movimenti ultimi N giorni
@app.get("/summaries/generate")
async def generate_summary(days: int = 7):
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        since = datetime.utcnow() - timedelta(days=days)
        rows = await conn.fetch(
            """
            SELECT type, SUM(amount) as total
            FROM movements
            WHERE created_at >= $1
            GROUP BY type
            """,
            since
        )
        await conn.close()

        summary = {row["type"]: float(row["total"]) for row in rows}
        return {"days": days, "summary": summary}

    except Exception as e:
        return {"error": "db_failed_summary", "detail": str(e)}

