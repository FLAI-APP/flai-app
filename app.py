import os
import json
import psycopg2
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

# ---------------------------
# Configurazione OpenAI
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY non trovata nelle variabili d'ambiente")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Configurazione FastAPI
# ---------------------------
app = FastAPI(title="FLAI App", version="1.0.0")

# ---------------------------
# Connessione DB
# ---------------------------
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", 5432),
    )
    return conn

# ---------------------------
# Root endpoint
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "FLAI App attiva 🚀"}

# ---------------------------
# Movements endpoint
# ---------------------------
@app.get("/movements")
def get_movements(limit: int = 10):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT description, amount, date FROM movements ORDER BY date DESC LIMIT %s",
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        movements = [
            {"date": r[2].isoformat(), "description": r[0], "amount": float(r[1])}
            for r in rows
        ]
        return {"movements": movements}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Generate Summary endpoint
# ---------------------------
@app.post("/summaries/generate")
async def generate_summary(request: Request, days: int = 30):
    try:
        data = await request.json()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT description, amount, date FROM movements WHERE date >= NOW() - INTERVAL '%s days' ORDER BY date DESC",
            (days,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return {"summary": f"Nessun movimento trovato negli ultimi {days} giorni."}

        text_data = "\n".join([f"{r[2]} - {r[0]}: {r[1]} CHF" for r in rows])

        prompt = f"""
        Sei un assistente finanziario.
        Riassumi le seguenti transazioni degli ultimi {days} giorni in modo chiaro e utile:

        {text_data}
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sei un assistente finanziario esperto."},
                {"role": "user", "content": prompt}
            ]
        )

        summary = completion.choices[0].message.content
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Analytics endpoint
# ---------------------------
@app.get("/analytics")
def analytics():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT SUM(amount), AVG(amount) FROM movements")
        total, avg = cur.fetchone()
        cur.close()
        conn.close()

        return {
            "total": float(total) if total else 0,
            "average": float(avg) if avg else 0,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Webhook endpoint
# ---------------------------
@app.post("/webhook")
async def webhook(request: Request):
    try:
        payload = await request.json()
        print("Webhook ricevuto:", json.dumps(payload, indent=2))
        return JSONResponse(content={"status": "received", "data": payload})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
