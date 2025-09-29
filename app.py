import os
import json
import datetime
import pg8000
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# Connessione al DB
def get_db():
    conn = pg8000.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME"),
    )
    return conn

@app.get("/")
def home():
    return {"status": "ok", "message": "FLAI API attiva 🚀"}

# Movimenti (entrate/uscite)
@app.post("/movements")
async def add_movement(request: Request):
    data = await request.json()
    type_ = data.get("type")
    amount = data.get("amount")

    if type_ not in ["in", "out"] or amount is None:
        raise HTTPException(status_code=400, detail="Dati non validi")

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO movements (type, amount, created_at) VALUES (%s, %s, %s)",
        (type_, amount, datetime.datetime.utcnow())
    )
    conn.commit()
    cur.close()
    conn.close()

    return {"status": "success"}

# Analytics di base
@app.get("/analytics/overview")
def analytics(days: int = 7):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
          DATE(created_at) as d,
          COALESCE(SUM(CASE WHEN type='in'  THEN amount END), 0) as in_amt,
          COALESCE(SUM(CASE WHEN type='out' THEN amount END), 0) as out_amt
        FROM movements
        WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
        GROUP BY DATE(created_at)
        ORDER BY d ASC;
        """,
        (days,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    result = [
        {"date": str(r[0]), "in": float(r[1]), "out": float(r[2]), "net": float(r[1]) - float(r[2])}
        for r in rows
    ]

    return result
