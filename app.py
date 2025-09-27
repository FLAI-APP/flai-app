kimport os
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Request, Body, HTTPException, Query
from fastapi.responses import JSONResponse

# ========= CONFIG =========
DB_URL = os.getenv("DATABASE_URL", "")
API_KEY = os.getenv("API_KEY_APP", "")  # es. flai_Chiasso13241

if not DB_URL:
    raise RuntimeError("DATABASE_URL non impostata")
if not API_KEY:
    # non blocchiamo l'avvio, ma avvisiamo nei log
    print("⚠️  API_KEY_APP non impostata: imposta su Render → Environment")

# Connessione Postgres (Render vuole quasi sempre SSL)
def get_conn():
    return psycopg2.connect(DB_URL, sslmode="require", connect_timeout=10)

# ========= FASTAPI =========
APP = FastAPI(title="flai-app", version="1.0.0")

# --- Auth middleware: tutto protetto tranne / e /healthz
@APP.middleware("http")
async def auth(request: Request, call_next):
    if request.url.path not in ("/", "/healthz"):
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse({"error": "invalid api key"}, status_code=401)
    return await call_next(request)

# --- Ping base (aperto)
@APP.get("/")
def root():
    return {"ok": True, "service": "flai-app", "time": datetime.utcnow().isoformat()}

# --- Health DB (aperto)
@APP.get("/healthz")
def healthz():
    try:
        with get_conn() as c:
            with c.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ========= MOVEMENTS =========
@APP.post("/movements")
def create_movement(item: dict = Body(...)):
    """
    Body accettato:
    {
      "type": "in" | "out",
      "amount": 2500,
      "currency": "CHF",
      "category": "sales",
      "note": "incasso giorno",
      "voce": "generale"   (opzionale)
    }
    """
    try:
        t = (item.get("type") or "").lower()
        if t not in ("in", "out"):
            raise HTTPException(422, detail="type must be 'in' or 'out'")

        amt = item.get("amount")
        cur = item.get("currency", "CHF")
        cat = item.get("category") or "generale"
        note = item.get("note")
        voce = item.get("voce") or "generale"

        with get_conn() as conn:
            with conn.cursor() as c:
                c.execute(
                    """
                    INSERT INTO movements(type, amount, currency, category, note, voce, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s, NOW())
                    RETURNING id
                    """,
                    (t, amt, cur, cat, note, voce),
                )
                new_id = c.fetchone()[0]
        return {"status": "ok", "id": new_id}
    except Exception as e:
        return JSONResponse({"error": "db_failed_insert", "detail": str(e)}, status_code=500)

@APP.get("/movements")
def list_movements(fr: str = Query(None, alias="from"), to: str = None, limit: int = 200):
    """
    Query opzionali: ?from=YYYY-MM-DD&to=YYYY-MM-DD&limit=200
    """
    try:
        q = """
            SELECT id, type, amount, currency, category, note, voce, created_at
            FROM movements
            WHERE 1=1
        """
        params = []
        if fr:
            q += " AND created_at >= %s"; params.append(fr)
        if to:
            q += " AND created_at < %s"; params.append(to)
        q += " ORDER BY created_at DESC LIMIT %s"; params.append(limit)

        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as c:
                c.execute(q, params)
                items = c.fetchall()
            with conn.cursor(cursor_factory=RealDictCursor) as c2:
                c2.execute(
                    """
                    SELECT
                      COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
                      COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                    FROM movements
                    WHERE (%s IS NULL OR created_at >= %s)
                      AND (%s IS NULL OR created_at < %s)
                    """,
                    (fr, fr, to, to),
                )
                totals = c2.fetchone()
        return {"items": items, "totals": totals}
    except Exception as e:
        return JSONResponse({"error": "db_failed_query", "detail": str(e)}, status_code=500)

# ========= SUMMARIES =========
@APP.post("/summaries/generate")
def generate_summary(days: int = Query(30, ge=1, le=365)):
    """
    Riepilogo ultimo N giorni: tot in, tot out, netto.
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT 
                      COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
                      COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                    FROM movements
                    WHERE created_at >= NOW() - (%s || ' days')::interval
                    """,
                    (str(days),),
                )
                r = cur.fetchone() or {"total_in": 0, "total_out": 0}
        total_in = float(r["total_in"])
        total_out = float(r["total_out"])
        return {
            "window_days": days,
            "in": total_in,
            "out": total_out,
            "net": total_in - total_out,
        }
    except Exception as e:
        return JSONResponse({"error": "db_failed_summary", "detail": str(e)}, status_code=500)

