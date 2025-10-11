import os
import time
import json
from typing import Optional
import datetime as dt
from io import StringIO
import csv
from fastapi.responses import StreamingResponse
from fastapi.responses import Response

# ==== DASHBOARD AUTH (HTTP Basic) ====
import secrets
from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def require_dashboard_auth(credentials: HTTPBasicCredentials = Depends(security)):
    user_ok = secrets.compare_digest(
        credentials.username, os.getenv("DASHBOARD_USER", "admin")
    )
    pwd_ok = secrets.compare_digest(
        credentials.password, os.getenv("DASHBOARD_PASSWORD", "")
    )
    if not (user_ok and pwd_ok):
        # Questo fa apparire il popup del browser
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True
# =====================================

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request, Body, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
APP = FastAPI(title="FLAI APP")

API_KEY_APP = os.getenv("API_KEY_APP", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "").strip()

# CORS (da env, virgole separate)
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
APP.add_middleware(
    CORSMiddleware,
    allow_origins=origins or [],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------

@APP.middleware("http")
async def security_and_rate_limit(request: Request, call_next):
    # normalizza il path (senza slash finale)
    path = request.url.path.rstrip("/") or "/"

    # Pagine/asset che il browser deve poter chiamare senza X-API-Key
    WHITELIST_PREFIXES = (
        "/",              # root -> redirect a /dashboard
        "/healthz",
        "/dashboard",     # include /dashboard/data e /dashboard/pdf
        "/static",
        "/favicon.ico",
        "/docs",
        "/openapi.json",
    )
    if any(path == p or path.startswith(p + "/") for p in WHITELIST_PREFIXES):
        return await call_next(request)

    expected = (os.getenv("API_KEY_APP") or "").strip()
    got = (request.headers.get("X-API-Key") or "").strip()
    if not expected or got != expected:
        raise HTTPException(status_code=401, detail="invalid api key")

    return await call_next(request)

# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    # psycopg3: row_factory = dict_row per dict
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def ensure_schema():
    # Crea tabelle minime se non esistono (NON rompe se già esistono)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
          id SERIAL PRIMARY KEY,
          content TEXT NOT NULL,
          reply   TEXT,
          created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """)
        # La tua tabella movements esiste già: non la tocco nella struttura.
        # Creo solo se manca del tutto con le colonne base che usiamo.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS movements (
          id SERIAL PRIMARY KEY,
          type VARCHAR(10) NOT NULL,
          amount NUMERIC(14,2) NOT NULL,
          currency VARCHAR(8) NOT NULL DEFAULT 'CHF',
          category VARCHAR(50),
          note TEXT,
          created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """)
        conn.commit()

@APP.on_event("startup")
def _startup():
    ensure_schema()

# -----------------------------------------------------------------------------
# Root + health
# -----------------------------------------------------------------------------
@APP.get("/")
async def root():
    return {"ok": True, "app": "flai-app"}

@APP.get("/healthz")
async def healthz():
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            row = cur.fetchone()
        return {"ok": True, "db": row["ok"] == 1}
    except Exception as e:
        return JSONResponse({"error": "db_bootstrap_failed", "detail": str(e)}, status_code=500)

# -----------------------------------------------------------------------------
# Chat demo (POST /messages) — echo + salva su DB
# -----------------------------------------------------------------------------
@APP.post("/messages")
async def create_message(payload: dict = Body(...)):
    text: str = (payload.get("message") or "").strip()
    if not text:
        raise HTTPException(422, detail="message required")

    reply = f"Hai scritto: {text}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages(content, reply) VALUES (%s, %s) RETURNING id",
                (text, reply),
            )
            new_id = cur.fetchone()["id"]
            conn.commit()
        return {"id": new_id, "reply": reply}
    except Exception as e:
        return JSONResponse({"error": "db_failed_insert_message", "detail": str(e)}, status_code=500)

# -----------------------------------------------------------------------------
# Movements (crea/lista)
# -----------------------------------------------------------------------------
@APP.post("/movements")
async def create_movement(item: dict = Body(...)):
    t = item.get("type")
    if t not in ("in", "out"):
        raise HTTPException(422, detail="type must be 'in' or 'out'")
    amt = item.get("amount")
    if amt is None:
        raise HTTPException(422, detail="amount required")

    cur = item.get("currency", "CHF")
    cat = item.get("category")
    note = item.get("note")

    try:
        with get_conn() as conn, conn.cursor() as c:
            c.execute(
                """
                INSERT INTO movements(type, amount, currency, category, note)
                VALUES (%s,%s,%s,%s,%s) RETURNING id
                """,
                (t, amt, cur, cat, note),
            )
            new_id = c.fetchone()["id"]
            conn.commit()
        return {"ok": True, "id": new_id}
    except Exception as e:
        return JSONResponse({"error": "db_failed_insert", "detail": str(e)}, status_code=500)

@APP.get("/movements")
async def list_movements(_from: Optional[str] = Query(None, alias="from"),
                         to: Optional[str] = None):
    try:
        q = """
            SELECT id, type, amount, currency, category, note, created_at
            FROM movements
            WHERE 1=1
            """
        params = []
        if _from:
            q += " AND created_at >= %s"
            params.append(_from)
        if to:
            q += " AND created_at < %s"
            params.append(to)
        q += " ORDER BY created_at DESC LIMIT 200"

        with get_conn() as conn, conn.cursor() as c:
            c.execute(q, params)
            rows = c.fetchall()
        return rows
    except Exception as e:
        return JSONResponse({"error": "db_failed_query", "detail": str(e)}, status_code=500)

# -----------------------------------------------------------------------------
# WhatsApp Webhook
# -----------------------------------------------------------------------------
# GET /webhook (verify) — usa alias per i nomi con il punto (hub.*)
@APP.get("/webhook")
async def whatsapp_verify(
    hub_mode: str = Query("", alias="hub.mode"),
    hub_challenge: str = Query("", alias="hub.challenge"),
    hub_verify_token: str = Query("", alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "", media_type="text/plain")
    raise HTTPException(status_code=403, detail="verification_failed")

# POST /webhook — demo: se payload ha {"message": "..."} risponde con echo
@APP.post("/webhook")
async def whatsapp_events(payload: dict = Body(...)):
    # Demo pathway (test manuali)
    if "message" in payload:
        text = str(payload["message"]).strip()
        reply = f"Echo: {text or '(vuoto)'}"
        return {"ok": True, "reply": reply}

    # Placeholder per formato Meta (silenzioso se non riconosciuto)
    try:
        entry = payload.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return {"status": "ok"}
        msg = messages[0]
        from_wa = msg.get("from")
        text = (msg.get("text", {}) or {}).get("body", "")
        # Qui potresti generare la risposta AI e inviarla con WhatsApp Cloud API.
        return {"status": "ok", "from": from_wa, "preview_text": text[:120]}
    except Exception:
        return {"status": "ok"}


# -----------------------------------------------------------------------------
# REPORT & EXPORT
# -----------------------------------------------------------------------------

def _parse_iso_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        return None

@APP.get("/reports/weekly")
async def report_weekly(days: int = 7, category: str | None = None):
    """
    Ritorna un report giornaliero degli ultimi `days` giorni:
    per ogni giorno: IN, OUT e NET. Se `category` è indicata, filtra per categoria.

    Nota: per evitare problemi di cast nei parametri SQL, calcoliamo in Python la data di inizio
    e la passiamo come DATE a Postgres. Così niente placeholder dentro INTERVAL.
    """
    from datetime import date, timedelta

    if days < 1:
        days = 1

    start_date = date.today() - timedelta(days=days - 1)

    try:
        with get_conn() as conn, conn.cursor() as cur:
            # 1) generiamo i giorni a partire da start_date fino a oggi (incluso)
            # 2) aggreghiamo movements per giorno (e opzionale categoria)
            # 3) left join per includere giorni senza movimenti
            if category:
                cur.execute(
                    """
                    WITH days AS (
                      SELECT generate_series(%s::date, CURRENT_DATE::date, INTERVAL '1 day')::date AS d
                    ),
                    agg AS (
                      SELECT
                        DATE(created_at) AS d,
                        COALESCE(SUM(CASE WHEN type='in'  THEN amount END), 0) AS in_amt,
                        COALESCE(SUM(CASE WHEN type='out' THEN amount END), 0) AS out_amt
                      FROM movements
                      WHERE created_at >= %s::date
                        AND category = %s
                      GROUP BY DATE(created_at)
                    )
                    SELECT
                      days.d                       AS date,
                      COALESCE(agg.in_amt, 0)      AS in,
                      COALESCE(agg.out_amt, 0)     AS out,
                      COALESCE(agg.in_amt, 0) - COALESCE(agg.out_amt, 0) AS net
                    FROM days
                    LEFT JOIN agg ON agg.d = days.d
                    ORDER BY days.d ASC
                    """,
                    (start_date, start_date, category),
                )
            else:
                cur.execute(
                    """
                    WITH days AS (
                      SELECT generate_series(%s::date, CURRENT_DATE::date, INTERVAL '1 day')::date AS d
                    ),
                    agg AS (
                      SELECT
                        DATE(created_at) AS d,
                        COALESCE(SUM(CASE WHEN type='in'  THEN amount END), 0) AS in_amt,
                        COALESCE(SUM(CASE WHEN type='out' THEN amount END), 0) AS out_amt
                      FROM movements
                      WHERE created_at >= %s::date
                      GROUP BY DATE(created_at)
                    )
                    SELECT
                      days.d                       AS date,
                      COALESCE(agg.in_amt, 0)      AS in,
                      COALESCE(agg.out_amt, 0)     AS out,
                      COALESCE(agg.in_amt, 0) - COALESCE(agg.out_amt, 0) AS net
                    FROM days
                    LEFT JOIN agg ON agg.d = days.d
                    ORDER BY days.d ASC
                    """,
                    (start_date, start_date),
                )

            rows = cur.fetchall()
            result = [
                {"date": str(r["date"]), "in": float(r["in"]), "out": float(r["out"]), "net": float(r["net"])}
                for r in rows
            ]
            return {"ok": True, "days": days, "category": category, "series": result}
    except Exception as e:
        return {"error": "db_failed_report", "detail": str(e)}

# ------------------------------------------------------------------
# REPORT: MONTHLY (totali per mese, con filtro opzionale di categoria)
# ------------------------------------------------------------------

from fastapi import Query  # (se l'hai già importato sopra, questa riga non dà fastidio)

@APP.get("/reports/monthly")
async def reports_monthly(
    months: int = Query(6, ge=1, le=24, description="Numero di mesi da oggi a ritroso"),
    category: str | None = None,
):
    """
    Riepilogo per mese degli ultimi N mesi (default 6).
    - Somma 'in', 'out' e 'net' per ciascun mese.
    - Filtro opzionale per categoria (es. ?category=sales).
    - Ordine cronologico crescente (dal mese più vecchio al più recente).
    """
    sql = """
        WITH months AS (
          SELECT (date_trunc('month', CURRENT_DATE) - (INTERVAL '1 month' * gs))::date AS m_start
          FROM generate_series(0, %s - 1) AS gs
        )
        SELECT
          to_char(m.m_start, 'YYYY-MM') AS month,
          COALESCE(SUM(CASE WHEN mv.type='in'  THEN mv.amount END), 0) AS in_amt,
          COALESCE(SUM(CASE WHEN mv.type='out' THEN mv.amount END), 0) AS out_amt
        FROM months m
        LEFT JOIN movements mv
          ON date_trunc('month', mv.created_at) = m.m_start
         AND (%s::text IS NULL OR mv.category = %s::text)
        GROUP BY m.m_start
        ORDER BY m.m_start ASC;
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            # NB: months è usato solo nel generate_series; category viene passato due volte
            cur.execute(sql, (months, category, category))
            rows = cur.fetchall()

        series = []
        for r in rows:
            month, in_amt, out_amt = r["month"], float(r["in_amt"]), float(r["out_amt"])
            series.append({
                "month": month,
                "in": in_amt,
                "out": out_amt,
                "net": round(in_amt - out_amt, 2),
            })

        return {
            "ok": True,
            "months": months,
            "category": category,
            "series": series
        }
    except Exception as e:
        return {"error": "db_failed_report_monthly", "detail": str(e)}


@APP.get("/export/movements.csv")
async def export_movements_csv(
    _from: str | None = Query(None, alias="from"),
    to: str | None = None,
    category: str | None = None,
):
    """
    Esporta i movimenti in CSV (scaricabile).
    Filtri: from (YYYY-MM-DD), to (YYYY-MM-DD), category.
    """
    d_from = _parse_iso_date(_from)
    d_to = _parse_iso_date(to)

    try:
        with get_conn() as conn, conn.cursor() as c:
            q = """
                SELECT id, type, amount, currency, category, note, created_at
                FROM movements
                WHERE 1=1
            """
            params: list = []
            if d_from:
                q += " AND created_at::date >= %s"
                params.append(d_from)
            if d_to:
                q += " AND created_at::date <= %s"
                params.append(d_to)
            if category:
                q += " AND category = %s"
                params.append(category)
            q += " ORDER BY created_at DESC"
            c.execute(q, params)
            rows = c.fetchall()

        # Genera CSV in memoria
        buf = StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "type", "amount", "currency", "category", "note", "created_at"])  # header
        for r in rows:
            writer.writerow([
                r["id"],
                r["type"],
                r["amount"],
                r["currency"],
                r["category"] or "",
                (r["note"] or "").replace("\n", " ").strip(),
                r["created_at"].isoformat(sep=" ", timespec="seconds") if r.get("created_at") else "",
            ])
             buf.seek(0)

            filename = "movements_export.csv"
            headers = {
                "Content-Disposition": f'attachment; filename="{filename}"'
        }
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)

    except Exception as e:
        return JSONResponse({"error": "db_failed_export", "detail": str(e)}, status_code=500)

# ================================
# REPORT: YEARLY & CUSTOM
# ================================
import datetime as dt
from fastapi import Query

def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

@APP.get("/reports/yearly")
async def report_yearly(
    years: int = Query(5, ge=1, le=50, description="Quanti anni (incluso l'anno corrente)"),
    category: str | None = Query(None, description="Filtra per categoria (opzionale)"),
):
    """
    Serie annuale (ultimi 'years' anni fino all'anno corrente).
    Output: [{year, in, out, net}, ...]
    """
        from reportlab.lib.pagesizes import A4try:
        years = _clamp(years, 1, 50)
        with get_conn() as conn, conn.cursor() as c:
            # Serie di anni (date al 1° gennaio) e aggregati per anno
            q = """
                WITH years AS (
                  SELECT generate_series(
                    date_trunc('year', CURRENT_DATE) - ((%s::int - 1) * INTERVAL '1 year'),
                    date_trunc('year', CURRENT_DATE),
                    INTERVAL '1 year'
                  )::date AS y
                ),
                agg AS (
                  SELECT
                    date_trunc('year', created_at)::date AS y,
                    COALESCE(SUM(CASE WHEN type='in'  THEN amount END), 0) AS in_amt,
                    COALESCE(SUM(CASE WHEN type='out' THEN amount END), 0) AS out_amt
                  FROM movements
                  WHERE created_at >= date_trunc('year', CURRENT_DATE) - ((%s::int - 1) * INTERVAL '1 year')
                    AND created_at <  date_trunc('year', CURRENT_DATE) + INTERVAL '1 year'
                    {cat_filter}
                  GROUP BY date_trunc('year', created_at)::date
                )
                SELECT
                  EXTRACT(YEAR FROM years.y)::int AS year,
                  COALESCE(agg.in_amt,0)  AS in,
                  COALESCE(agg.out_amt,0) AS out,
                  COALESCE(agg.in_amt,0) - COALESCE(agg.out_amt,0) AS net
                FROM years
                LEFT JOIN agg ON agg.y = years.y
                ORDER BY years.y ASC
            """
            params: list = [years, years]
            cat_filter = ""
            if category:
                cat_filter = "AND category = %s"
                params.append(category)
            q = q.format(cat_filter=cat_filter)

            c.execute(q, params)
            rows = c.fetchall()

        series = [
            {"year": r["year"], "in": float(r["in"]), "out": float(r["out"]), "net": float(r["net"])}
            for r in rows
        ]
        return {"ok": True, "years": years, "category": category, "series": series}
    except Exception as e:
        return {"error": "db_failed_report_yearly", "detail": str(e)}

@APP.get("/reports/custom")
async def report_custom(
    from_: str = Query(..., alias="from", description="Data inizio (YYYY-MM-DD)"),
    to:   str = Query(..., description="Data fine inclusa (YYYY-MM-DD)"),
    category: str | None = Query(None, description="Filtra per categoria (opzionale)"),
):
    """
    Totali su un range personalizzato [from, to].
    Output: { totals: {in, out, net} }
    """
    # Parse date sicuro
    try:
        d_from = dt.date.fromisoformat(from_)
        d_to   = dt.date.fromisoformat(to)
    except Exception:
        return {"error": "bad_dates", "detail": "Usa formato YYYY-MM-DD per 'from' e 'to'."}
    if d_from > d_to:
        return {"error": "bad_range", "detail": "'from' deve essere <= 'to'."}

    try:
        with get_conn() as conn, conn.cursor() as c:
            q = """
                SELECT
                  COALESCE(SUM(CASE WHEN type='in'  THEN amount END), 0) AS in_amt,
                  COALESCE(SUM(CASE WHEN type='out' THEN amount END), 0) AS out_amt
                FROM movements
                WHERE created_at::date BETWEEN %s AND %s
                {cat_filter}
            """
            params: list = [d_from, d_to]
            cat_filter = ""
            if category:
                cat_filter = "AND category = %s"
                params.append(category)
            q = q.format(cat_filter=cat_filter)

            c.execute(q, params)
            r = c.fetchone()

        in_amt  = float(r["in_amt"])
        out_amt = float(r["out_amt"])
        return {
            "ok": True,
            "from": d_from.isoformat(),
            "to": d_to.isoformat(),
            "category": category,
            "totals": {"in": in_amt, "out": out_amt, "net": in_amt - out_amt},
        }
    except Exception as e:
        return {"error": "db_failed_report_custom", "detail": str(e)}

# ================================
# REPORT PDF (range personalizzato)
# ================================
from io import BytesIO

@APP.get("/reports/pdf")
async def report_pdf(
    from_: str = Query(..., alias="from", description="Data inizio (YYYY-MM-DD)"),
    to:   str = Query(..., description="Data fine inclusa (YYYY-MM-DD)"),
    category: str | None = Query(None, description="Categoria opzionale"),
    limit_items: int = Query(50, ge=0, le=500, description="Quanti movimenti mostrare nel PDF (max 500)")
):
    """
    Genera un PDF con:
      - periodo richiesto
      - totali (in/out/net)
      - tabellina ultimi N movimenti del periodo (max 500)
    """
    # 1) parse date sicuro
    try:
        d_from = dt.date.fromisoformat(from_)
        d_to   = dt.date.fromisoformat(to)
    except Exception:
        return {"error": "bad_dates", "detail": "Usa YYYY-MM-DD per 'from' e 'to'."}
    if d_from > d_to:
        return {"error": "bad_range", "detail": "'from' deve essere <= 'to'."}

    # 2) leggi dati dal DB
    try:
        with get_conn() as conn, conn.cursor() as c:
            # totali
            q_tot = """
                SELECT
                  COALESCE(SUM(CASE WHEN type='in'  THEN amount END), 0) AS in_amt,
                  COALESCE(SUM(CASE WHEN type='out' THEN amount END), 0) AS out_amt
                FROM movements
                WHERE created_at::date BETWEEN %s AND %s
                {cat_filter}
            """
            params = [d_from, d_to]
            cat_filter = ""
            if category:
                cat_filter = "AND category = %s"
                params.append(category)
            q_tot = q_tot.format(cat_filter=cat_filter)
            c.execute(q_tot, params)
            r = c.fetchone()
            in_amt  = float(r["in_amt"])
            out_amt = float(r["out_amt"])
            net_amt = in_amt - out_amt

            # ultimi N movimenti
            q_mov = f"""
                SELECT id, type, amount, currency, category, note, created_at
                FROM movements
                WHERE created_at::date BETWEEN %s AND %s
                {cat_filter}
                ORDER BY created_at DESC
                LIMIT %s
            """.format(cat_filter=cat_filter)
            params_mov = [d_from, d_to] + ([category] if category else []) + [limit_items]
            c.execute(q_mov, params_mov)
            movs = c.fetchall()
    except Exception as e:
        return {"error": "db_failed_report_pdf", "detail": str(e)}

    # 3) genera PDF (ReportLab)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, title="FLAI - Report")
        styles = getSampleStyleSheet()
        story = []

        title = f"FLAI – Report {d_from.isoformat()} → {d_to.isoformat()}"
        if category:
            title += f"  (categoria: {category})"
        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 8))

        story.append(Paragraph(
            f"<b>Totali</b>: Entrate = <b>{in_amt:,.2f}</b>  |  Uscite = <b>{out_amt:,.2f}</b>  |  Netto = <b>{net_amt:,.2f}</b>",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))

        # tabella movimenti (se presenti)
        data = [["ID", "Tipo", "Importo", "Valuta", "Categoria", "Nota", "Data/Ora"]]
        for m in movs:
            data.append([
                m["id"],
                m["type"],
                f"{float(m['amount']):,.2f}",
                m["currency"],
                m["category"] or "",
                (m["note"] or "")[:40],
                m["created_at"].strftime("%Y-%m-%d %H:%M"),
            ])
        if len(data) > 1:
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("TEXTCOLOR", (0,0), (-1,0), colors.black),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("ALIGN", (2,1), (2,-1), "RIGHT"),
            ]))
            story.append(Paragraph(f"Movimenti mostrati: {len(data)-1}", styles["Heading4"]))
            story.append(t)
        else:
            story.append(Paragraph("Nessun movimento nel periodo selezionato.", styles["Italic"]))

        doc.build(story)
        pdf_bytes = buf.getvalue()
        buf.close()

        fname = f"flai-report_{d_from.isoformat()}_{d_to.isoformat()}"
        if category:
            fname += f"_{category}"
        fname += ".pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="{}"'.format(fname)}
        )
    except Exception as e:
        return {"error": "pdf_failed", "detail": str(e)}

# === EMAIL: SENDER + REPORT PDF (definitivo) ==================================
from typing import Optional, List, Tuple
from pydantic import BaseModel, EmailStr
from fastapi import Query, HTTPException
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import io
import datetime as dt

# PDF (usiamo reportlab e testo nero esplicito)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import black

# === Config mail dalle ENV (già impostate su Render)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USERNAME or "noreply@example.com")
MAIL_TO_DEFAULT = os.getenv("MAIL_TO", "")

def _send_email(
    to_addr: str,
    subject: str,
    body_text: str,
    attachments: Optional[List[Tuple[str, bytes, str]]] = None,  # (filename, content, mimetype)
) -> None:
    if not (SMTP_HOST and SMTP_PORT and SMTP_USERNAME and SMTP_PASSWORD):
        raise HTTPException(status_code=500, detail="smtp_not_configured")

    msg = MIMEMultipart()
    msg["From"] = MAIL_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain", "utf-8"))

    for (fname, content, mimetype) in (attachments or []):
        part = MIMEApplication(content, _subtype=(mimetype.split("/")[-1] if "/" in mimetype else "octet-stream"))
        part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
        msg.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.ehlo()
        s.starttls(context=context)
        s.login(SMTP_USERNAME, SMTP_PASSWORD)
        s.send_message(msg)

# --- utilità DB (riusa la tua get_conn esistente) -----------------------------
def _fetch_movements_between(d_from: dt.date, d_to: dt.date, category: Optional[str]) -> List[dict]:
    q = """
        SELECT type, amount, currency, category, coalesce(note,'') as note, created_at
        FROM movements
        WHERE created_at::date BETWEEN %s AND %s
    """
    params: List = [d_from, d_to]
    if category:
        q += " AND category = %s"
        params.append(category)
    q += " ORDER BY created_at ASC, id ASC"

    with get_conn() as conn, conn.cursor() as c:
        c.execute(q, params)
        rows = c.fetchall()

    # rows è una lista di dict (row_factory=dict_row nel tuo get_conn)
    return rows

def _make_pdf_report_bytes(
    rows: list[dict],
    d_from: dt.date | None,
    d_to: dt.date | None,
    title_suffix: str = "",
) -> bytes:
    """
    Genera un PDF identico per Dashboard e Email.
    rows: lista di dict con chiavi: id, type, amount, currency, category, note, created_at (datetime)
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.lib.colors import black

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    width, height = A4
    margin = 15 * mm
    x = margin
    y = height - margin

    # ---------- Header ----------
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 16)
    title = "FLAI – Report"
    if title_suffix:
        title += f" · {title_suffix}"
    if d_from or d_to:
        df = d_from.isoformat() if d_from else "…"
        dt_ = d_to.isoformat() if d_to else "…"
        title += f" ({df} → {dt_})"
    c.drawString(x, y, title)
    y -= 10 * mm

    # ---------- Totali ----------
    from decimal import Decimal
    s_in = Decimal("0")
    s_out = Decimal("0")
    currency = ""
    for r in rows:
        amt = Decimal(str(r["amount"]))
        if r["type"] == "in":
            s_in += amt
        else:
            s_out += amt
        if not currency:
            currency = r.get("currency", "")

    net = s_in - s_out
    c.setFont("Helvetica-Bold", 11)
    c.drawString(
        x,
        y,
        f"Entrate: {s_in:,.2f} {currency}   •   Uscite: {s_out:,.2f} {currency}   •   Netto: {net:,.2f} {currency}".replace(
            ",", "X"
        ).replace(".", ",").replace("X", ".")
    )
    y -= 8 * mm

    # ---------- Intestazioni tabella ----------
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 0 * mm, y, "ID")
    c.drawString(x + 14 * mm, y, "TP")
    c.drawString(x + 26 * mm, y, "AMOUNT")
    c.drawString(x + 55 * mm, y, "CUR")
    c.drawString(x + 70 * mm, y, "CATEGORY")
    c.drawString(x + 120 * mm, y, "NOTE")
    c.drawString(x + 170 * mm, y, "CREATED AT")
    y -= 5.5 * mm

    c.setFont("Helvetica", 10)

    def fmt_money(v: Decimal) -> str:
        return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def fmt_date(dv) -> str:
        try:
            return dv.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(dv)

    rows_per_page = int((y - margin) // (5.5 * mm))
    i = 0
    for r in rows:
        if i and i % rows_per_page == 0:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x + 0 * mm, y, "ID")
            c.drawString(x + 14 * mm, y, "TP")
            c.drawString(x + 26 * mm, y, "AMOUNT")
            c.drawString(x + 55 * mm, y, "CUR")
            c.drawString(x + 70 * mm, y, "CATEGORY")
            c.drawString(x + 120 * mm, y, "NOTE")
            c.drawString(x + 170 * mm, y, "CREATED AT")
            y -= 5.5 * mm
            c.setFont("Helvetica", 10)

        amt = Decimal(str(r["amount"]))
        c.drawString(x + 0 * mm, y, str(r["id"]))
        c.drawString(x + 14 * mm, y, r.get("type", ""))
        c.drawRightString(x + 52 * mm, y, fmt_money(amt))
        c.drawString(x + 55 * mm, y, r.get("currency", ""))
        c.drawString(x + 70 * mm, y, (r.get("category") or "")[:46])
        c.drawString(x + 120 * mm, y, (r.get("note") or "")[:46])
        c.drawString(x + 170 * mm, y, fmt_date(r.get("created_at")))
        y -= 5.5 * mm
        i += 1

    if not rows:
        c.drawString(x, y, "Nessun movimento nel periodo selezionato.")

    c.save()
    return buf.getvalue()

# ---------- Schemi richieste ----------
class EmailTestPayload(BaseModel):
    to: EmailStr
    subject: str
    body: str

class ReportEmailPayload(BaseModel):
    to: Optional[EmailStr] = None  # se mancante usa MAIL_TO_DEFAULT

# ---------- Endpoint: email di prova (solo testo) ----------
@APP.post("/email/test")
async def email_test(payload: EmailTestPayload):
    _send_email(payload.to, payload.subject, payload.body)
    return {"ok": True, "to": payload.to, "subject": payload.subject}

# ---------- Endpoint: email con PDF (parametri OBBLIGATORI) ----------
@APP.post("/email/report")
async def email_report(
    payload: ReportEmailPayload,
    from_date: dt.date = Query(..., alias="from"),
    to_date: dt.date = Query(..., alias="to"),
    category: Optional[str] = None,
):
    """
    Invia un PDF con il report del periodo. Se from/to mancano → 422 (automatico).
    """
    to_addr = (payload.to or MAIL_TO_DEFAULT or SMTP_USERNAME)
    if not to_addr:
        raise HTTPException(status_code=400, detail="missing_to_address")

    pdf_bytes = _make_pdf_report_bytes(from_date, to_date, category)
    fname = f"flai-report_{from_date.isoformat()}_{to_date.isoformat()}"
    if category:
        fname += f"_{category}"
    fname += ".pdf"

    subject = f"FLAI – Report PDF ({from_date} → {to_date})"
    if category:
        subject += f" – {category}"

    body = "In allegato trovi il PDF dei movimenti."
    _send_email(to_addr, subject, body, attachments=[(fname, pdf_bytes, "application/pdf")])
    return {"ok": True, "to": to_addr, "filename": fname, "bytes": len(pdf_bytes)}
# === FINE EMAIL ===============================================================


# === DASHBOARD (HTML + API + PDF) =============================================

from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from fastapi import Request, HTTPException, Query
import os, io
import datetime as dt
from decimal import Decimal

BRAND_BLUE = os.getenv("BRAND_BLUE", "#000A22")
ACCENT_GOLD = os.getenv("ACCENT_GOLD", "#AA8F15")
LOGO_URL    = (os.getenv("LOGO_URL", "") or "").strip()
GITHUB_REPO = (os.getenv("GITHUB_REPO", "FLAI-APP/flai-app") or "").strip("/")

# Se LOGO_URL è relativo (es. refs/heads/main/...), trasformalo in raw GitHub
if LOGO_URL and not LOGO_URL.startswith(("http://", "https://")):
    LOGO_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{LOGO_URL.lstrip('/')}"

def _iso_or_none(s: str | None) -> dt.date | None:
    if not s: return None
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        return None

def _like(s: str | None) -> str | None:
    return f"%{s.strip()}%" if s and s.strip() else None

def _fetch_dashboard_rows(conn, d_from, d_to, typ, q):
    sql = """
        SELECT id, type, amount::numeric(14,2) AS amount, currency,
               COALESCE(category,'') AS category, COALESCE(note,'') AS note,
               created_at
        FROM movements
        WHERE 1=1
    """
    params: list = []
    if d_from:
        sql += " AND created_at::date >= %s"; params.append(d_from)
    if d_to:
        sql += " AND created_at::date <= %s"; params.append(d_to)
    if typ in ("in","out"):
        sql += " AND type = %s"; params.append(typ)
    if q:
        like = _like(q)
        sql += """ AND (
            CAST(id AS TEXT) ILIKE %s OR type ILIKE %s OR
            CAST(amount AS TEXT) ILIKE %s OR currency ILIKE %s OR
            category ILIKE %s OR note ILIKE %s OR
            TO_CHAR(created_at,'YYYY-MM-DD HH24:MI') ILIKE %s
        )"""
        params += [like]*7
    sql += " ORDER BY created_at DESC, id DESC"   # nessun LIMIT
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

@APP.get("/", include_in_schema=False)  # type: ignore
async def root():
    return RedirectResponse("/dashboard")

@APP.get("/dashboard", response_class=HTMLResponse)  # type: ignore
async def dashboard(request: Request) -> HTMLResponse:
    html = f"""<!doctype html>
<html lang="it"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FLAI · Dashboard</title>
<link rel="preconnect" href="https://cdn.jsdelivr.net"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg:#0b0f14; --panel:#1b2129; --panel2:#232b35; --text:#e6e7ea; --muted:#9aa4b2;
  --brand:{BRAND_BLUE}; --accent:{ACCENT_GOLD};
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.35 -apple-system,Segoe UI,Roboto,system-ui,sans-serif}}
.header{{background:var(--brand);padding:10px 16px;display:flex;align-items:center;gap:12px}}
.logo{{width:44px;height:44px;border-radius:8px;overflow:hidden;position:relative}}
.logo img{{width:100%;height:100%;object-fit:cover;filter:brightness(0.80) saturate(0.95)}}
.logo::after{{content:"";position:absolute;inset:0;box-shadow:inset 0 0 24px 12px rgba(0,0,0,.65);
  outline:1px solid rgba(0,10,34,.25);border-radius:8px;pointer-events:none}}
.title{{font-weight:700;letter-spacing:.3px}}
.wrap{{max-width:1280px;margin:18px auto;padding:0 16px}}

.filters{{display:flex;align-items:center;gap:10px;flex-wrap:nowrap;overflow:auto}}
.filters label{{color:var(--muted);font-size:12px;margin-right:6px;white-space:nowrap}}
input[type=date],select,input[type=text]{{background:var(--panel);color:var(--text);
  border:1px solid #324158;border-radius:12px;padding:10px 12px}}
select.pill{{appearance:none;padding-right:28px;border-radius:12px;background:var(--panel);color:#cbd3df}}
input#q{{width:360px;max-width:360px;flex:0 0 360px}}
.btn{{background:var(--accent);color:#111;border:0;border-radius:12px;padding:10px 12px;
  font-weight:700;cursor:pointer;white-space:nowrap}}
.btn:active{{transform:translateY(1px)}}
.btn-muted{{background:#2e3643;color:#cbd3df}}

.stats{{display:flex;gap:14px;margin:14px 0 16px}}
.card{{flex:1;background:var(--panel2);border:1px solid #2f3b4c;border-radius:12px;padding:18px}}
.card h4{{margin:0 0 6px;color:#b9c2cf;font-weight:600;font-size:12px;letter-spacing:.5px}}
.card .v{{font-size:20px;font-weight:800}}

.table{{background:var(--panel2);border:1px solid #2f3b4c;border-radius:12px;overflow:auto;margin-bottom:18px}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:12px 14px;text-align:left;border-bottom:1px solid #2b3545;white-space:nowrap}}
thead th{{color:#b9c2cf;font-size:12px;letter-spacing:.4px}}
tbody tr:hover{{background:#1f2732}}
th.col-id,td.col-id{{width:60px}}
th.col-type,td.col-type{{width:80px}}
th.col-amt,td.col-amt{{width:120px;text-align:right}}
th.col-cur,td.col-cur{{width:80px}}
th.col-cat,td.col-cat{{width:180px}}
th.col-note,td.col-note{{width:260px;overflow:hidden;text-overflow:ellipsis}}
th.col-date,td.col-date{{width:170px}}

.table-head{{display:flex;justify-content:space-between;align-items:center;margin:6px 2px 10px}}
.table-head .right{{display:flex;gap:8px}}

.charts{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}}
.chart-card{{background:var(--panel2);border:1px solid #2f3b4c;border-radius:12px;padding:12px}}
.chart-title{{margin:0 0 8px;color:#b9c2cf;font-weight:600;font-size:12px;letter-spacing:.5px}}
.chart-wrap{{height:320px}}
@media (max-width: 980px){{ .charts{{grid-template-columns:1fr}} .chart-wrap{{height:260px}} }}
</style></head>
<body>
  <div class="header">
    <div class="logo">{('<img src="{LOGO_URL}" alt="logo"/>') if LOGO_URL else ''}</div>
    <div class="title">FLAI Dashboard</div>
  </div>

  <div class="wrap">
    <div class="filters">
      <label>Dal</label><input id="from" type="date">
      <label>Al</label><input id="to" type="date">
      <label>Tipo</label>
      <select id="type" class="pill">
        <option value="">Tutti</option>
        <option value="in">Entrate</option>
        <option value="out">Uscite</option>
      </select>
      <label>Barra di ricerca</label>
      <input id="q" type="text" placeholder="Barra di ricerca (id, tipo, amount, note, category)">
      <button id="apply" class="btn">Applica filtri</button>
      <button id="pdf" class="btn">Scarica PDF</button>
    </div>

    <div class="stats">
      <div class="card"><h4>ENTRATE</h4><div class="v" id="sum_in">0.00 CHF</div></div>
      <div class="card"><h4>USCITE</h4><div class="v" id="sum_out">0.00 CHF</div></div>
      <div class="card"><h4>NETTO</h4><div class="v" id="sum_net">0.00 CHF</div></div>
    </div>

    <div class="table">
      <div class="table-head">
        <div style="height:1px"></div>
        <div class="right"><button id="toggleRows" class="btn btn-muted">Vedi tutto</button></div>
      </div>
      <table id="tbl">
        <thead><tr>
          <th class="col-id">ID</th>
          <th class="col-type">TIPO</th>
          <th class="col-amt">AMOUNT</th>
          <th class="col-cur">CURRENCY</th>
          <th class="col-cat">CATEGORY</th>
          <th class="col-note">NOTE</th>
          <th class="col-date">CREATED AT</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="charts">
      <div class="chart-card">
        <h4 class="chart-title">ANDAMENTO GIORNALIERO</h4>
        <div class="chart-wrap"><canvas id="lineChart"></canvas></div>
      </div>
      <div class="chart-card">
        <h4 class="chart-title">PESO ENTRATE / USCITE</h4>
        <div class="chart-wrap"><canvas id="pieChart"></canvas></div>
      </div>
    </div>
  </div>

<script>
function money(v) {{
  try {{
    return new Intl.NumberFormat('it-CH', {{ style: 'decimal', minimumFractionDigits: 2, maximumFractionDigits: 2 }}).format(Number(v));
  }} catch {{ return v; }}
}}
function fmtDate(iso) {{
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n) => String(n).padStart(2, '0');
  return d.getFullYear() + "-" + pad(d.getMonth()+1) + "-" + pad(d.getDate()) + " " + pad(d.getHours()) + ":" + pad(d.getMinutes());
}}
function currentQuery() {{
  const f = document.getElementById('from').value;
  const t = document.getElementById('to').value;
  const typ = document.getElementById('type').value;
  const q = document.getElementById('q').value.trim();
  const u = new URL(window.location.origin + '/dashboard/data');
  if (f) u.searchParams.set('from', f);
  if (t) u.searchParams.set('to', t);
  if (typ) u.searchParams.set('type', typ);
  if (q) u.searchParams.set('q', q);
  return u;
}}

let showAll = false; const SHOW_LIMIT = 20;
document.getElementById('toggleRows').addEventListener('click', () => {{
  showAll = !showAll;
  document.getElementById('toggleRows').textContent = showAll ? 'Mostra 20' : 'Vedi tutto';
  loadData();
}});
document.getElementById('apply').addEventListener('click', loadData);
document.getElementById('pdf').addEventListener('click', () => {{
  const u = currentQuery(); u.pathname = '/dashboard/pdf';
  const a = document.createElement('a'); a.href = u.toString(); a.download = 'flai-report.pdf';
  document.body.appendChild(a); a.click(); a.remove();
}});

let lineChart, pieChart;
function renderCharts(allRows) {{
  const byDay = {{}}; let sumIn = 0, sumOut = 0;
  for (const r of allRows) {{
    const day = (r.created_at || '').slice(0,10);
    const amt = Number(r.amount || 0);
    const isIn = (r.type === 'in');
    if (!byDay[day]) byDay[day] = {{ in: 0, out: 0 }};
    if (isIn) {{ byDay[day].in += amt; sumIn += amt; }} else {{ byDay[day].out += amt; sumOut += amt; }}
  }}
  const days = Object.keys(byDay).sort();
  const inSerie = days.map(d => byDay[d].in);
  const outSerie = days.map(d => byDay[d].out);

  const lc = document.getElementById('lineChart').getContext('2d');
  if (lineChart) lineChart.destroy();
  lineChart = new Chart(lc, {{
    type: 'line',
    data: {{ labels: days, datasets: [
      {{ label: 'Entrate', data: inSerie, tension: .35 }},
      {{ label: 'Uscite',  data: outSerie, tension: .35 }}
    ] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: '#cbd3df' }} }} }},
      scales:  {{ x: {{ ticks: {{ color: '#cbd3df' }} }}, y: {{ ticks: {{ color: '#cbd3df' }} }} }}
    }}
  }});

  const pc = document.getElementById('pieChart').getContext('2d');
  if (pieChart) pieChart.destroy();
  pieChart = new Chart(pc, {{
    type: 'pie',
    data: {{ labels: ['Entrate','Uscite'], datasets: [ {{ data: [sumIn, sumOut] }} ] }},
    options: {{ responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: '#cbd3df' }} }} }}
    }}
  }});
}}

async function loadData() {{
  const url = currentQuery();
  const res = await fetch(url);
  if (!res.ok) {{ alert('Errore nel caricamento dati (' + res.status + ')'); return; }}
  const js = await res.json();

  document.getElementById('sum_in').textContent  = money(js.sum_in)  + " CHF";
  document.getElementById('sum_out').textContent = money(js.sum_out) + " CHF";
  document.getElementById('sum_net').textContent = money(js.sum_net) + " CHF";

  const all = js.rows || [];
  const rows = showAll ? all : all.slice(0, SHOW_LIMIT);

  const tb = document.querySelector('#tbl tbody'); tb.innerHTML = '';
  for (const r of rows) {{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="col-id">${{r.id}}</td>
      <td class="col-type">${{r.type}}</td>
      <td class="col-amt" style="text-align:right">${{money(r.amount)}}</td>
      <td class="col-cur">${{r.currency}}</td>
      <td class="col-cat">${{r.category ?? ''}}</td>
      <td class="col-note">${{r.note ?? ''}}</td>
      <td class="col-date">${{fmtDate(r.created_at)}}</td>`;
    tb.appendChild(tr);
  }}

}}

loadData();
</script>  renderCharts(all);
}}

loadData();
</script>

</body></html>"""
    return HTMLResponse(html)

@APP.get("/dashboard/data", response_class=JSONResponse)  # type: ignore
async def dashboard_data(
    request: Request,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    type: str | None = None,
    q: str | None = None,
):
    d_from = _iso_or_none(from_)
    d_to   = _iso_or_none(to)
    if type not in (None, "", "in", "out"):
        raise HTTPException(status_code=400, detail="invalid type")
    with get_conn() as conn:
        rows = _fetch_dashboard_rows(conn, d_from, d_to, type, q)

    s_in = Decimal("0"); s_out = Decimal("0"); out_rows = []
    for r in rows:
        amt = Decimal(str(r["amount"]))
        if r["type"] == "in": s_in += amt
        else: s_out += amt
        out_rows.append({
            "id": r["id"], "type": r["type"], "amount": str(amt),
            "currency": r["currency"], "category": r["category"], "note": r["note"],
            "created_at": r["created_at"].isoformat() if hasattr(r["created_at"],"isoformat") else r["created_at"],
        })
    return JSONResponse({"rows": out_rows, "sum_in": str(s_in), "sum_out": str(s_out), "sum_net": str(s_in - s_out)})


@APP.get("/dashboard/pdf")  # type: ignore
async def dashboard_pdf(
    request: Request,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    type: str | None = None,
    q: str | None = None,
):
    d_from = _iso_or_none(from_)
    d_to   = _iso_or_none(to)
    if type not in (None, "", "in", "out"):
        raise HTTPException(status_code=400, detail="invalid type")

    with get_conn() as conn:
        rows = _fetch_dashboard_rows(conn, d_from, d_to, type, q)

    # PDF pulito con intestazione e colonne allineate, NIENTE errori di quoting
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.lib.colors import black, HexColor

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4

        # barra in alto con brand
        c.setFillColor(HexColor(BRAND_BLUE))
        c.rect(0, height - 14*mm, width, 14*mm, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(18*mm, height - 10*mm, "FLAI – Report")

        # riga filtri
        y = height - 22*mm
        pieces = []
        if d_from or d_to:
            pieces.append(f"{d_from or '…'} → {d_to or '…'}")
        if type:
            pieces.append(type)
        if q:
            pieces.append(f"filtro: {q}")
        c.setFont("Helvetica", 10)
        c.setFillColor(black)
        if pieces:
            c.drawString(18*mm, y, " · ".join(pieces))
            y -= 6*mm

        # intestazione tabella
        c.setFont("Helvetica-Bold", 10)
        x_id=18*mm; x_tp=30*mm; x_amt=42*mm; x_cur=66*mm; x_cat=82*mm; x_note=122*mm; x_dt=176*mm
        c.drawString(x_id,y,"ID")
        c.drawString(x_tp,y,"TP")
        c.drawString(x_amt,y,"AMOUNT")
        c.drawString(x_cur,y,"CUR")
        c.drawString(x_cat,y,"CATEGORY")
        c.drawString(x_note,y,"NOTE")
        c.drawString(x_dt,y,"CREATED AT")
        y -= 5.5*mm
        c.setFont("Helvetica", 10)

        s_in = Decimal("0"); s_out = Decimal("0")
        for r in rows:
            amt = Decimal(str(r["amount"]))
            if r["type"] == "in": s_in += amt
            else: s_out += amt

            if y < 20*mm:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 20*mm

            c.drawString(x_id, y, str(r["id"]))
            c.drawString(x_tp, y, r["type"])
            # importo allineato a destra
            c.drawRightString(x_amt + 20*mm, y, f"{amt:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            c.drawString(x_cur, y, r["currency"])
            c.drawString(x_cat, y, (r["category"] or "")[:30])
            c.drawString(x_note, y, (r["note"] or "")[:50])

            created = r["created_at"]
            if hasattr(created, "strftime"):
                created = created.strftime("%Y-%m-%d %H:%M")
            # data allineata a destra: niente tagli
            c.drawRightString(x_dt + 22*mm, y, str(created))

            y -= 5.0*mm

        # totali
        y -= 6*mm
        net = s_in - s_out
        cur = rows[0]["currency"] if rows else "CHF"
        c.setFont("Helvetica-Bold", 11)
        c.drawString(18*mm, y, f"Entrate: {s_in:.2f} {cur}  •  Uscite: {s_out:.2f} {cur}  •  Netto: {net:.2f} {cur}")
        c.save()

        buf.seek(0)
        fname = f"flai-report_{(d_from or '...')}_{(d_to or '...')}.pdf"
        return Response(
            content=buf.read(),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'}
        )

    except Exception:
        # fallback .txt se la reportlab fallisse
        txt = "\n".join([
            f"{r['id']}\t{r['type']}\t{r['amount']}\t{r['currency']}\t{r['category']}\t{r['note']}"
            for r in rows
        ])
        fname = f"flai-report_{(d_from or '...')}_{(d_to or '...')}.txt"
        return Response(
            content=txt.encode("utf-8"),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'}
        )

# === FINE DASHBOARD ==========================================================


from typing import Optional, Literal
from decimal import Decimal
from datetime import datetime, timezone
from pydantic import BaseModel, Field, constr

class MovementIn(BaseModel):
    type: Literal["in","out"] = Field(..., description="in = entrata, out = uscita")
    amount: Decimal = Field(..., gt=0)
    currency: constr(min_length=3, max_length=3) = "CHF"
    category: Optional[str] = None
    note: Optional[str] = None
    created_at: Optional[datetime] = None

@APP.post("/api/movements")
def create_movement(m: MovementIn):
    dt = m.created_at or datetime.now(timezone.utc)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO movements(type, amount, currency, category, note, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (m.type, m.amount, m.currency.upper(), m.category, m.note, dt)
        )
        row = dict(cur.fetchone())
        conn.commit()
    return {"ok": True, "id": row["id"], "created_at": row["created_at"].isoformat()}
