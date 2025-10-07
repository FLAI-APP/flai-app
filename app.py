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
# Whitelist: niente API key su queste rotte usate dal browser
    path = request.url.path
    WHITELIST = ("/healthz", "/dashboard", "/dashboard/data", "/reports/pdf")
    if path.startswith(WHITELIST):
      return await call_next(request)

    x_api_key = request.headers.get("X-API-Key")
    if x_api_key != os.getenv("API_KEY_APP"):
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
    try:
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
            headers={"Content-Disposition": f'attachment; filename="{fname}"'}
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
    d_from: dt.date,
    d_to: dt.date,
    category: Optional[str],
) -> bytes:
    """
    PDF minimale ma leggibile: intestazione, totali, e tabellina movimenti.
    """
    movements = _fetch_movements_between(d_from, d_to, category)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 15 * mm
    x = margin
    y = height - margin

    # Header
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 16)
    title = "FLAI – Report movimenti"
    if category:
        title += f" (categoria: {category})"
    c.drawString(x, y, title); y -= 10 * mm

    c.setFont("Helvetica", 11)
    c.drawString(x, y, f"Periodo: {d_from.isoformat()} → {d_to.isoformat()}"); y -= 7 * mm

    # Totali
    tot_in = sum(r["amount"] for r in movements if r["type"] == "in")
    tot_out = sum(r["amount"] for r in movements if r["type"] == "out")
    tot_net = tot_in - tot_out
    cur = movements[0]["currency"] if movements else "CHF"
    c.drawString(x, y, f"Totale entrate: {tot_in:.2f} {cur}   •   Totale uscite: {tot_out:.2f} {cur}   •   Netto: {tot_net:.2f} {cur}")
    y -= 10 * mm

    # Tabella
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, "Data")
    c.drawString(x + 28*mm, y, "Tipo")
    c.drawString(x + 45*mm, y, "Importo")
    c.drawString(x + 75*mm, y, "Cat.")
    c.drawString(x + 105*mm, y, "Nota")
    y -= 5 * mm
    c.setFont("Helvetica", 10)

    rows_per_page = int((y - margin) // (5.5 * mm))
    i = 0
    for r in movements:
        if i and i % rows_per_page == 0:
            c.showPage()
            c.setFillColor(black); c.setFont("Helvetica", 10)
            y = height - margin
        date_s = r["created_at"].strftime("%Y-%m-%d")
        c.drawString(x, y, date_s)
        c.drawString(x + 28*mm, y, "Entrata" if r["type"] == "in" else "Uscita")
        c.drawRightString(x + 72*mm, y, f'{r["amount"]:.2f} {r["currency"]}')
        c.drawString(x + 75*mm, y, (r["category"] or "")[:18])
        c.drawString(x + 105*mm, y, (r["note"] or "")[:40])
        y -= 5.5 * mm
        i += 1

    if not movements:
        c.drawString(x, y, "Nessun movimento nel periodo selezionato.")

    c.showPage()
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


# ==============================================================
# DASHBOARD (HTML + API dati) — filtri, tabella, export PDF
# ==============================================================

import os, io, json, base64, datetime as dt
from typing import Optional, List, Any, Dict
from decimal import Decimal

from fastapi import Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ---- Brand da ENV (puoi regolare per matchare il logo) ----
BRAND_BLUE  = os.getenv("BRAND_BLUE",  "#0B1E3A")   # un pelo più CHIARO del precedente
ACCENT_GOLD = os.getenv("ACCENT_GOLD", "#AA8F15")
LOGO_URL    = os.getenv("LOGO_URL", "")

# ---------- Helpers ----------
def _iso_or_none(s: Optional[str]) -> Optional[dt.date]:
    if not s: return None
    try: return dt.date.fromisoformat(s)
    except Exception: return None

def _like_or_none(s: Optional[str]) -> Optional[str]:
    return f"%{s.strip()}%" if s and s.strip() else None

def _get_val(row, key, idx):
    """Compat: estrae valore sia da dict-row che da tuple-row."""
    try:
        return row[key]
    except Exception:
        return row[idx]

def _fetch_rows_and_totals(conn, d_from: Optional[dt.date], d_to: Optional[dt.date],
                           typ: str, q: Optional[str]) -> Dict[str, Any]:
    sql = """
        SELECT id, type, amount::numeric(14,2) AS amount, currency, category, COALESCE(note,'') AS note, created_at
        FROM movements
        WHERE 1=1
    """
    params: List[Any] = []
    if d_from:
        sql += " AND created_at::date >= %s"
        params.append(d_from)
    if d_to:
        sql += " AND created_at::date <= %s"
        params.append(d_to)
    if typ in ("in", "out"):
        sql += " AND type = %s"
        params.append(typ)
    if q:
        like = _like_or_none(q)
        sql += """ AND (
            CAST(id AS TEXT) ILIKE %s OR type ILIKE %s OR currency ILIKE %s OR
            category ILIKE %s OR note ILIKE %s
        )"""
        params += [like, like, like, like, like]
    sql += " ORDER BY created_at DESC LIMIT 500"

    tsql = """
        SELECT
          COALESCE(SUM(CASE WHEN type='in'  THEN amount ELSE 0 END),0)::numeric(14,2) AS tin,
          COALESCE(SUM(CASE WHEN type='out' THEN amount ELSE 0 END),0)::numeric(14,2) AS tout
        FROM movements
        WHERE 1=1
    """
    tparams: List[Any] = []
    if d_from:
        tsql += " AND created_at::date >= %s"
        tparams.append(d_from)
    if d_to:
        tsql += " AND created_at::date <= %s"
        tparams.append(d_to)
    if typ in ("in", "out"):
        tsql += " AND type = %s"
        tparams.append(typ)
    if q:
        like = _like_or_none(q)
        tsql += """ AND (
            CAST(id AS TEXT) ILIKE %s OR type ILIKE %s OR currency ILIKE %s OR
            category ILIKE %s OR note ILIKE %s
        )"""
        tparams += [like, like, like, like, like]

    with get_conn() as c, c.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.execute(tsql, tparams)
        tin, tout = cur.fetchone()

    out_rows = []
    for r in rows:
        rid       = _get_val(r, "id",         0)
        rtype     = _get_val(r, "type",       1)
        amount    = _get_val(r, "amount",     2)
        currency  = _get_val(r, "currency",   3)
        category  = _get_val(r, "category",   4)
        note      = _get_val(r, "note",       5)
        created_at= _get_val(r, "created_at", 6)
        # normalizza tipi
        try:
            amount_f = float(amount)
        except Exception:
            amount_f = float(str(amount))
        created_iso = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        out_rows.append({
            "id": rid, "type": rtype, "amount": amount_f, "currency": currency,
            "category": category, "note": note, "created_at": created_iso
        })

    return {"rows": out_rows, "in_total": float(tin), "out_total": float(tout)}

# ---------- API dati ----------
@APP.get("/dashboard/data")
async def dashboard_data(
    request: Request,
    from_: Optional[str] = Query(default=None, alias="from"),
    to: Optional[str] = None,
    type: str = "all",
    q: Optional[str] = None,
):
    d_from = _iso_or_none(from_)
    d_to   = _iso_or_none(to)
    if type not in ("all", "in", "out"):
        raise HTTPException(400, "invalid type")
    try:
        data = _fetch_rows_and_totals(get_conn(), d_from, d_to, type, q)
        return JSONResponse({"ok": True, **data})
    except Exception as e:
        return JSONResponse({"ok": False, "error": "db_failed", "detail": str(e)}, status_code=500)

# ---------- HTML ----------
@APP.get("/dashboard")
async def dashboard(request: Request) -> HTMLResponse:
    html = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FLAI · Dashboard</title>
<style>
:root {{
  --brand-blue:{BRAND_BLUE};
  --accent-gold:{ACCENT_GOLD};
  --bg:#0f141b; --panel:#1c2430; --text:#e8eef6; --muted:#9aa4b2; --line:#2a3a4d;
  --radius:12px;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial; }}

.header {{ background:var(--brand-blue); padding:10px 16px; }}
.header-inner {{ display:flex; align-items:center; gap:14px; }}
.brand img {{ height:28px; width:28px; object-fit:contain; filter:brightness(.85) saturate(1.05); }}
.brand-name {{ font-weight:800; letter-spacing:.3px; }}

.toolbar {{ margin-top:10px; display:flex; align-items:center; gap:10px; }}
.toolbar .grow {{ flex:1; }}
input, select {{ background:#0f141b; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px 10px; }}
button.btn {{ background:var(--accent-gold); color:#111; border:none; padding:8px 12px; border-radius:10px; font-weight:700; cursor:pointer; }}
button.btn:active {{ transform:translateY(1px); }}

.kpis {{ display:flex; gap:16px; margin:14px 16px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:14px 16px; flex:1; }}
.card h4 {{ margin:0 0 8px 0; font-size:12px; color:var(--muted); letter-spacing:.6px; }}
.card .value {{ font-size:18px; font-weight:800; }}

.table-wrap {{ background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); margin:0 16px 30px 16px; overflow:auto; }}
table {{ width:100%; border-collapse:collapse; table-layout:fixed; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; }}
th {{ color:var(--muted); font-size:12px; letter-spacing:.6px; }}
th:nth-child(1) {{ width:60px;  }}  /* ID piccolo */
th:nth-child(2) {{ width:90px;  }}  /* TYPE piccolo */
th:nth-child(3) {{ width:130px; }}  /* AMOUNT medio */
th:nth-child(4) {{ width:90px;  }}  /* CURRENCY piccolo */
th:nth-child(5) {{ width:170px; }}  /* CATEGORY medio */
th:nth-child(6) {{ width:auto;  }}  /* NOTE largo */
th:nth-child(7) {{ width:160px; }}  /* CREATED AT medio */
td.amount {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.created {{ white-space:nowrap; }}
</style>
</head>
<body>
  <div class="header">
    <div class="header-inner">
      <div class="brand">
        {"<img src='"+LOGO_URL+"' alt='logo'/>" if LOGO_URL else ""}
      </div>
      <div class="brand-name">FLAI · Dashboard</div>
    </div>
    <!-- Toolbar dentro la barra blu -->
    <div class="toolbar">
      <label>Dal</label><input type="date" id="from">
      <label>al</label><input type="date" id="to">
      <select id="typ">
        <option value="all">TUTTI I TIPI</option>
        <option value="in">SOLO ENTRATE</option>
        <option value="out">SOLO USCITE</option>
      </select>
      <input id="q" placeholder="cerca testo (id, tipo, amount, note, categoria)" style="min-width:260px;">
      <button class="btn" id="apply">Applica filtri</button>
      <div class="grow"></div>
      <button class="btn" id="pdf">Scarica PDF</button>
    </div>
  </div>

  <div class="kpis">
    <div class="card"><h4>ENTRATE</h4><div class="value" id="v_in">0.00 CHF</div></div>
    <div class="card"><h4>USCITE</h4><div class="value" id="v_out">0.00 CHF</div></div>
    <div class="card"><h4>NETTO</h4><div class="value" id="v_net">0.00 CHF</div></div>
  </div>

  <div class="table-wrap">
    <table id="tbl">
      <thead>
        <tr>
          <th>ID</th><th>TIPO</th><th>AMOUNT</th><th>CURRENCY</th><th>CATEGORY</th><th>NOTE</th><th>CREATED AT</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

<script>
(function() {{
  const fmtMoney = new Intl.NumberFormat('it-CH', {{style:'decimal', minimumFractionDigits:2, maximumFractionDigits:2}});
  const CHF = n => fmtMoney.format(n) + " CHF";

  const elFrom = document.getElementById('from');
  const elTo   = document.getElementById('to');
  const elTyp  = document.getElementById('typ');
  const elQ    = document.getElementById('q');
  const elApply= document.getElementById('apply');
  const elPDF  = document.getElementById('pdf');
  const tb     = document.querySelector('#tbl tbody');

  // default: ultimo mese
  const today = new Date();
  const fromD = new Date(today); fromD.setMonth(fromD.getMonth()-1);
  elFrom.value = fromD.toISOString().slice(0,10);
  elTo.value   = today.toISOString().slice(0,10);

  async function load() {{
    const p = new URLSearchParams();
    if (elFrom.value) p.set('from', elFrom.value);
    if (elTo.value)   p.set('to', elTo.value);
    if (elTyp.value !== 'all') p.set('type', elTyp.value);
    if (elQ.value.trim()) p.set('q', elQ.value.trim());

    const r = await fetch('/dashboard/data?' + p.toString(), {{
      headers: {{ 'X-API-Key': '{os.getenv("API_KEY_APP","")}' }}
    }});
    if (!r.ok) {{ alert('Errore nel caricamento dati ('+r.status+')'); return; }}
    const j = await r.json();
    if (!j.ok) {{ alert('Errore: ' + (j.error||'unknown')); return; }}

    // KPI
    document.getElementById('v_in').textContent  = CHF(j.in_total);
    document.getElementById('v_out').textContent = CHF(j.out_total);
    document.getElementById('v_net').textContent = CHF(j.in_total - j.out_total);

    // Tabella
    tb.innerHTML = '';
    for (const row of j.rows) {{
      const d = new Date(row.created_at);
      const nice = d.toLocaleDateString('it-CH') + ' ' + d.toLocaleTimeString('it-CH', {{hour:'2-digit', minute:'2-digit'}});
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{row.id}}</td>
        <td>${{row.type}}</td>
        <td class="amount">${{fmtMoney.format(row.amount)}}</td>
        <td>${{row.currency}}</td>
        <td>${{row.category||''}}</td>
        <td>${{row.note||''}}</td>
        <td class="created">${{nice}}</td>
      `;
      tb.appendChild(tr);
    }}
  }}

  elApply.addEventListener('click', load);
  document.addEventListener('keydown', (e)=>{{ if (e.key==='Enter') load(); }});
  elPDF.addEventListener('click', ()=>{{
    const p = new URLSearchParams();
    if (elFrom.value) p.set('from', elFrom.value);
    if (elTo.value)   p.set('to', elTo.value);
    if (elTyp.value !== 'all') p.set('type', elTyp.value);
    if (elQ.value.trim()) p.set('q', elQ.value.trim());
    window.location.href = '/reports/pdf?' + p.toString();
  }});

  load();
}})();
</script>
</body></html>"""
    return HTMLResponse(content=html)
