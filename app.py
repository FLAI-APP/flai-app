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


# ====================== DASHBOARD (HTML + DATA + PDF) ======================
# Import locali alla sezione (ok se duplicano import già presenti)
import os, io
from datetime import datetime, timedelta, date
from typing import Optional

from fastapi import Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED

import psycopg
from psycopg.rows import dict_row

# ---- Config letta da env (riusa le tue variabili) ----
DASH_USER = os.getenv("DASHBOARD_USER") or os.getenv("DASH_USER") or "flai"
DASH_PASS = os.getenv("DASHBOARD_PASSWORD") or os.getenv("DASH_PASSWORD") or "flai-test"
DATABASE_URL = os.getenv("DATABASE_URL", "")
BRAND_BLUE = os.getenv("BRAND_BLUE", "#0B2A44")
ACCENT_GOLD = os.getenv("ACCENT_GOLD", "#D4AF37")
LOGO_URL = os.getenv("LOGO_URL", "")  # URL pubblico al logo (PNG/JPG)

dash_sec = HTTPBasic()

def dash_auth(creds: HTTPBasicCredentials = Depends(dash_sec)):
    if creds.username != DASH_USER or creds.password != DASH_PASS:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

def dash_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL mancante")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def dash_parse_date(s: Optional[str]) -> Optional[date]:
    if not s: return None
    try: return datetime.strptime(s, "%Y-%m-%d").date()
    except: return None

def dash_default_range():
    today = date.today()
    return (today - timedelta(days=29)), today

def dash_fetch(d_from: Optional[date], d_to: Optional[date], mov_type: Optional[str], q: Optional[str]):
    sql = """
      SELECT id, "type", amount, currency, category, note, created_at
      FROM movements
      WHERE 1=1
    """
    p = {}
    if d_from:
        sql += " AND created_at >= %(df)s"
        p["df"] = datetime(d_from.year, d_from.month, d_from.day)
    if d_to:
        sql += " AND created_at < %(dt)s"
        p["dt"] = datetime(d_to.year, d_to.month, d_to.day) + timedelta(days=1)
    if mov_type in ("in","out"):
        sql += ' AND "type" = %(t)s'
        p["t"] = mov_type
    if q:
        sql += """
          AND (
            CAST(id AS TEXT) ILIKE %(pat)s OR
            "type" ILIKE %(pat)s OR
            CAST(amount AS TEXT) ILIKE %(pat)s OR
            currency ILIKE %(pat)s OR
            category ILIKE %(pat)s OR
            note ILIKE %(pat)s OR
            TO_CHAR(created_at,'YYYY-MM-DD HH24:MI') ILIKE %(pat)s
          )
        """
        p["pat"] = f"%{q}%"
    sql += " ORDER BY created_at DESC LIMIT 2000"

    tot = """
      SELECT
        COALESCE(SUM(CASE WHEN "type"='in'  THEN amount END),0) AS total_in,
        COALESCE(SUM(CASE WHEN "type"='out' THEN amount END),0) AS total_out
      FROM movements
      WHERE 1=1
    """
    if d_from:  tot += " AND created_at >= %(df)s"
    if d_to:    tot += " AND created_at < %(dt)s"
    if mov_type in ("in","out"): tot += ' AND "type" = %(t)s'
    if q:
        tot += """
          AND (
            CAST(id AS TEXT) ILIKE %(pat)s OR
            "type" ILIKE %(pat)s OR
            CAST(amount AS TEXT) ILIKE %(pat)s OR
            currency ILIKE %(pat)s OR
            category ILIKE %(pat)s OR
            note ILIKE %(pat)s OR
            TO_CHAR(created_at,'YYYY-MM-DD HH24:MI') ILIKE %(pat)s
          )
        """

    with dash_conn() as c:
        rows = c.execute(sql, p).fetchall()
        totals = c.execute(tot, p).fetchone()

    items = []
    for r in rows:
        ts = r["created_at"]
        ts_fmt = ts.strftime("%Y-%m-%d %H:%M") if isinstance(ts, datetime) else str(ts)
        items.append({
            "id": r["id"],
            "type": r["type"],
            "amount": float(r["amount"]) if r["amount"] is not None else 0.0,
            "currency": r.get("currency") or "",
            "category": (r.get("category") or ""),
            "note": (r.get("note") or ""),
            "created_at": ts_fmt,
        })
    t_in  = float(totals["total_in"] if totals and totals["total_in"] is not None else 0.0)
    t_out = float(totals["total_out"] if totals and totals["total_out"] is not None else 0.0)
    return items, {"in": t_in, "out": t_out, "net": t_in - t_out}

@APP.get("/dashboard", response_class=HTMLResponse)
def dashboard(_: bool = Depends(dash_auth)):
    logo = (f'<img src="{LOGO_URL}" alt="logo" style="height:40px;opacity:.80"/>' if LOGO_URL
            else '<div style="color:#fff;font-weight:800;">FLAI</div>')
    css = f"""
    :root {{
      --brand: {BRAND_BLUE};
      --accent: {ACCENT_GOLD};
      --bg: #0f1115;
      --card: #151922;
      --muted: #8ea0b6;
      --text: #e5efff;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text);
           font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
    header {{ background:var(--brand); padding:12px 18px; display:flex; align-items:center; gap:12px;
             border-bottom:1px solid rgba(255,255,255,.08); }}
    .title {{ font-weight:800; letter-spacing:.4px; }}
    .wrap {{ max-width:1280px; margin:18px auto; padding:0 16px; }}
    .toolbar {{ display:grid; grid-template-columns: repeat(7, minmax(0,1fr)); gap:10px; align-items:end; margin-bottom:12px; }}
    label {{ font-size:12px; color:var(--muted); display:block; margin-bottom:6px; }}
    input, select {{ width:100%; padding:10px 12px; background:#0f1320; color:var(--text);
                     border:1px solid #243149; border-radius:8px; outline:none; }}
    button, .btn {{ background:var(--accent); color:#0b0b0b; border:none; border-radius:8px; padding:10px 14px;
                    font-weight:800; cursor:pointer; text-decoration:none; text-align:center; }}
    .cards {{ display:grid; grid-template-columns: repeat(3,1fr); gap:12px; margin:10px 0 16px; }}
    .card {{ background:var(--card); border:1px solid #243149; border-radius:14px; padding:16px; }}
    .card h3 {{ margin:0 0 6px; font-size:13px; color:var(--muted); font-weight:700; }}
    .card .val {{ font-size:22px; font-weight:900; }}
    .table-wrap {{ background:var(--card); border:1px solid #243149; border-radius:14px; overflow:hidden; }}
    table {{ width:100%; border-collapse:collapse; }}
    thead th {{ text-align:left; font-size:12px; color:var(--muted); background:#0f1320;
               padding:10px; border-bottom:1px solid #243149; }}
    tbody td {{ padding:10px; border-bottom:1px solid #1d2537; font-size:14px; }}
    tbody tr:hover {{ background:#10182a; }}
    .num {{ text-align:right; font-variant-numeric: tabular-nums; }}
    .w-id{{width:70px}} .w-type{{width:90px}} .w-amt{{width:140px}} .w-cur{{width:90px}}
    .w-cat{{width:260px}} .w-note{{width:auto}} .w-ts{{width:170px}}
    .err {{ color:#ff6b6b; font-weight:800; margin-top:8px; }}
    """
    html = f"""
<!doctype html>
<html><head>
  <meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>FLAI Dashboard</title>
  <style>{css}</style>
</head>
<body>
<header>
  {logo}
  <div class="title">FLAI Dashboard</div>
</header>
<div class="wrap">
  <div class="toolbar">
    <div><label>Dal</label><input type="date" id="from"></div>
    <div><label>Al</label><input type="date" id="to"></div>
    <div><label>Tipo</label>
      <select id="movtype"><option value="">Tutti</option><option value="in">Entrate</option><option value="out">Uscite</option></select>
    </div>
    <div style="grid-column: span 3;"><label>Ricerca unica</label>
      <input id="q" placeholder="es. fornitori, CHF, 2025-09, nota...">
    </div>
    <div><label>&nbsp;</label><button id="apply">Applica filtri</button></div>
    <div><label>&nbsp;</label><a id="pdfbtn" class="btn" href="#" target="_blank">Scarica PDF</a></div>
  </div>

  <div class="cards">
    <div class="card"><h3>Entrate</h3><div id="v_in" class="val">0.00</div></div>
    <div class="card"><h3>Uscite</h3><div id="v_out" class="val">0.00</div></div>
    <div class="card"><h3>Netto</h3><div id="v_net" class="val">0.00</div></div>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th class="w-id">ID</th><th class="w-type">TIPO</th><th class="w-amt num">AMOUNT</th>
          <th class="w-cur">CURRENCY</th><th class="w-cat">CATEGORY</th>
          <th class="w-note">NOTE</th><th class="w-ts">CREATED AT</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <div id="err" class="err"></div>
</div>
<script>
function fmt(x){{return (Math.round((x||0)*100)/100).toFixed(2);}}
function params(){{
  const usp=new URLSearchParams();
  const f=document.getElementById('from').value, t=document.getElementById('to').value;
  const mt=document.getElementById('movtype').value, q=document.getElementById('q').value.trim();
  if(f)usp.set('from',f); if(t)usp.set('to',t); if(mt)usp.set('type',mt); if(q)usp.set('q',q);
  return usp;
}}
async function loadData(){{
  const e=document.getElementById('err'); e.textContent='';
  const qs=params().toString();
  try{{
    const r=await fetch('/dashboard/data'+(qs?('?'+qs):'')); if(!r.ok) throw new Error('HTTP '+r.status);
    const j=await r.json();
    document.getElementById('v_in').textContent=fmt(j.totals.in);
    document.getElementById('v_out').textContent=fmt(j.totals.out);
    document.getElementById('v_net').textContent=fmt(j.totals.net);
    const tb=document.getElementById('tbody'); tb.innerHTML='';
    for(const m of j.items){{
      const tr=document.createElement('tr');
      tr.innerHTML = `
        <td class="w-id">${{m.id}}</td>
        <td class="w-type">${{m.type}}</td>
        <td class="w-amt num">${{fmt(m.amount)}}</td>
        <td class="w-cur">${{m.currency}}</td>
        <td class="w-cat">${{m.category||''}}</td>
        <td class="w-note">${{m.note||''}}</td>
        <td class="w-ts">${{m.created_at}}</td>`;
      tb.appendChild(tr);
    }}
    document.getElementById('pdfbtn').href = '/dashboard/pdf'+(qs?('?'+qs):'');
  }}catch(err){{
    console.error(err); e.textContent='Errore nel caricamento dati ('+err.message+')';
  }}
}}
function setDefaults(){{
  const t=new Date(), s=new Date(t.getTime()-29*24*60*60*1000);
  const iso=d=>d.toISOString().slice(0,10);
  document.getElementById('from').value=iso(s);
  document.getElementById('to').value=iso(t);
}}
document.getElementById('apply').addEventListener('click',e=>{{e.preventDefault();loadData();}});
setDefaults(); loadData();
</script>
</body></html>
    """
    return HTMLResponse(html)

@APP.get("/dashboard/data")
def dashboard_data(
    _ok: bool = Depends(dash_auth),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str]   = Query(None, alias="to"),
    type_: Optional[str] = Query(None, alias="type"),
    q: Optional[str]     = Query(None, alias="q"),
):
    try:
        df = dash_parse_date(from_)
        dt = dash_parse_date(to)
        if not df and not dt:
            df, dt = dash_default_range()
        items, totals = dash_fetch(df, dt, type_, q)
        return {"items": items, "totals": totals}
    except Exception as e:
        return JSONResponse({"error":"db_failed","detail":str(e)}, status_code=500)

@APP.get("/dashboard/pdf")
def dashboard_pdf(
    _ok: bool = Depends(dash_auth),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str]   = Query(None, alias="to"),
    type_: Optional[str] = Query(None, alias="type"),
    q: Optional[str]     = Query(None, alias="q"),
):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm

        df = dash_parse_date(from_)
        dt = dash_parse_date(to)
        if not df and not dt:
            df, dt = dash_default_range()
        items, totals = dash_fetch(df, dt, type_, q)

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        W,H = A4
        c.setFont("Helvetica-Bold",14)
        c.drawString(20*mm, H-20*mm, "FLAI - Report Movimenti")
        c.setFont("Helvetica",10)
        c.drawString(20*mm, H-26*mm, f"Periodo: {df or '-'} → {dt or '-'}  |  type={type_ or 'tutti'}  q={q or ''}")
        c.setFont("Helvetica-Bold",11)
        c.drawString(20*mm, H-36*mm, f"Entrate: {totals['in']:.2f}  Uscite: {totals['out']:.2f}  Netto: {totals['net']:.2f}")

        y = H-46*mm
        c.setFont("Helvetica-Bold",9)
        c.drawString(20*mm,y,"ID"); c.drawString(35*mm,y,"Tipo"); c.drawString(55*mm,y,"Importo")
        c.drawString(85*mm,y,"Valuta"); c.drawString(105*mm,y,"Categoria"); c.drawString(145*mm,y,"Note"); c.drawString(190*mm,y,"Data")
        y -= 6*mm; c.setFont("Helvetica",9)
        for m in items:
            if y < 20*mm: c.showPage(); y = H-20*mm; c.setFont("Helvetica",9)
            c.drawString(20*mm,y,str(m["id"]))
            c.drawString(35*mm,y,m["type"])
            c.drawRightString(82*mm,y,f"{m['amount']:.2f}")
            c.drawString(85*mm,y,m["currency"])
            c.drawString(105*mm,y,(m["category"] or "")[:18])
            c.drawString(145*mm,y,(m["note"] or "")[:30])
            c.drawRightString(200*mm,y,m["created_at"])
            y -= 6*mm

        c.showPage(); c.save(); buf.seek(0)
        return StreamingResponse(buf, media_type="application/pdf",
            headers={"Content-Disposition": 'inline; filename="flai-report.pdf"'})
    except Exception as e:
        return JSONResponse({"error":"pdf_failed","detail":str(e)}, status_code=500)
# ====================== /DASHBOARD ======================

