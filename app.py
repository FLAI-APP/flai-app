# app.py — piani + tenants + quota (compatibile con endpoints esistenti)
import os
from datetime import datetime, date
from decimal import Decimal

from fastapi import FastAPI, Request, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text

APP = FastAPI()

# -------- CORS --------
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
APP.add_middleware(
    CORSMiddleware,
    allow_origins=origins or [],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- ENV & DB --------
API_KEY_APP  = os.getenv("API_KEY_APP", "")
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
if DATABASE_URL and "sslmode=" not in DATABASE_URL:
    sep = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = DATABASE_URL + f"{sep}sslmode=require"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# =========================================
# Bootstrap schema (idempotente)
# =========================================
def _bootstrap_schema():
    with engine.begin() as conn:
        # MOVEMENTS (come prima)
        conn.execute(text("CREATE TABLE IF NOT EXISTS movements (id SERIAL PRIMARY KEY);"))
        cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name='movements'
        """)).scalars().all()
        if "importo" in cols:
            if "amount" not in cols:
                conn.execute(text("ALTER TABLE movements RENAME COLUMN importo TO amount;"))
            else:
                conn.execute(text("ALTER TABLE movements DROP COLUMN importo;"))

        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS type VARCHAR(10);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS amount NUMERIC(14,2);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS currency VARCHAR(8);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS category VARCHAR(50);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS note TEXT;"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS voce VARCHAR(50);"))
        conn.execute(text("ALTER TABLE movements ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;"))
        conn.execute(text("UPDATE movements SET currency='CHF' WHERE currency IS NULL;"))
        conn.execute(text("UPDATE movements SET created_at=NOW() WHERE created_at IS NULL;"))
        conn.execute(text("UPDATE movements SET voce='generale' WHERE voce IS NULL;"))
        conn.execute(text("UPDATE movements SET type='in' WHERE type IS NULL;"))
        conn.execute(text("UPDATE movements SET amount=0 WHERE amount IS NULL;"))
        conn.execute(text("ALTER TABLE movements ALTER COLUMN currency SET NOT NULL;"))
        conn.execute(text("ALTER TABLE movements ALTER COLUMN type SET NOT NULL;"))
        conn.execute(text("ALTER TABLE movements ALTER COLUMN amount SET NOT NULL;"))
        conn.execute(text("ALTER TABLE movements ALTER COLUMN voce SET NOT NULL;"))

        # PLANS (piani)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS plans (
              id SERIAL PRIMARY KEY,
              name VARCHAR(30) UNIQUE NOT NULL,
              monthly_limit INTEGER NOT NULL,    -- richieste/mese
              features TEXT                      -- JSON opzionale (stringa)
            );
        """))

        # TENANTS (clienti)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tenants (
              id SERIAL PRIMARY KEY,
              name VARCHAR(120) NOT NULL,
              plan_id INTEGER NOT NULL REFERENCES plans(id),
              tenant_key VARCHAR(64) UNIQUE NOT NULL,  -- chiave usata dal cliente (X-Tenant-Key)
              created_at TIMESTAMP DEFAULT NOW()
            );
        """))

        # QUOTAS (consumi per mese)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS quotas (
              id SERIAL PRIMARY KEY,
              tenant_id INTEGER NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
              period VARCHAR(7) NOT NULL,              -- es. '2025-09'
              used_requests INTEGER NOT NULL DEFAULT 0,
              UNIQUE (tenant_id, period)
            );
        """))

try:
    _bootstrap_schema()
except Exception as e:
    print("SCHEMA_BOOTSTRAP_ERROR:", e, flush=True)

# =========================================
# Helpers piani/tenant/quota
# =========================================
def _require_api_key(request: Request):
    if request.headers.get("x-api-key") != API_KEY_APP:
        raise HTTPException(status_code=401, detail="invalid api key")

def _period_now() -> str:
    today = date.today()
    return f"{today.year:04d}-{today.month:02d}"

def _get_tenant_and_plan(tenant_key: str):
    if not tenant_key:
        raise HTTPException(400, "X-Tenant-Key missing")
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT t.id AS tenant_id, t.name AS tenant_name, p.id AS plan_id, p.name AS plan_name, p.monthly_limit
            FROM tenants t
            JOIN plans p ON p.id = t.plan_id
            WHERE t.tenant_key = :k
        """), {"k": tenant_key}).mappings().first()
    if not row:
        raise HTTPException(401, "tenant not found / invalid tenant key")
    return row

def _enforce_and_count_quota(tenant_id: int, plan_limit: int, increment: int = 1):
    period = _period_now()
    with engine.begin() as conn:
        # ensure row
        conn.execute(text("""
            INSERT INTO quotas(tenant_id, period, used_requests)
            VALUES (:tid, :p, 0)
            ON CONFLICT (tenant_id, period) DO NOTHING
        """), {"tid": tenant_id, "p": period})

        current = conn.execute(text("""
            SELECT used_requests FROM quotas WHERE tenant_id=:tid AND period=:p
        """), {"tid": tenant_id, "p": period}).scalar_one()

        if current + increment > plan_limit:
            raise HTTPException(402, f"quota exceeded for period {period} (limit={plan_limit}, used={current})")

        conn.execute(text("""
            UPDATE quotas SET used_requests = used_requests + :inc
            WHERE tenant_id=:tid AND period=:p
        """), {"inc": increment, "tid": tenant_id, "p": period})

# =========================================
# Middleware: protezione per API key (admin) e tenant key (client)
# =========================================
@APP.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    open_paths = {"/", "/healthz", "/debug"}
    admin_paths = {"/admin/seed_plans", "/admin/create_tenant", "/admin/quotas"}

    # admin endpoints: richiedono X-API-Key
    if path in admin_paths:
        _require_api_key(request)
        return await call_next(request)

    # endpoints protetti lato cliente (richiedono X-Tenant-Key + contano quota)
    client_protected_prefixes = ("/movements", "/analytics", "/summaries")
    if path.startswith(client_protected_prefixes):
        # tenant key è richiesta e quota si conta dentro i singoli handler
        if not request.headers.get("x-tenant-key"):
            return JSONResponse({"error":"missing_tenant_key"}, status_code=401)

    # open
    return await call_next(request)

# =========================================
# Routes base
# =========================================
@APP.get("/")
def root():
    return {"status": "ok", "service": "flai-app"}

@APP.get("/healthz")
def healthz():
    return "ok"

@APP.get("/debug")
def debug():
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        dbok = True
    except Exception as e:
        return {"db_ok": False, "err": str(e)}
    return {"db_ok": dbok}

# =========================================
# ADMIN endpoints (setup piani/tenant/quote)
# =========================================
@APP.post("/admin/seed_plans")
def admin_seed_plans():
    """
    Crea/aggiorna piani Base/Pro/Premium.
    Base: 1000 req/mese
    Pro:  10000 req/mese
    Premium: 100000 req/mese
    """
    PLANS = [
        ("Base", 1000,  '{"reports":false,"vision":false,"audio":false}'),
        ("Pro",  10000, '{"reports":true,"vision":true,"audio":true}'),
        ("Premium", 100000, '{"reports":true,"vision":true,"audio":true,"phone":true}')
    ]
    with engine.begin() as conn:
        for name, limit, feats in PLANS:
            conn.execute(text("""
                INSERT INTO plans(name, monthly_limit, features)
                VALUES (:n, :l, :f)
                ON CONFLICT (name) DO UPDATE SET monthly_limit=:l, features=:f
            """), {"n": name, "l": limit, "f": feats})
    return {"ok": True, "seeded": [p[0] for p in PLANS]}

@APP.post("/admin/create_tenant")
def admin_create_tenant(item: dict = Body(...)):
    """
    Body esempio:
    { "name":"Ristorante Sole", "plan":"Pro", "tenant_key":"sole_ABC123" }
    Se tenant_key non è fornita, la generiamo.
    """
    name = (item.get("name") or "").strip()
    plan_name = (item.get("plan") or "Base").strip()
    tenant_key = (item.get("tenant_key") or f"tk_{int(datetime.utcnow().timestamp())}").strip()
    if not name:
        raise HTTPException(422, "name required")
    with engine.begin() as conn:
        plan = conn.execute(text("SELECT id FROM plans WHERE name=:n"), {"n": plan_name}).scalar()
        if not plan:
            raise HTTPException(404, f"plan '{plan_name}' not found (seed plans first)")
        conn.execute(text("""
            INSERT INTO tenants(name, plan_id, tenant_key) VALUES (:nm, :pid, :tk)
        """), {"nm": name, "pid": plan, "tk": tenant_key})
    return {"ok": True, "tenant_key": tenant_key, "plan": plan_name, "name": name}

@APP.get("/admin/quotas")
def admin_quotas():
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT t.name AS tenant, q.period, q.used_requests, p.monthly_limit
            FROM quotas q
            JOIN tenants t ON t.id = q.tenant_id
            JOIN plans p ON p.id = t.plan_id
            ORDER BY q.period DESC, tenant ASC
        """)).mappings().all()
    return {"items": [dict(r) for r in rows]}

# =========================================
# Movements (protetti da X-Tenant-Key + quota)
# =========================================
@APP.post("/movements")
def create_movement(request: Request, item: dict = Body(...)):
    tenant_key = request.headers.get("x-tenant-key")
    tinfo = _get_tenant_and_plan(tenant_key)

    try:
        t = (item.get("type") or "").strip().lower()
        if t not in ("in", "out"):
            raise HTTPException(422, "type must be 'in' or 'out'")
        amt = Decimal(str(item.get("amount", 0)))
        cur = (item.get("currency") or "CHF").upper()
        cat = item.get("category")
        note = item.get("note")
        voce = item.get("voce") or "generale"

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO movements(type, amount, currency, category, note, voce, created_at)
                VALUES (:t, :a, :c, :cat, :n, :v, :ts)
            """), {"t": t, "a": amt, "c": cur, "cat": cat, "n": note, "v": voce, "ts": datetime.utcnow()})
        # 1 richiesta consumata
        _enforce_and_count_quota(tinfo["tenant_id"], tinfo["monthly_limit"], increment=1)
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": "db_failed_insert", "detail": str(e)}

@APP.get("/movements")
def list_movements(request: Request, _from: str = Query(None, alias="from"), to: str = None, limit: int = 200):
    tenant_key = request.headers.get("x-tenant-key")
    tinfo = _get_tenant_and_plan(tenant_key)

    try:
        q = "SELECT id, type, amount, currency, category, note, voce, created_at FROM movements WHERE 1=1"
        params = {}
        if _from:
            q += " AND created_at >= :f"; params["f"] = _from
        if to:
            q += " AND created_at < :t"; params["t"] = to
        q += " ORDER BY created_at DESC LIMIT :lim"
        params["lim"] = limit

        with engine.begin() as conn:
            rows = conn.execute(text(q), params).mappings().all()
        # 1 richiesta consumata
        _enforce_and_count_quota(tinfo["tenant_id"], tinfo["monthly_limit"], increment=1)
        return {"items": [dict(r) for r in rows]}
    except Exception as e:
        return {"error": "db_failed_query", "detail": str(e)}

@APP.post("/movements/bulk")
def bulk_movements(request: Request, items: list[dict] = Body(...)):
    tenant_key = request.headers.get("x-tenant-key")
    tinfo = _get_tenant_and_plan(tenant_key)

    try:
        if not isinstance(items, list) or not items:
            raise HTTPException(422, "body must be a non-empty JSON array")
        now = datetime.utcnow()
        to_insert = []
        for it in items:
            t = (it.get("type") or "").strip().lower()
            if t not in ("in","out"):
                raise HTTPException(422, "each item.type must be 'in' or 'out'")
            amt = Decimal(str(it.get("amount",0)))
            cur = (it.get("currency") or "CHF").upper()
            cat = it.get("category")
            note = it.get("note")
            voce = it.get("voce") or "generale"
            to_insert.append({"t":t,"a":amt,"c":cur,"cat":cat,"n":note,"v":voce,"ts":now})

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO movements(type, amount, currency, category, note, voce, created_at)
                SELECT :t, :a, :c, :cat, :n, :v, :ts
            """), to_insert)
        # N richieste consumate = numero elementi (puoi cambiarlo a 1 se preferisci)
        _enforce_and_count_quota(tinfo["tenant_id"], tinfo["monthly_limit"], increment=len(to_insert))
        return {"status":"ok","inserted":len(to_insert)}
    except HTTPException:
        raise
    except Exception as e:
        return {"error":"db_failed_bulk","detail":str(e)}

# =========================================
# Analytics (protetti) — compact totali
# =========================================
@APP.get("/analytics/overview")
def analytics_overview(request: Request, days: int = 30, compact: int = 1):
    tenant_key = request.headers.get("x-tenant-key")
    tinfo = _get_tenant_and_plan(tenant_key)
    if days <= 0 or days > 365:
        raise HTTPException(422, "days must be between 1 and 365")

    try:
        with engine.begin() as conn:
            totals = conn.execute(text("""
                SELECT
                  COALESCE(SUM(CASE WHEN type='in'  THEN amount END),0) AS total_in,
                  COALESCE(SUM(CASE WHEN type='out' THEN amount END),0) AS total_out
                FROM movements
                WHERE created_at >= CURRENT_DATE - ((CAST(:d AS integer) - 1) * INTERVAL '1 day')
            """), {"d": days}).mappings().first()

        tin  = float(totals["total_in"]  or 0)
        tout = float(totals["total_out"] or 0)
        # quota: 1 richiesta
        _enforce_and_count_quota(tinfo["tenant_id"], tinfo["monthly_limit"], increment=1)
        return {
            "days": days,
            "totals": {"in": tin, "out": tout, "net": tin - tout}
        }
    except Exception as e:
        return {"error":"db_failed_analytics","detail":str(e)}

