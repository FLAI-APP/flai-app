import os, io, json, base64
import httpx
from fastapi import FastAPI, Request

APP = FastAPI()

# === ENV VARS (le metterai su Render) ===
VERIFY_TOKEN   = os.getenv("META_VERIFY_TOKEN")     # es: flai-verify-123
WA_TOKEN       = os.getenv("META_WA_TOKEN")         # token WhatsApp Cloud API
PHONE_ID       = os.getenv("META_PHONE_ID")         # WhatsApp Phone Number ID
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")        # la tua chiave OpenAI
SHEETS_URL     = os.getenv("SHEETS_WEBHOOK_URL")    # URL Apps Script (già testato)

# ---------- WhatsApp VERIFY (GET) ----------
@APP.get("/webhook")
def verify(mode: str=None, challenge: str=None, token: str=None,
          hub_mode: str=None, hub_challenge: str=None, hub_verify_token: str=None):
    mode = mode or hub_mode
    challenge = challenge or hub_challenge
    token = token or hub_verify_token
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return int(challenge)
    return "forbidden"

# ---------- Helpers WhatsApp ----------
async def wa_reply_text(to: str, text: str):
    url = f"https://graph.facebook.com/v20.0/{PHONE_ID}/messages"
    payload = {"messaging_product":"whatsapp","to":to,"type":"text","text":{"body":text[:4000]}}
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    async with httpx.AsyncClient(timeout=30) as c:
        await c.post(url, headers=headers, json=payload)

async def wa_get_media_bytes(media_id: str) -> bytes:
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    async with httpx.AsyncClient(timeout=60) as c:
        meta = await c.get(f"https://graph.facebook.com/v20.0/{media_id}", headers=headers)
        meta.raise_for_status()
        url = meta.json()["url"]
        r = await c.get(url, headers=headers)
        r.raise_for_status()
        return r.content

# ---------- OpenAI ----------
async def ai_chat(text: str) -> str:
    import openai
    openai.api_key = OPENAI_API_KEY
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
          {"role":"system","content":"Sei un assistente aziendale. Sii chiaro e conciso."},
          {"role":"user","content": text}
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

async def ai_transcribe(audio_bytes: bytes) -> str:
    import openai
    openai.api_key = OPENAI_API_KEY
    fileobj = io.BytesIO(audio_bytes)
    tr = openai.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=("audio.ogg", fileobj, "audio/ogg")
    )
    return tr.text.strip()

async def ai_vision_extract(img_bytes: bytes) -> str:
    import openai
    openai.api_key = OPENAI_API_KEY
    b64 = base64.b64encode(img_bytes).decode()
    messages = [
      {"role":"system","content":"Estrai da scontrino/fattura righe contabili (descrizione, importo, data). Testo sintetico."},
      {"role":"user","content":[
          {"type":"text","text":"Rendi il testo pulito; una riga per voce con importo e data se presente."},
          {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}
      ]}
    ]
    resp = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return resp.choices[0].message.content.strip()

async def extract_movements(text: str) -> dict:
    import openai
    openai.api_key = OPENAI_API_KEY
    prompt = """Dal testo seguente estrai movimenti contabili in JSON con chiave 'movimenti'.
Ogni movimento: { "tipo":"entrata|uscita", "voce":"...", "importo": numero, "note":"...", "fonte":"whatsapp" }.
Rispondi SOLO JSON valido."""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":prompt},{"role":"user","content": text}],
        temperature=0
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except:
        return {"movimenti":[]}

async def save_to_sheet(movements: dict):
    if not movements.get("movimenti"): return
    async with httpx.AsyncClient(timeout=20) as c:
        await c.post(SHEETS_URL, json=movements)

# ---------- Webhook WhatsApp (POST) ----------
@APP.post("/webhook")
async def incoming(req: Request):
    body = await req.json()
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                sender = msg.get("from")
                if "text" in msg:
                    text = msg["text"]["body"]
                    reply = await ai_chat(text)
                    await wa_reply_text(sender, reply)
                    await save_to_sheet(await extract_movements(text))
                elif "audio" in msg:
                    bytes_ = await wa_get_media_bytes(msg["audio"]["id"])
                    text = await ai_transcribe(bytes_)
                    reply = await ai_chat(text)
                    await wa_reply_text(sender, reply)
                    await save_to_sheet(await extract_movements(text))
                elif "image" in msg:
                    bytes_ = await wa_get_media_bytes(msg["image"]["id"])
                    extracted = await ai_vision_extract(bytes_)
                    reply = await ai_chat(f"Testo estratto dall'immagine:\n{extracted}")
                    await wa_reply_text(sender, reply)
                    await save_to_sheet(await extract_movements(extracted))
    return {"ok": True}

