from fastapi import Query, Request

@APP.get("/webhook")
def verify(
    request: Request,
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    hub_mode_alt: str | None = Query(None, alias="hub_mode"),
    hub_challenge_alt: str | None = Query(None, alias="hub_challenge"),
    hub_verify_token_alt: str | None = Query(None, alias="hub_verify_token"),
):
    # Prendi il valore dai parametri "dot" o dagli "underscore"
    mode = hub_mode or hub_mode_alt
    challenge = hub_challenge or hub_challenge_alt
    token = hub_verify_token or hub_verify_token_alt

    if mode == "subscribe" and token == META_VERIFY_TOKEN and challenge:
        try:
            return int(challenge)
        except:
            return challenge
    return "forbidden"

