"""
Vera candidate bot — implements the /v1/healthz, /v1/metadata, /v1/context,
/v1/tick and /v1/reply contract described in challenge-testing-brief.md and
demonstrated in api-call-examples.md.

Run:
    export OPENAI_API_KEY=sk-...
    uvicorn main:app --host 0.0.0.0 --port 8080
"""
from dotenv import load_dotenv
load_dotenv()
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# OpenAI SDK is optional at import time so the health/context endpoints
# still work even if no key is configured yet (useful for local testing).
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

# from openai import OpenAI
# client = OpenAI()
# try:
#     resp = client.chat.completions.create(
#         model='gpt-4o-mini',
#         max_tokens=10,
#         messages=[{'role':'user','content':'hi'}]
#     )
#     print(resp.choices[0].message.content)
# except Exception as e:
#     print(repr(e))
from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
)
resp = client.chat.completions.create(
    model='gemini-2.5-flash',
    messages=[{'role':'user','content':'who is the pm of india?'}]
)
print(resp.choices[0].message.content)
# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

MODEL_NAME = os.environ.get("VERA_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
TEAM_NAME = os.environ.get("VERA_TEAM_NAME", "Team Alpha")
TEAM_MEMBERS = os.environ.get("VERA_TEAM_MEMBERS", "Alice").split(",")
CONTACT_EMAIL = os.environ.get("VERA_CONTACT_EMAIL", "alice@example.com")
BOT_VERSION = "1.0.0"

AUTO_REPLY_MARKERS = [
    "thank you for contacting",
    "our team will respond shortly",
    "we have received your message",
    "will get back to you shortly",
    "currently unavailable",
]

HOSTILE_MARKERS = [
    "stop messaging", "not interested", "leave me alone", "unsubscribe",
    "stop contacting", "why are you bothering", "this is useless",
    "annoying", "harassing", "go away",
]

INTENT_TRANSITION_MARKERS = [
    "let's do it", "lets do it", "ok let's", "sounds good", "yes please",
    "go ahead", "sure, do it", "confirm", "book it", "send it",
]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# _client = OpenAI() if (OpenAI and os.environ.get("OPENAI_API_KEY")) else None
_client = (
    OpenAI(base_url=LLM_BASE_URL) if (OpenAI and os.environ.get("OPENAI_API_KEY") and LLM_BASE_URL)
    else OpenAI() if (OpenAI and os.environ.get("OPENAI_API_KEY"))
    else None
)

app = FastAPI(title="Vera Candidate Bot")
START = time.time()

# --------------------------------------------------------------------------- #
# In-memory stores
#
# NOTE: this is intentionally simple (dict-based) for the challenge. Swap for
# Redis/Postgres if you need multi-worker / crash-safe persistence.
# --------------------------------------------------------------------------- #

# (scope, context_id) -> {"version": int, "payload": dict}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id -> conversation state
# {
#   "turns": [{"from": "merchant"|"vera", "message": str, "at": str}],
#   "merchant_id": str | None,
#   "customer_id": str | None,
#   "trigger_id": str | None,
#   "ended": bool,
#   "waiting_until": float | None,     # epoch seconds
#   "auto_reply_streak": int,
#   "sent_bodies": set[str],
# }
conversations: dict[str, dict] = {}

# suppression_key -> expiry epoch seconds (or None = forever)
suppressed_keys: dict[str, Optional[float]] = {}

# merchant_id -> expiry epoch seconds (blanket suppression after hostility)
suppressed_merchants: dict[str, float] = {}


def _now_epoch() -> float:
    return time.time()


def _contexts_loaded_counts() -> dict[str, int]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts.keys():
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def get_payload(scope: str, context_id: str | None) -> dict | None:
    if not context_id:
        return None
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class ContextBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class Action(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: Literal["vera", "merchant_on_behalf"]
    trigger_id: str
    template_name: str
    template_params: list[str] = Field(default_factory=list)
    body: str
    cta: str
    suppression_key: str
    rationale: str


class TickResponse(BaseModel):
    actions: list[Action]


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


class ReplyResponse(BaseModel):
    action: Literal["send", "wait", "end"]
    body: Optional[str] = None
    cta: Optional[str] = None
    wait_seconds: Optional[int] = None
    rationale: str


# --------------------------------------------------------------------------- #
# Health / metadata
# --------------------------------------------------------------------------- #

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": _contexts_loaded_counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": MODEL_NAME,
        "approach": "single-prompt composer with retrieval over digest items + dispatch by trigger.kind",
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Context ingestion
# --------------------------------------------------------------------------- #

@app.post("/v1/context")
async def push_context(body: ContextBody):
    key = (body.scope, body.context_id)
    current = contexts.get(key)

    if current and current["version"] >= body.version:
        raise HTTPException(
            status_code=409,
            detail={"accepted": False, "reason": "stale_version", "current_version": current["version"]},
        )

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z",
    }


# --------------------------------------------------------------------------- #
# Suppression helpers
# --------------------------------------------------------------------------- #

def is_suppressed(suppression_key: str, merchant_id: str | None, now: float) -> bool:
    exp = suppressed_keys.get(suppression_key)
    if exp is not None and (exp is True or now < exp):  # True == forever
        return True
    if exp is True:
        return True
    if merchant_id:
        m_exp = suppressed_merchants.get(merchant_id)
        if m_exp and now < m_exp:
            return True
    return False


def suppress_key(suppression_key: str, forever: bool = False, seconds: float | None = None):
    if forever:
        suppressed_keys[suppression_key] = None  # None = don't expire (handled below)
        suppressed_keys[suppression_key] = float("inf")
    elif seconds:
        suppressed_keys[suppression_key] = _now_epoch() + seconds


def suppress_merchant(merchant_id: str, days: float = 30):
    suppressed_merchants[merchant_id] = _now_epoch() + days * 86400


# --------------------------------------------------------------------------- #
# LLM composer
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT_TICK = """\
You are Vera, an AI growth assistant that messages local business owners (merchants) \
on behalf of a platform, and sometimes messages their customers on the merchant's behalf.

Rules for every message you compose:
- Never include a URL (http, https, www) — this is a hard fail.
- Be specific: cite real numbers/details from the context provided, not generic filler.
- Match the category's voice/tone guidance and avoid any taboo vocabulary listed.
- Keep the message short enough to read on WhatsApp (roughly 2-4 sentences).
- End with a clear next step. Prefer an open-ended question unless the trigger \
  clearly calls for a binary or multi-choice CTA (e.g. slot booking).
- If sending on behalf of the merchant to a customer, mention the merchant/clinic by name, \
  and respect the customer's stated language/tone preferences if given.
- Restraint is fine: if the material genuinely isn't relevant or interesting, say so in \
  your rationale — but if asked to compose, still produce your best message; the choice \
  to skip an action for a given trigger is made by the caller, not you.

Return ONLY a JSON object, no markdown fences, no preamble, with exactly these keys:
{
  "body": "<the full message text, no URLs>",
  "cta": "<one of: open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none>",
  "rationale": "<one or two sentences on why this message and CTA were chosen>"
}
"""

SYSTEM_PROMPT_REPLY = """\
You are Vera, continuing an existing WhatsApp-style conversation with a merchant \
(or occasionally a customer messaged on the merchant's behalf).

You will be given the conversation so far and the latest inbound message. Decide \
whether to send a reply, wait, or end the conversation:

- "end": the other party explicitly opted out, was hostile, or the conversation has \
  reached a natural close with no more value to add.
- "wait": the inbound message is clearly an automated/canned auto-reply (not a real \
  human response). Suggest a wait_seconds appropriate to the situation (a few hours \
  for a first auto-reply, up to 24h if it recurs).
- "send": anything else — including honoring explicit requests, answering follow-up \
  questions, redirecting out-of-scope asks back to the original topic without being \
  pushy, or moving from qualification straight to action if the other party has \
  clearly signaled they're ready (don't keep asking qualifying questions once someone \
  has said "yes, let's do it" or equivalent — advance to a concrete next step instead).

Never include a URL in body. Never repeat a message body that has already been sent \
in this conversation (a list of previously sent bodies is provided — compose something \
new if you send again).

Return ONLY a JSON object, no markdown fences, no preamble, with these keys \
(omit body/cta if action is not "send"; omit wait_seconds if action is not "wait"):
{
  "action": "send" | "wait" | "end",
  "body": "<message text>",
  "cta": "<open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none>",
  "wait_seconds": <integer, only if action == "wait">,
  "rationale": "<one or two sentences explaining the decision>"
}
"""


def _call_llm(system: str, user_payload: dict) -> dict:
    if _client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not set / openai package unavailable — "
            "cannot compose messages. Set the env var and retry."
        )
    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    # strip stray markdown fences just in case (response_format=json_object
    # should prevent this, but keep the net for models/base-urls that ignore it)
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM did not return valid JSON: {e}\nRaw: {text[:500]}")


def _strip_urls(body: str) -> str:
    # Defensive net in case the model slips one in anyway — hard fail is -3/URL
    # per the brief, so we scrub rather than risk it.
    return URL_PATTERN.sub("", body).strip()


def compose_for_trigger(trigger: dict, merchant: dict | None, category: dict | None,
                         customer: dict | None) -> dict:
    payload = {
        "trigger": trigger,
        "merchant": merchant,
        "category": category,
        "customer": customer,
    }
    result = _call_llm(SYSTEM_PROMPT_TICK, payload)
    result["body"] = _strip_urls(result.get("body", ""))
    return result


def compose_reply(conv_state: dict, incoming_message: str, merchant: dict | None,
                   category: dict | None, customer: dict | None) -> dict:
    payload = {
        "conversation_history": conv_state["turns"],
        "latest_message": incoming_message,
        "previously_sent_bodies": list(conv_state["sent_bodies"]),
        "merchant": merchant,
        "category": category,
        "customer": customer,
    }
    result = _call_llm(SYSTEM_PROMPT_REPLY, payload)
    if result.get("action") == "send" and "body" in result:
        result["body"] = _strip_urls(result["body"])
    return result


# --------------------------------------------------------------------------- #
# /v1/tick
# --------------------------------------------------------------------------- #

@app.post("/v1/tick", response_model=TickResponse)
async def tick(body: TickBody):
    now = _now_epoch()
    actions: list[Action] = []

    for trigger_id in body.available_triggers:
        trigger = get_payload("trigger", trigger_id)
        if not trigger:
            continue

        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")
        merchant = get_payload("merchant", merchant_id)
        if not merchant:
            continue

        category = get_payload("category", merchant.get("category_slug"))
        customer = get_payload("customer", customer_id) if customer_id else None

        suppression_key = trigger.get("suppression_key", f"trigger:{trigger_id}")
        if is_suppressed(suppression_key, merchant_id, now):
            continue

        try:
            composed = compose_for_trigger(trigger, merchant, category, customer)
        except RuntimeError:
            # No LLM configured / composer failure -> skip this trigger rather
            # than send a broken/empty message.
            continue

        if not composed.get("body"):
            continue  # bot chose restraint

        send_as = "merchant_on_behalf" if customer_id else "vera"
        conv_id = f"conv_{merchant_id}_{trigger_id}"

        conversations.setdefault(conv_id, {
            "turns": [],
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trigger_id,
            "ended": False,
            "waiting_until": None,
            "auto_reply_streak": 0,
            "sent_bodies": set(),
        })
        conversations[conv_id]["turns"].append(
            {"from": "vera", "message": composed["body"], "at": body.now}
        )
        conversations[conv_id]["sent_bodies"].add(composed["body"])

        actions.append(Action(
            conversation_id=conv_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            send_as=send_as,
            trigger_id=trigger_id,
            template_name=f"vera_{trigger.get('kind', 'generic')}_v1",
            template_params=[],
            body=composed["body"],
            cta=composed.get("cta", "open_ended"),
            suppression_key=suppression_key,
            rationale=composed.get("rationale", ""),
        ))

    return TickResponse(actions=actions)


# --------------------------------------------------------------------------- #
# /v1/reply
# --------------------------------------------------------------------------- #

def _is_auto_reply(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in AUTO_REPLY_MARKERS)


def _is_hostile(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in HOSTILE_MARKERS)


@app.post("/v1/reply", response_model=ReplyResponse)
async def reply(body: ReplyBody):
    conv = conversations.setdefault(body.conversation_id, {
        "turns": [],
        "merchant_id": body.merchant_id,
        "customer_id": body.customer_id,
        "trigger_id": None,
        "ended": False,
        "waiting_until": None,
        "auto_reply_streak": 0,
        "sent_bodies": set(),
    })

    if conv["ended"]:
        return ReplyResponse(action="end", rationale="Conversation already closed.")

    conv["turns"].append({"from": body.from_role, "message": body.message, "at": body.received_at})

    merchant = get_payload("merchant", body.merchant_id) if body.merchant_id else None
    category = get_payload("category", merchant.get("category_slug")) if merchant else None
    customer = get_payload("customer", body.customer_id) if body.customer_id else None

    # --- deterministic guardrails before spending an LLM call --------------

    if _is_hostile(body.message):
        conv["ended"] = True
        if body.merchant_id:
            suppress_merchant(body.merchant_id, days=30)
        return ReplyResponse(
            action="end",
            rationale="Merchant expressed clear hostility / opt-out intent; closing "
                       "conversation and suppressing this merchant for 30 days.",
        )

    if _is_auto_reply(body.message):
        conv["auto_reply_streak"] += 1
        streak = conv["auto_reply_streak"]
        if streak == 1:
            try:
                composed = compose_reply(conv, body.message, merchant, category, customer)
                composed_body = composed.get("body") or (
                    "Looks like an auto-reply — when the owner sees this, just reply "
                    "whenever's convenient."
                )
            except RuntimeError:
                composed_body = (
                    "Looks like an auto-reply — when the owner sees this, just reply "
                    "whenever's convenient."
                )
            conv["turns"].append({"from": "vera", "message": composed_body, "at": body.received_at})
            conv["sent_bodies"].add(composed_body)
            return ReplyResponse(
                action="send", body=composed_body, cta="binary_yes_no",
                rationale="First auto-reply detected; one gentle nudge flagging it for the owner.",
            )
        elif streak == 2:
            return ReplyResponse(
                action="wait", wait_seconds=86400,
                rationale="Same auto-reply twice in a row — owner likely not at phone. "
                           "Waiting 24h before retry.",
            )
        else:
            conv["ended"] = True
            return ReplyResponse(
                action="end",
                rationale="Auto-reply 3x in a row with zero real engagement; closing conversation.",
            )
    else:
        conv["auto_reply_streak"] = 0

    # --- otherwise, let the composer decide (handles intent transitions, ---
    # --- curveball redirects, normal follow-ups, etc.) ----------------------

    try:
        composed = compose_reply(conv, body.message, merchant, category, customer)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    action = composed.get("action", "send")

    if action == "end":
        conv["ended"] = True
        return ReplyResponse(action="end", rationale=composed.get("rationale", "Conversation closed."))

    if action == "wait":
        wait_seconds = int(composed.get("wait_seconds", 14400))
        conv["waiting_until"] = _now_epoch() + wait_seconds
        return ReplyResponse(
            action="wait", wait_seconds=wait_seconds,
            rationale=composed.get("rationale", "Waiting before next contact."),
        )

    # action == "send"
    reply_body = composed.get("body", "").strip()
    if not reply_body or reply_body in conv["sent_bodies"]:
        # anti-repetition guard: force a fresh compose once more, else fall back
        try:
            retry = compose_reply(conv, body.message, merchant, category, customer)
            reply_body = retry.get("body", "").strip()
        except RuntimeError:
            pass
        if not reply_body or reply_body in conv["sent_bodies"]:
            reply_body = "Got it — let me know if there's anything specific you'd like next."

    conv["turns"].append({"from": "vera", "message": reply_body, "at": body.received_at})
    conv["sent_bodies"].add(reply_body)

    return ReplyResponse(
        action="send",
        body=reply_body,
        cta=composed.get("cta", "open_ended"),
        rationale=composed.get("rationale", ""),
    )