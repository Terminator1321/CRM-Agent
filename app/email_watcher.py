"""
Standalone background poller -- deliberately NOT part of the chat/voice
agent graph. It runs on its own asyncio task, wakes up every POLL_INTERVAL
seconds, reads unread Gmail via the reused EmailReader, classifies each
message, and either:

  (a) routes it to the CRM team member already assigned to that client
      (delivered live into their chat/voice session if they're online,
      via app.presence), or
  (b) for a brand-new/unassigned client, asks a fallback team member to
      confirm before creating a CRM Lead.

Everything IMAP/SMTP-related here is your existing EmailReader.py, moved
in as-is -- only the polling loop itself changed, from a blocking
`while True: time.sleep()` thread to an asyncio task that fits into
FastAPI's event loop (see lifespan.py for how it's started/stopped).

CONFIG YOU NEED TO FILL IN (see bottom of file):
  - POLL_INTERVAL_SECONDS
  - FALLBACK_USER_EMAIL   ("top authority" / on-call person for new leads
                            and for when the assigned owner is offline)
  - The exact CRM doctype/field names for "who owns this deal" -- I've
    guessed CRM Deal / CRM Lead with a `deal_owner` / `lead_owner` field
    based on crm_client.py's generic doctype API; confirm against your
    actual Frappe CRM schema (crm_metadata("CRM Deal") will show you).
"""
import asyncio
import re
from typing import Optional

from EmailReader import EmailReader
from CRM_Unified.crm_client import crm_client
from CRM_Unified.tools import crm_search, crm_create
from . import agent_setup, presence, state
from .logging_setup import logger

POLL_INTERVAL_SECONDS = 60  # tune to 60-300s; every 5 min is plenty for most inboxes
FALLBACK_USER_EMAIL = "sales.lead@yourcompany.com"  # TODO: set to your real on-call/admin user

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

CLASSIFY_SYSTEM = (
    "You triage inbound emails for a CRM. Reply with ONLY a compact JSON object:\n"
    '{"sender_type": "client"|"team_member"|"other", '
    '"intent": "new_client"|"existing_client_task"|"general", '
    '"summary": "<one sentence>"}\n'
    "For now, treat anyone not obviously an internal team member as a client. "
    "\"new_client\" = they are introducing themselves / asking to talk / expressing "
    "interest for the first time. \"existing_client_task\" = a reply or request tied "
    "to an ongoing deal/relationship."
)


async def _classify_email(subject: str, body: str) -> dict:
    """LLM classification, run off the event loop since .invoke() is blocking."""
    from .graph.chat_state import parse_json_loose

    def _call():
        resp = agent_setup.assistant.llm.model.invoke([
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": f"Subject: {subject}\n\n{body[:2000]}"},
        ])
        return resp.content

    try:
        raw = await asyncio.to_thread(_call)
        return parse_json_loose(raw)
    except Exception:
        logger.exception("[EmailWatcher] classification failed; defaulting to general/client")
        return {"sender_type": "client", "intent": "general", "summary": subject}


def _extract_sender_email(from_header: str) -> Optional[str]:
    match = _EMAIL_RE.search(from_header or "")
    return match.group(0).lower() if match else None


async def _find_assigned_owner(sender_email: str) -> Optional[dict]:
    """
    Looks up the client by email in CRM and returns the deal/lead they're
    already tied to, if any, plus its owner. Returns None for a brand-new
    contact with no CRM record.

    NOTE: adjust doctype + field names to match your actual Frappe CRM
    schema -- these are best-guess field names from typical Frappe CRM setups.
    """
    def _search():
        return crm_search(
            "CRM Deal",
            filters={"email": sender_email},
        )

    try:
        result = await asyncio.to_thread(_search)
    except Exception:
        logger.exception("[EmailWatcher] CRM lookup failed for %s", sender_email)
        return None

    import json
    try:
        deals = json.loads(result) if isinstance(result, str) else result
    except Exception:
        deals = None

    if not deals:
        return None
    deal = deals[0] if isinstance(deals, list) else deals
    return deal


async def _deliver_to_team_member(owner_email: str, message: dict) -> bool:
    """Push into the owner's live chat/voice session if they're online.
    Returns True if delivered live, False if they're offline (caller should
    fall back)."""
    return await presence.notify(owner_email, message)


async def _handle_existing_client_email(sender_email: str, subject: str, body: str, summary: str):
    deal = await _find_assigned_owner(sender_email)
    owner_email = (deal or {}).get("deal_owner") or (deal or {}).get("lead_owner")

    notification = {
        "type": "email_notification",
        "channel": "voice_and_chat",
        "from": sender_email,
        "subject": subject,
        "summary": summary,
        "body_preview": body[:500],
    }

    if owner_email:
        delivered = await _deliver_to_team_member(owner_email, notification)
        if delivered:
            logger.info("[EmailWatcher] Delivered live to assigned owner %s", owner_email)
            return
        logger.info("[EmailWatcher] Owner %s offline; falling back to %s", owner_email, FALLBACK_USER_EMAIL)

    # Owner not found, or offline -- fall back to the configured contact.
    await _deliver_to_team_member(FALLBACK_USER_EMAIL, notification)


async def _handle_new_client_email(sender_email: str, subject: str, body: str, summary: str):
    """No existing CRM record. Ask a human to confirm before creating a Lead
    -- try to reach them live first; if nobody's online, email the top
    authority directly and wait for their reply (handled by
    _check_pending_replies on later poll cycles)."""
    confirmation_request = {
        "type": "lead_confirmation_request",
        "from": sender_email,
        "subject": subject,
        "summary": summary,
        "body_preview": body[:500],
    }
    delivered = await _deliver_to_team_member(FALLBACK_USER_EMAIL, confirmation_request)
    if delivered:
        return

    # Nobody online -- ask by email instead, and remember we're waiting on
    # a reply before we act.
    logger.info("[EmailWatcher] No one online; emailing %s for lead approval", FALLBACK_USER_EMAIL)
    reader = EmailReader()
    ref_token = reader.new_ref_token()
    base_subject = f"New client inquiry -- create a lead? ({sender_email})"
    question_subject = reader.embed_ref_token(base_subject, ref_token)
    question_body = (
        f"A new client email came in that isn't tied to an existing CRM record.\n\n"
        f"From    : {sender_email}\n"
        f"Subject : {subject}\n"
        f"Summary : {summary}\n\n"
        f"---\n{body[:800]}\n---\n\n"
        f"Reply to this email with \"yes\" (or \"create it\") to create a lead, "
        f"or \"no\" to skip it.\n\n"
        f"(Please keep the [REF-{ref_token}] tag in the subject line when replying "
        f"-- some mail apps strip the parts we'd normally use to match your reply "
        f"back to this specific request.)"
    )
    ok, msg, question_message_id = await asyncio.to_thread(
        reader.send_new_email, FALLBACK_USER_EMAIL, question_subject, question_body
    )
    if not ok:
        logger.error("[EmailWatcher] Could not email %s for approval: %s", FALLBACK_USER_EMAIL, msg)
        return

    import json
    payload = json.dumps({
        "sender_email": sender_email,
        "subject": subject,
        "body": body,
    })
    await asyncio.to_thread(
        reader.create_pending_action, question_message_id, "create_lead", payload, FALLBACK_USER_EMAIL, question_subject, ref_token
    )
    logger.info("[EmailWatcher] Pending lead-approval sent to %s, waiting on reply", FALLBACK_USER_EMAIL)


CONFIRM_REPLY_SYSTEM = (
    "Reply to a yes/no confirmation email. Reply with ONLY a compact JSON object: "
    '{"approved": true|false}\n'
    'Treat "yes", "create it", "go ahead", "approved", "do it" as true. '
    'Treat "no", "skip", "don\'t", "reject" as false. If genuinely unclear, use false.'
)


async def _classify_confirmation_reply(body: str) -> bool:
    from .graph.chat_state import parse_json_loose

    def _call():
        resp = agent_setup.assistant.llm.model.invoke([
            {"role": "system", "content": CONFIRM_REPLY_SYSTEM},
            {"role": "user", "content": body[:500]},
        ])
        return resp.content

    try:
        raw = await asyncio.to_thread(_call)
        return bool(parse_json_loose(raw).get("approved"))
    except Exception:
        logger.exception("[EmailWatcher] Could not classify confirmation reply; treating as 'no'")
        return False


async def _check_pending_replies(reader: EmailReader, new_msgs: list[dict]):
    """For every newly-arrived message, check whether it's a reply to a
    question we're waiting on (e.g. the lead-approval email). If so, act on
    it instead of treating it as a fresh inbound email."""
    import json

    for msg in new_msgs:
        pending = await asyncio.to_thread(reader.find_pending_action_for_reply, msg)
        if not pending:
            continue

        question_message_id, action_type, payload_json, asked_to, status = pending
        approved = await _classify_confirmation_reply(msg.get("body", ""))

        if action_type == "create_lead":
            payload = json.loads(payload_json)
            if approved:
                try:
                    result = await create_lead_from_email(
                        payload["sender_email"], payload["subject"], payload["body"]
                    )
                    logger.info("[EmailWatcher] Lead created after email approval: %s", result)
                    reply_body = f"Done -- lead created for {payload['sender_email']}."
                except Exception:
                    logger.exception("[EmailWatcher] Lead creation failed after approval")
                    reply_body = f"Approval received, but lead creation failed for {payload['sender_email']}. Check logs."
            else:
                logger.info("[EmailWatcher] Lead creation declined for %s", payload["sender_email"])
                reply_body = f"Okay, skipped creating a lead for {payload['sender_email']}."

            await asyncio.to_thread(
                reader.reply_to_email, msg.get("message_id", ""), reply_body
            )
            await asyncio.to_thread(
                reader.resolve_pending_action, question_message_id, "approved" if approved else "rejected"
            )


async def create_lead_from_email(sender_email: str, subject: str, body: str) -> str:
    """Call this from the "confirm lead" route once a person approves."""
    def _create():
        return crm_create("CRM Lead", {
            "email": sender_email,
            "lead_name": sender_email.split("@")[0],
            "notes": f"Auto-created from inbound email.\nSubject: {subject}\n\n{body[:1000]}",
        })
    return await asyncio.to_thread(_create)


async def _process_one(msg: dict):
    sender_email = _extract_sender_email(msg.get("from", ""))
    if not sender_email:
        return
    classification = await _classify_email(msg.get("subject", ""), msg.get("body", ""))

    if classification.get("sender_type") != "client":
        logger.info("[EmailWatcher] Skipping non-client email from %s (for now)", sender_email)
        return

    if classification.get("intent") == "new_client":
        await _handle_new_client_email(sender_email, msg.get("subject", ""), msg.get("body", ""), classification.get("summary", ""))
    else:
        await _handle_existing_client_email(sender_email, msg.get("subject", ""), msg.get("body", ""), classification.get("summary", ""))


async def run_email_watcher(poll_interval: int = POLL_INTERVAL_SECONDS):
    """Started as a background asyncio task from lifespan.py. Runs until
    cancelled at shutdown."""
    reader = EmailReader()
    try:
        await asyncio.to_thread(reader.connect)
    except Exception:
        logger.exception("[EmailWatcher] Gmail connect failed -- watcher not started")
        return

    logger.info("[EmailWatcher] Started, polling every %ds", poll_interval)
    try:
        while True:
            try:
                new_msgs = await asyncio.to_thread(reader.sync_unread)

                # Replies to a pending question (e.g. lead approval) are
                # handled separately and should NOT also be run through
                # fresh classification below.
                still_pending = []
                for msg in new_msgs:
                    pending = await asyncio.to_thread(reader.find_pending_action_for_reply, msg)
                    if pending:
                        await _check_pending_replies(reader, [msg])
                    else:
                        still_pending.append(msg)

                for msg in still_pending:
                    try:
                        await _process_one(msg)
                    except Exception:
                        logger.exception("[EmailWatcher] Failed processing email %r", msg.get("subject"))
            except Exception:
                logger.exception("[EmailWatcher] Poll cycle failed")
            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        logger.info("[EmailWatcher] Stopping (shutdown)")
        raise
    finally:
        await asyncio.to_thread(reader.disconnect)