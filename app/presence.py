"""
Tracks which chat/voice sessions are currently live, and maps them back to
the CRM user (via state.session_identities) so the email watcher can find
"is this team member online right now" and push a message into whichever
channel (voice or chat) they're actually connected on.

This did not exist before -- state.session_identities only recorded which
CRM user a session belongs to, not whether that session is currently
connected. Both chat's SSE stream and the voice websocket need to call
register()/unregister()/set_busy() below; see the two call-site notes at
the bottom of this file for exactly where.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import state

# session_id -> ConnectionRecord
_connections: dict[str, "ConnectionRecord"] = {}


@dataclass
class ConnectionRecord:
    kind: str  # "chat" | "voice"
    send: Callable[[dict], "asyncio.Future"]  # async fn(payload: dict) -> None
    last_seen: float = field(default_factory=time.time)
    busy: bool = False  # True while a response is actively streaming/speaking


def register(session_id: str, kind: str, send_fn) -> None:
    """Call this the moment a chat SSE stream opens or a voice websocket
    connects. `send_fn` must be an async function that takes one dict and
    delivers it on that channel (already wired to that session's own
    send_json/patchMessage-equivalent)."""
    _connections[session_id] = ConnectionRecord(kind=kind, send=send_fn)


def unregister(session_id: str) -> None:
    _connections.pop(session_id, None)


def set_busy(session_id: str, busy: bool) -> None:
    """Call this when a turn starts streaming/speaking (busy=True) and again
    when it finishes (busy=False). Lets notify() know whether it's safe to
    interrupt right now or whether it should wait."""
    rec = _connections.get(session_id)
    if rec:
        rec.busy = busy
        rec.last_seen = time.time()


def find_sessions_for_user(crm_user_email: str) -> list[str]:
    """Reverse-looks-up state.session_identities to find every live session
    currently bound to this CRM user."""
    return [
        sid
        for sid, identity in state.session_identities.items()
        if identity and getattr(identity, "user", None) == crm_user_email
        and sid in _connections
    ]


def is_online(crm_user_email: str) -> bool:
    return len(find_sessions_for_user(crm_user_email)) > 0


async def notify(
    crm_user_email: str,
    payload: dict,
    *,
    max_wait_seconds: float = 120.0,
    poll_interval: float = 1.0,
) -> bool:
    """
    Delivers `payload` to every live session belonging to crm_user_email.
    If a session is mid-response (busy=True), waits for it to finish before
    sending -- rather than talking over an in-progress voice response or
    interleaving into a mid-stream chat answer.

    Returns True if delivered to at least one session, False if the user
    had no live session (caller should treat this as "offline" and fall
    back to the configured backup contact).
    """
    sessions = find_sessions_for_user(crm_user_email)
    if not sessions:
        return False

    delivered = False
    waited = 0.0
    for sid in sessions:
        rec = _connections.get(sid)
        if not rec:
            continue
        while rec.busy and waited < max_wait_seconds:
            await asyncio.sleep(poll_interval)
            waited += poll_interval
            rec = _connections.get(sid)
            if not rec:
                break
        if rec:
            try:
                await rec.send(payload)
                delivered = True
            except Exception:
                # Connection likely dropped between the check and the send;
                # treat as not-delivered for this session, keep trying others.
                pass
    return delivered


# --------------------------------------------------------------------------
# Wiring notes (apply these two small edits where each channel is defined):
#
# 1) app/routes/chat_routes.py, inside chat_stream()'s event_gen():
#      - on stream start:  presence.register(session_id, "chat", send_fn)
#        where send_fn wraps whatever you use to push an out-of-band system
#        message into that SSE connection (a queue the generator also reads
#        from works well here).
#      - right before the LLM call:      presence.set_busy(session_id, True)
#      - in the `finally` block:         presence.set_busy(session_id, False)
#      - on disconnect / stream end:     presence.unregister(session_id)
#
# 2) Voice/ws_voice.py, inside register_voice_ws()'s ws_voice():
#      - after `await ws.accept()`:      presence.register(session_id, "voice", send_json)
#      - when state["speaking"] flips:   presence.set_busy(session_id, state["speaking"])
#      - in the disconnect handler:      presence.unregister(session_id)
# --------------------------------------------------------------------------