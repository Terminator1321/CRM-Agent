"""
Tracks which chat/voice connections are currently live, and maps them back
to the CRM user (via state.session_identities) so the email watcher can
find "is this team member online right now" and push a message into
whichever channel(s) they're actually connected on.

This did not exist before -- state.session_identities only recorded which
CRM user a session belongs to, not whether that session is currently
connected. Both the chat notifications websocket and the voice websocket
call register()/unregister()/set_busy() below.

Keyed by (session_id, kind) rather than just session_id: the same
session_id is shared between Chat and Voice (that's intentional -- it's
how a voice call's history carries back into Chat), so a person can have
BOTH a chat connection and a voice connection open at once under the same
session_id. Keying by session_id alone would let one silently overwrite
the other's registration.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import state

ConnKey = tuple[str, str]  # (session_id, kind)

# (session_id, kind) -> ConnectionRecord
_connections: dict[ConnKey, "ConnectionRecord"] = {}


@dataclass
class ConnectionRecord:
    kind: str  # "chat" | "voice"
    send: Callable[[dict], "asyncio.Future"]  # async fn(payload: dict) -> None
    last_seen: float = field(default_factory=time.time)
    busy: bool = False  # True while a response is actively streaming/speaking


def register(session_id: str, kind: str, send_fn) -> None:
    """Call this the moment a chat notifications websocket or a voice
    websocket connects. `send_fn` must be an async function that takes one
    dict and delivers it on that channel."""
    _connections[(session_id, kind)] = ConnectionRecord(kind=kind, send=send_fn)


def unregister(session_id: str, kind: str) -> None:
    _connections.pop((session_id, kind), None)


def set_busy(session_id: str, busy: bool, kind: Optional[str] = None) -> None:
    """Call this when a turn starts streaming/speaking (busy=True) and again
    when it finishes (busy=False). Lets notify() know whether it's safe to
    interrupt right now or whether it should wait.

    kind=None (default) sets busy on every connection for this session_id
    -- fine for voice, where the same connection sends and streams. Chat
    passes kind="chat" explicitly, since the per-message stream that knows
    "busy" and the persistent notifications connection that holds the
    ConnectionRecord are two different objects (see notification_routes.py)."""
    for key, rec in _connections.items():
        if key[0] == session_id and (kind is None or key[1] == kind):
            rec.busy = busy
            rec.last_seen = time.time()


def find_connections_for_user(crm_user_email: str) -> list[ConnKey]:
    """Reverse-looks-up state.session_identities to find every live
    connection (chat and/or voice) currently bound to this CRM user."""
    session_ids = {
        sid
        for sid, identity in state.session_identities.items()
        if identity and getattr(identity, "user", None) == crm_user_email
    }
    return [key for key in _connections if key[0] in session_ids]


def is_online(crm_user_email: str) -> bool:
    return len(find_connections_for_user(crm_user_email)) > 0


async def notify(
    crm_user_email: str,
    payload: dict,
    *,
    max_wait_seconds: float = 120.0,
    poll_interval: float = 1.0,
) -> bool:
    """
    Delivers `payload` to every live connection (chat and voice both, if
    both are open) belonging to crm_user_email. If a connection is
    mid-response (busy=True), waits for it to finish before sending --
    rather than talking over an in-progress voice response or interleaving
    into a mid-stream chat answer.

    Returns True if delivered to at least one connection, False if the
    user had none open (caller should treat this as "offline" and fall
    back to the configured backup contact).
    """
    keys = find_connections_for_user(crm_user_email)
    if not keys:
        return False

    delivered = False
    for key in keys:
        rec = _connections.get(key)
        if not rec:
            continue
        waited = 0.0
        while rec.busy and waited < max_wait_seconds:
            await asyncio.sleep(poll_interval)
            waited += poll_interval
            rec = _connections.get(key)
            if not rec:
                break
        if rec:
            try:
                await rec.send(payload)
                delivered = True
            except Exception:
                # Connection likely dropped between the check and the send;
                # treat as not-delivered for this one, keep trying others.
                pass
    return delivered