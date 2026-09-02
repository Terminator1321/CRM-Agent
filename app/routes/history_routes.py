"""Session switcher support: list known sessions, replay one session's
full message history (for hydrating the chat UI after a reload or a
session switch), and merge a live-voice transcript back into the shared
history once a Realtime voice call ends.

Chat (/api/chat/stream) and Web-Speech voice (Voice/ws_voice.py) already
read/write the SAME LangGraph checkpointer keyed by session_id, so as
long as the frontend uses one shared session id across Chat and Voice,
they already see each other's turns automatically. The one gap is
Realtime/WebRTC voice (Voice/voice_routes.py): it's a separate OpenAI-side
conversation the backend never observes token-by-token, so the frontend
posts the turns it collected once a call ends via POST .../transcript
below, which folds them into the same checkpointer + session index.
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from .. import session_index
from ..graph.nodes import load_stream_history, save_stream_history

router = APIRouter()


def _role_of(msg) -> str:
    msg_type = getattr(msg, "type", None)
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"
    return msg_type or "assistant"


@router.get("/api/sessions")
async def list_sessions(limit: int = 50):
    """Session index for the switcher UI: one row per session_id with a
    title (from its first message), when it was last active, and which
    surface (chat/voice) it was last used from."""
    return {"sessions": await session_index.list_sessions(limit=limit)}


@router.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Full message history for session_id, straight from the same
    LangGraph checkpointer /api/chat/stream and the voice paths already
    write to -- so switching sessions, or opening Chat after a voice call
    (or vice versa), replays exactly what the agent itself remembers."""
    messages = await load_stream_history(session_id)
    out = []
    for m in messages:
        role = _role_of(m)
        if role not in ("user", "assistant"):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content or "")
        if not content:
            continue
        out.append({"role": role, "text": content})
    return {"session_id": session_id, "messages": out}


class TranscriptTurn(BaseModel):
    role: str  # 'user' | 'assistant'
    text: str


class PushTranscriptRequest(BaseModel):
    turns: List[TranscriptTurn]
    channel: str = "voice"


@router.post("/api/sessions/{session_id}/transcript")
async def push_transcript(session_id: str, req: PushTranscriptRequest):
    """Appends turns spoken during a live Realtime-voice call into the
    shared checkpointer, so returning to Chat (or a later voice call) has
    full context of what was said out loud. Called once by the frontend
    when a Realtime voice call ends, with whatever turns it collected."""
    new_messages = []
    preview = None
    for turn in req.turns:
        text = (turn.text or "").strip()
        if not text:
            continue
        if turn.role == "user":
            new_messages.append(HumanMessage(content=text))
            preview = preview or text
        else:
            new_messages.append(AIMessage(content=text))

    if new_messages:
        await save_stream_history(session_id, new_messages)
        await session_index.touch(
            session_id, channel=req.channel, preview_text=preview, increment=len(new_messages)
        )

    return {"success": True, "saved": len(new_messages)}


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Removes session_id from the switcher list only -- the underlying
    checkpointed conversation is left alone (matches 'clear this from my
    list' rather than a full destructive wipe)."""
    await session_index.delete(session_id)
    return {"success": True}
