"""Session index for the frontend's session switcher.

Deliberately NOT another new database: it reuses the same aiosqlite
connection (state.checkpoint_conn) that the LangGraph checkpointer
already keeps open in stream_history.sqlite for /api/chat/stream, ws_voice
and the Realtime-voice transcript path. That connection is autocommit
(isolation_level=None, see app/lifespan.py), so writes here are visible
immediately and never block on a competing transaction.

This table only ever stores UI metadata (title/last channel/counts) --
the actual conversation content lives in the checkpointer itself and is
read back via app.graph.nodes.load_stream_history. Losing this table
(e.g. by deleting stream_history.sqlite) only loses the switcher's
labels, never the conversation.
"""
import datetime as dt
from typing import Optional

from . import state

_table_ready = False


async def _ensure_table() -> bool:
    """Creates the index table on first use. Returns False (and does
    nothing else) if the checkpointer connection isn't up yet -- callers
    treat that as 'indexing unavailable right now', never a hard error."""
    global _table_ready
    if _table_ready:
        return True
    if state.checkpoint_conn is None:
        return False
    await state.checkpoint_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_index (
            session_id    TEXT PRIMARY KEY,
            title         TEXT,
            last_channel  TEXT NOT NULL DEFAULT 'chat',
            message_count INTEGER NOT NULL DEFAULT 0,
            updated_at    TEXT NOT NULL
        )
        """
    )
    try:
        await state.checkpoint_conn.commit()
    except Exception:
        pass
    _table_ready = True
    return True


def _short_title(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "New session"
    return text[:60] + "…" if len(text) > 60 else text


async def touch(session_id: str, channel: str = "chat", preview_text: Optional[str] = None, increment: int = 0) -> None:
    """Upserts the index row for session_id. Never raises -- indexing a
    session must never take down the actual chat/voice turn it's tracking."""
    if not session_id:
        return
    try:
        if not await _ensure_table():
            return
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        title = _short_title(preview_text) if preview_text else None
        await state.checkpoint_conn.execute(
            """
            INSERT INTO session_index (session_id, title, last_channel, message_count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                title         = COALESCE(session_index.title, excluded.title),
                last_channel  = excluded.last_channel,
                message_count = session_index.message_count + ?,
                updated_at    = excluded.updated_at
            """,
            (session_id, title, channel, increment, now, increment),
        )
        try:
            await state.checkpoint_conn.commit()
        except Exception:
            pass
    except Exception:
        pass


async def list_sessions(limit: int = 50) -> list:
    try:
        if not await _ensure_table():
            return []
        cursor = await state.checkpoint_conn.execute(
            "SELECT session_id, title, last_channel, message_count, updated_at "
            "FROM session_index ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "session_id": r[0],
                "title": r[1] or "New session",
                "channel": r[2],
                "message_count": r[3],
                "updated_at": r[4],
            }
            for r in rows
        ]
    except Exception:
        return []


async def delete(session_id: str) -> None:
    try:
        if not await _ensure_table():
            return
        await state.checkpoint_conn.execute(
            "DELETE FROM session_index WHERE session_id = ?", (session_id,)
        )
        try:
            await state.checkpoint_conn.commit()
        except Exception:
            pass
    except Exception:
        pass
