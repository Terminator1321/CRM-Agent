"""
audit_log.py

Durable, queryable record of what was said in every session, for
after-the-fact review (e.g. "what did a user ask this agent to do, and
what actions did it take, in session X on date Y").

Deliberately separate from LangGraph's checkpointer: MemorySaver (used for
the agent's short-term/task-context memory in server.py) is in-process
only and vanishes on restart — fine for "remember this conversation while
it's active", wrong for "keep a record we can audit later". This writes
to a local SQLite file instead, which is durable, requires no extra
dependency (sqlite3 is stdlib), and is easy to query or export.

Swap DB_PATH for a shared/networked path (or point this at a proper
database) if multiple server processes need to log to the same place.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("audit-log")

DB_PATH = Path(__file__).parent / "audit_log.db"

_lock = threading.Lock()  # sqlite3 connections aren't thread-safe to share


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,          -- 'user' | 'assistant' | 'tool'
                content TEXT NOT NULL,
                tool_name TEXT,
                tool_args TEXT,              -- JSON, only set when role = 'tool'
                created_at TEXT NOT NULL     -- ISO 8601 UTC
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_log_session ON conversation_log(session_id, id)"
        )


def log_turn(
    session_id: str,
    role: str,
    content: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
) -> None:
    """Appends one entry. Never raises — a logging failure should not take
    down the actual conversation turn it's trying to record."""
    try:
        with _lock, _connect() as conn:
            conn.execute(
                "INSERT INTO conversation_log (session_id, role, content, tool_name, tool_args, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    role,
                    content,
                    tool_name,
                    json.dumps(tool_args) if tool_args is not None else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to write audit log entry for session '%s'", session_id)


def get_transcript(session_id: str) -> list[dict[str, Any]]:
    """Full ordered transcript for one session — what was said and every
    tool action taken, oldest first."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, tool_name, tool_args, created_at "
            "FROM conversation_log WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "tool_name": r["tool_name"],
            "tool_args": json.loads(r["tool_args"]) if r["tool_args"] else None,
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def list_sessions(since: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
    """One row per session: id, message count, first/last activity —
    a session index for an audit dashboard. `since` is an ISO date/datetime
    string (e.g. '2026-07-01'); only sessions active on or after it are
    returned."""
    query = (
        "SELECT session_id, COUNT(*) AS turn_count, "
        "MIN(created_at) AS started_at, MAX(created_at) AS last_active_at "
        "FROM conversation_log"
    )
    params: tuple = ()
    if since:
        query += " WHERE created_at >= ?"
        params = (since,)
    query += " GROUP BY session_id ORDER BY last_active_at DESC LIMIT ?"
    params = params + (limit,)

    with _lock, _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def export_json(session_id: Optional[str] = None) -> dict:
    """Builds a JSON-serializable audit export.

    - session_id given -> {"exported_at", "session_id", "turn_count", "transcript": [...]}
    - session_id omitted -> {"exported_at", "session_count", "sessions": [
          {"session_id", "turn_count", "started_at", "last_active_at", "transcript": [...]}, ...
      ]}  -- every session, each with its full transcript inlined.
    """
    exported_at = datetime.now(timezone.utc).isoformat()

    if session_id:
        transcript = get_transcript(session_id)
        return {
            "exported_at": exported_at,
            "session_id": session_id,
            "turn_count": len(transcript),
            "transcript": transcript,
        }

    sessions = list_sessions(limit=10_000)
    for s in sessions:
        s["transcript"] = get_transcript(s["session_id"])
    return {
        "exported_at": exported_at,
        "session_count": len(sessions),
        "sessions": sessions,
    }


def write_json_export(path: str, session_id: Optional[str] = None) -> str:
    """Writes export_json(...) to `path` as pretty-printed JSON and returns
    the path, for a CLI dump or a FastAPI FileResponse to hand back."""
    data = export_json(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


init_db()