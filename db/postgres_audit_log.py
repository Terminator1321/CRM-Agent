"""
db/postgres_audit_log.py

Postgres-backed audit trail, now written against the normalized schema
from the "Saving History in Postgres DataBase" area of the architecture
diagram (Untitled-2026-08-03-1515.excalidraw): overview / details /
tools / tools_sec / tools_details / token_details, replacing the old
flat `audit_log` table.

Public API is unchanged on purpose -- log_turn/get_transcript/
list_sessions/export_json/record_file_upload/time_tool_call keep the
same names, signatures, and return shapes as before, so server.py's
call sites don't need to change. Underneath, log_turn now fans a single
call out across overview (+ details, + tools_sec/tools_details when a
tool was involved), and get_transcript/list_sessions read it back via
joins reassembled into the same row shape the old flat table produced.

New, additive-only: log_turn() gained optional `tries`/`tokens_used`
kwargs (default None -- existing call sites that don't pass them are
unaffected) that land in the new `details` table. record_token_usage()/
get_token_details() are new, for the per-department token-budget ledger
(`token_details`) -- nothing in server.py populates that yet, since the
token-budget/RBAC flow it belongs to isn't wired up on the agent side
per the diagram; they're here ready for when it is.

Requires:
    pip install psycopg2-binary python-dotenv
    PGHOST / PGPORT / PGUSER / PGPASSWORD / PGDATABASE set in .env (see .env.example)
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool
from dotenv import load_dotenv

load_dotenv()

from db.init_db import get_connection_params

logger = logging.getLogger("audit-log")

_MIN_CONN = int(os.getenv("AUDIT_DB_POOL_MIN", "1"))
_MAX_CONN = int(os.getenv("AUDIT_DB_POOL_MAX", "10"))

_pool: Optional[pg_pool.ThreadedConnectionPool] = None


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        params = get_connection_params()
        _pool = pg_pool.ThreadedConnectionPool(_MIN_CONN, _MAX_CONN, **params)
    return _pool


@contextmanager
def _conn():
    conn = _get_pool().getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _get_pool().putconn(conn)


def ensure_session(session_id: str, user_id: Optional[str] = None) -> None:
    """Upserts the session row and bumps last_active_at. Called on every
    turn so `sessions` always reflects who owns/last touched a thread."""
    try:
        with _conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (session_id, user_id)
                VALUES (%s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    last_active_at = now(),
                    user_id = COALESCE(EXCLUDED.user_id, sessions.user_id)
                """,
                (session_id, user_id),
            )
    except Exception:
        logger.exception("Failed to upsert session '%s'", session_id)


def _ensure_tool(cur, tool_name: str) -> int:
    """Upserts a row in `tools` for `tool_name` and returns its tool_id.
    Called from inside log_turn's own connection/transaction (not its
    own _conn()) so it commits atomically with the overview/tools_sec
    rows it's part of."""
    cur.execute(
        """
        INSERT INTO tools (tool_name)
        VALUES (%s)
        ON CONFLICT (tool_name) DO UPDATE SET tool_name = EXCLUDED.tool_name
        RETURNING tool_id
        """,
        (tool_name,),
    )
    return cur.fetchone()[0]


def log_turn(
    session_id: str,
    role: str,
    content: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    user_id: Optional[str] = None,
    prompt_text: Optional[str] = None,
    tool_status: Optional[str] = None,
    error_message: Optional[str] = None,
    duration_ms: Optional[int] = None,
    tries: Optional[int] = None,
    tokens_used: Optional[int] = None,
) -> None:
    """Appends one turn to the new normalized schema. Never raises -- a
    logging failure must never take down the actual conversation turn
    it's trying to record.

    role: 'user' | 'assistant' | 'tool' | 'system'
    user_id: who prompted the underlying request (thread through from the
             originating user message to any tool calls it triggers)
    prompt_text: the user question this row is answering, if not `role='user'` itself
    duration_ms: how long this step took (LLM call latency / tool execution time)
    tries / tokens_used: optional depth info -> written to `details`
        (e.g. how many times the agent retried, tokens the user's
        request consumed). Omit if not known yet -- nothing requires them.

    Writes:
      - always: one `overview` row (prompt_text, output_text=content)
      - if tries/tokens_used given: one `details` row for that overview row
      - if tool_name given: upserts `tools`, inserts one `tools_sec` row
        (status/duration_ms), and one `tools_details` row (tool_args as
        input, content as output, error_message)
    """
    try:
        ensure_session(session_id, user_id)
        with _conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO overview (session_id, role, user_id, prompt_text, output_text)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (session_id, role, user_id, prompt_text, content),
            )
            overview_id = cur.fetchone()[0]

            if tries is not None or tokens_used is not None:
                cur.execute(
                    """
                    INSERT INTO details (overview_id, tries, tokens_used)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (overview_id) DO UPDATE SET
                        tries = COALESCE(EXCLUDED.tries, details.tries),
                        tokens_used = COALESCE(EXCLUDED.tokens_used, details.tokens_used)
                    """,
                    (overview_id, tries, tokens_used),
                )

            if tool_name:
                tool_id = _ensure_tool(cur, tool_name)
                cur.execute(
                    """
                    INSERT INTO tools_sec (overview_id, tool_id, status, duration_ms)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (overview_id, tool_id, tool_status or "success", duration_ms),
                )
                tools_sec_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO tools_details (tools_sec_id, tool_args, tool_output, error_message)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        tools_sec_id,
                        json.dumps(tool_args) if tool_args is not None else None,
                        content,
                        error_message,
                    ),
                )
    except Exception:
        logger.exception("Failed to write audit log entry for session '%s'", session_id)


@contextmanager
def time_tool_call():
    """Context manager yielding a callable that returns elapsed ms so far.

    Usage:
        with time_tool_call() as elapsed:
            result = await tool.ainvoke(args)
        audit_log.log_turn(..., duration_ms=elapsed())
    """
    start = time.perf_counter()
    yield lambda: int((time.perf_counter() - start) * 1000)


# Reassembles overview + details + tools_sec/tools/tools_details back into
# the same row shape the old flat audit_log query produced, so callers
# (get_transcript's consumers, export_json, the /api/audit/* routes)
# don't need to change.
_TRANSCRIPT_QUERY = """
    SELECT
        o.role, o.user_id, o.prompt_text, o.output_text AS content,
        t.tool_name, td.tool_args, ts.status AS tool_status,
        td.error_message, ts.duration_ms, o.created_at,
        d.tries, d.tokens_used
    FROM overview o
    LEFT JOIN tools_sec ts     ON ts.overview_id = o.id
    LEFT JOIN tools t          ON t.tool_id = ts.tool_id
    LEFT JOIN tools_details td ON td.tools_sec_id = ts.id
    LEFT JOIN details d        ON d.overview_id = o.id
    WHERE o.session_id = %s
    ORDER BY o.id ASC
"""


def get_transcript(session_id: str) -> list[dict[str, Any]]:
    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_TRANSCRIPT_QUERY, (session_id,))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def list_sessions(since: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
    query = """
        SELECT s.session_id, s.user_id, s.started_at, s.last_active_at,
               COUNT(o.id) AS turn_count
        FROM sessions s
        LEFT JOIN overview o ON o.session_id = s.session_id
    """
    params: list = []
    if since:
        query += " WHERE s.last_active_at >= %s"
        params.append(since)
    query += " GROUP BY s.session_id ORDER BY s.last_active_at DESC LIMIT %s"
    params.append(limit)

    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def export_json(session_id: Optional[str] = None) -> dict:
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
    data = export_json(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return path


# ---------------------------------------------------------------------
# Tool success/failure summary -- "did it work or fail how many times"
# (TOOLS_SEC's stated purpose), derived from tools_sec rather than a
# running counter so it can't drift out of sync with the raw log.
# ---------------------------------------------------------------------

def tool_stats(session_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Per-tool success/fail/not_found counts, optionally scoped to one
    session. Backs a 'which tools worked, which didn't, how often'
    view -- the thing TOOLS_SEC exists for."""
    query = """
        SELECT t.tool_name,
               COUNT(*) FILTER (WHERE ts.status = 'success')   AS success_count,
               COUNT(*) FILTER (WHERE ts.status = 'error')     AS error_count,
               COUNT(*) FILTER (WHERE ts.status = 'not_found') AS not_found_count,
               COUNT(*)                                        AS total_calls
        FROM tools_sec ts
        JOIN tools t ON t.tool_id = ts.tool_id
        JOIN overview o ON o.id = ts.overview_id
    """
    params: list = []
    if session_id:
        query += " WHERE o.session_id = %s"
        params.append(session_id)
    query += " GROUP BY t.tool_name ORDER BY total_calls DESC"

    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# Token budget ledger (token_details) -- per-department usage/allotment
# for the Agent and TTS. Nothing populates this yet (the token-check/
# RBAC flow it belongs to isn't wired up on the agent side per the
# diagram) -- these are here ready for when it is.
# ---------------------------------------------------------------------

def record_token_usage(
    department: str,
    agent_tokens_used: int = 0,
    tts_tokens_used: int = 0,
    agent_tokens_allotted: Optional[int] = None,
    tts_tokens_allotted: Optional[int] = None,
) -> None:
    """Adds to a department's running token usage, upserting the row if
    it doesn't exist yet. Pass *_allotted only when (re)setting a
    department's budget -- omitted, the existing allotment is kept."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO token_details (
                department, agent_tokens_used, tts_tokens_used,
                agent_tokens_allotted, tts_tokens_allotted
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (department) DO UPDATE SET
                agent_tokens_used = token_details.agent_tokens_used + EXCLUDED.agent_tokens_used,
                tts_tokens_used = token_details.tts_tokens_used + EXCLUDED.tts_tokens_used,
                agent_tokens_allotted = COALESCE(EXCLUDED.agent_tokens_allotted, token_details.agent_tokens_allotted),
                tts_tokens_allotted = COALESCE(EXCLUDED.tts_tokens_allotted, token_details.tts_tokens_allotted)
            """,
            (department, agent_tokens_used, tts_tokens_used, agent_tokens_allotted, tts_tokens_allotted),
        )


def get_token_details(department: Optional[str] = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM token_details"
    params: list = []
    if department:
        query += " WHERE department = %s"
        params.append(department)
    query += " ORDER BY department"

    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# File upload metadata (paired with storage/s3_storage.py) -- unchanged,
# file_uploads wasn't touched by the schema redesign.
# ---------------------------------------------------------------------

def record_file_upload(
    session_id: Optional[str],
    user_id: Optional[str],
    original_filename: str,
    content_type: str,
    file_size_bytes: int,
    checksum_sha256: str,
    upload_kind: str,
    s3_bucket: str,
    s3_key: str,
    s3_region: str,
    s3_version_id: Optional[str] = None,
    extracted_metadata: Optional[dict] = None,
    status: str = "processed",
) -> str:
    """Inserts one file_uploads row and returns its id (uuid string).
    Call this right after storage.s3_storage.upload_file() succeeds."""
    if session_id:
        ensure_session(session_id, user_id)
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO file_uploads (
                session_id, user_id, original_filename, content_type,
                file_size_bytes, checksum_sha256, upload_kind, status,
                s3_bucket, s3_key, s3_region, s3_version_id, extracted_metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                session_id,
                user_id,
                original_filename,
                content_type,
                file_size_bytes,
                checksum_sha256,
                upload_kind,
                status,
                s3_bucket,
                s3_key,
                s3_region,
                s3_version_id,
                json.dumps(extracted_metadata or {}),
            ),
        )
        row_id = cur.fetchone()[0]
    return str(row_id)


def list_file_uploads(session_id: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
    query = "SELECT * FROM file_uploads"
    params: list = []
    if session_id:
        query += " WHERE session_id = %s"
        params.append(session_id)
    query += " ORDER BY uploaded_at DESC LIMIT %s"
    params.append(limit)

    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]