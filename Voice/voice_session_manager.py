"""
Voice/voice_session_manager.py

Manages the lifecycle of ephemeral OpenAI tokens for WebRTC voice sessions.
Decoupled from server.py routing so it can be independently tested and extended.

Ephemeral tokens expire after 60 seconds on OpenAI's side; we proactively
refresh them at 50 seconds so sessions never die mid-conversation.
"""

import asyncio
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# OpenAI endpoint for ephemeral token generation
_OPENAI_REALTIME_SESSIONS_URL = "https://api.openai.com/v1/realtime/client_secrets"
_TOKEN_TTL_SECONDS = 50  # Refresh before the 60s OpenAI expiry

# In-memory registry: session_id -> session metadata dict
_active_sessions: dict[str, dict] = {}


async def create_voice_session(
    session_id: str,
    user_id: Optional[str],
    openai_api_key: str,
    model: str = "gpt-4o-realtime-preview-2024-12-17",
    voice: str = "alloy",
) -> dict:
    """
    Creates an ephemeral token from OpenAI and registers the voice session.
    Returns the session metadata dict including the ephemeral_token.
    """
    token_data = await _fetch_ephemeral_token(openai_api_key, model, voice)

    session_record = {
        "session_id": session_id,
        "user_id": user_id,
        "ephemeral_token": token_data.get("client_secret", {}).get("value"),
        "openai_session_id": token_data.get("id"),
        "model": model,
        "voice": voice,
        "created_at": time.time(),
        "openai_api_key": openai_api_key,  # stored for refresh
        "refresh_task": None,
    }

    # Cancel any previous refresh task for this session_id
    existing = _active_sessions.get(session_id)
    if existing and existing.get("refresh_task"):
        existing["refresh_task"].cancel()

    # Schedule proactive token refresh
    refresh_task = asyncio.create_task(
        _schedule_refresh(session_id, openai_api_key, model, voice)
    )
    session_record["refresh_task"] = refresh_task

    _active_sessions[session_id] = session_record
    logger.info("Voice session created for session_id=%s user_id=%s", session_id, user_id)
    return session_record


def get_voice_session(session_id: str) -> Optional[dict]:
    """Returns the active session record or None if not found / expired."""
    record = _active_sessions.get(session_id)
    if not record:
        return None
    # Treat sessions older than 2 * TTL as dead (refresh should have kept it alive)
    age = time.time() - record["created_at"]
    if age > _TOKEN_TTL_SECONDS * 2:
        logger.warning("Voice session %s appears stale (age=%.0fs), evicting.", session_id, age)
        revoke_voice_session(session_id)
        return None
    return record


def revoke_voice_session(session_id: str) -> None:
    """Tears down a voice session and cancels its refresh task."""
    record = _active_sessions.pop(session_id, None)
    if record:
        task = record.get("refresh_task")
        if task and not task.done():
            task.cancel()
        logger.info("Voice session revoked for session_id=%s", session_id)


async def _fetch_ephemeral_token(api_key: str, model: str, voice: str) -> dict:
    """Calls OpenAI's /realtime/sessions endpoint to create an ephemeral token."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            _OPENAI_REALTIME_SESSIONS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"session": {"modalities": ["audio", "text"], "model": model, "type": "realtime", "voice": voice}},
        )
        if resp.status_code >= 400:
            logger.error(f"OpenAI error {resp.status_code}: {resp.text}")

        resp.raise_for_status()
        return resp.json()


async def _schedule_refresh(session_id: str, api_key: str, model: str, voice: str):
    """
    Waits _TOKEN_TTL_SECONDS then silently refreshes the ephemeral token
    so the session remains live without any client interruption.
    """
    try:
        await asyncio.sleep(_TOKEN_TTL_SECONDS)
        if session_id not in _active_sessions:
            return  # session was revoked while we slept

        logger.info("Proactively refreshing ephemeral token for session_id=%s", session_id)
        token_data = await _fetch_ephemeral_token(api_key, model, voice)
        record = _active_sessions.get(session_id)
        if record:
            record["ephemeral_token"] = token_data.get("client_secret", {}).get("value")
            record["openai_session_id"] = token_data.get("id")
            record["created_at"] = time.time()

            # Schedule the next refresh
            next_task = asyncio.create_task(
                _schedule_refresh(session_id, api_key, model, voice)
            )
            record["refresh_task"] = next_task

    except asyncio.CancelledError:
        pass  # Session was explicitly revoked — expected.
    except Exception:
        logger.exception("Failed to refresh ephemeral token for session_id=%s", session_id)
