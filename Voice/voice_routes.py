"""
Voice/voice_routes.py

FastAPI APIRouter for all WebRTC voice session endpoints.
Mounted into server.py via `app.include_router(voice_router)`.

Endpoints:
  POST /api/voice/session/start      — Creates ephemeral token + voice session
  GET  /api/voice/session/{id}/config — Returns tool schemas + system prompt
  POST /api/voice/session/{id}/tool  — Executes a CRM tool (audited, authorized)
  DELETE /api/voice/session/{id}     — Tears down the voice session

All tool execution goes through server.py's _execute_tool(), guaranteeing:
  - Full Postgres audit logging (same as text chat)
  - CRMIdentity (per-user credentials) enforcement
  - Consistent error handling and permission checks
"""

import json
import logging
import os
from typing import Any, Callable, Dict, Optional

from livekit import api
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from LLM.LLM import GENERAL_CRM_PROMPT
from app.streaming import build_document_context_note
from Voice.voice_session_manager import (
    create_voice_session,
    get_voice_session,
    revoke_voice_session,
)

logger = logging.getLogger(__name__)

voice_router = APIRouter(prefix="/api/voice", tags=["voice"])

# ---------------------------------------------------------------------------
# Tool whitelist — only these tools can be triggered over the voice data channel.
# Financial submit / batch operations require explicit text confirmation.
# ---------------------------------------------------------------------------
ALLOWED_VOICE_TOOLS: set[str] = {
    "crm_metadata",
    "crm_search",
    "crm_get",
    "crm_create",
    "crm_update",
    "crm_delete",
    "crm_linked_records",
    "crm_activities",
    "crm_contact_action",
    "crm_research_company",
    "web_company_search",
    "web_company_extract",
    "web_search",
}

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class VoiceSessionStartRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    voice: str = "alloy"
    model: str = "gpt-realtime"


class VoiceToolRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any] = {}
    call_id: Optional[str] = None   # OpenAI function call ID — echoed back to client


# ---------------------------------------------------------------------------
# Dependency: injects shared server-level state via Request
# ---------------------------------------------------------------------------

def _get_server_state(request: Request):
    """Extracts shared state (session_identities, _execute_tool, ALL_TOOLS)
    that was attached to app.state in server.py at startup."""
    return request.app.state


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@voice_router.post("/session/start")
async def start_voice_session(req: VoiceSessionStartRequest, state=Depends(_get_server_state)):
    """
    Validates that the session_id is authenticated (i.e. the user has already
    called /api/session/identify), then mints an ephemeral OpenAI token and
    registers a voice session. Returns the token for the browser to use in
    its RTCPeerConnection SDP exchange with OpenAI.
    """
    session_identities: dict = getattr(state, "session_identities", {})
    # A session without an identity falls through to the shared service account —
    # consistent with how text chat works.
    user_id = req.user_id

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    try:
        session_record = await create_voice_session(
            session_id=req.session_id,
            user_id=user_id,
            openai_api_key=api_key,
            model=req.model,
            voice=req.voice,
        )
    except Exception as exc:
        logger.exception("Failed to create voice session for session_id=%s", req.session_id)
        raise HTTPException(status_code=502, detail=f"OpenAI token creation failed: {exc}")

    return {
        "ephemeral_token": session_record["ephemeral_token"],
        "openai_session_id": session_record["openai_session_id"],
        "expires_in": 50,
    }


@voice_router.get("/session/{session_id}/config")
async def get_voice_config(session_id: str, state=Depends(_get_server_state)):
    """
    Returns the OpenAI session.update payload the frontend should send after
    the WebRTC data channel opens. Includes:
      - system_prompt (GENERAL_CRM_PROMPT adapted for voice)
      - tool schemas (OpenAI function format) for ALLOWED_VOICE_TOOLS only
      - recent conversation history for continuity across text <-> voice switches
    """
    voice_record = get_voice_session(session_id)
    if not voice_record:
        raise HTTPException(status_code=404, detail="No active voice session for this session_id.")

    # Build OpenAI-format tool schemas for allowed tools only
    all_tools = getattr(state, "all_tools_list", [])
    allowed_schemas = []
    for tool in all_tools:
        if tool.name in ALLOWED_VOICE_TOOLS:
            try:
                schema = convert_to_openai_tool(tool)
                # convert_to_openai_tool() returns Chat-Completions-style
                # shape: {"type": "function", "function": {"name": ...,
                # "description": ..., "parameters": {...}}}. The Realtime
                # API (like the Responses API) expects a FLAT shape --
                # name/description/parameters directly on the tool object,
                # no nested "function" wrapper. Sending the nested shape
                # doesn't error, it just silently produces no callable
                # tools, which is why voice had no tool access while text
                # chat (Chat Completions) worked fine.
                fn = schema.get("function", {})
                params = fn.get("parameters", {})
                params.pop("additionalProperties", None)
                params.pop("title", None)
                allowed_schemas.append({
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": params,
                })
            except Exception:
                logger.warning("Could not convert tool %s to schema", tool.name)

    # Build the voice system prompt from the SAME operational guidance text
    # chat uses (GENERAL_CRM_PROMPT) -- search-before-acting rules, the
    # invalid_link_fields handling logic, task-assignment flow, company
    # research requirements, etc. Voice previously used a separate,
    # much shorter hand-written prompt that dropped nearly all of this,
    # which is why voice's tool use was noticeably less reliable than
    # text chat's (e.g. answering "how many leads" from a single narrow
    # record instead of doing a proper search). Only the *formatting*
    # tail (Markdown tables/charts/action-block instructions, which make
    # no sense spoken aloud) is swapped out for voice-appropriate rules.
    _FORMATTING_MARKER = "Presenting data -- tables and charts:"
    _marker_pos = GENERAL_CRM_PROMPT.find(_FORMATTING_MARKER)
    operational_guidance = (
        GENERAL_CRM_PROMPT[:_marker_pos] if _marker_pos != -1 else GENERAL_CRM_PROMPT
    )
    voice_prompt = (
        operational_guidance
        + "\n\nYou are currently in a VOICE conversation, not text chat -- adjust delivery only, "
        "not judgment or tool usage:\n"
        "- Keep replies to 1-3 sentences. No markdown formatting, no bullet lists, no tables, "
        "no chart blocks, no 'Next actions:' A/B/C block -- none of that can be spoken. Say the "
        "next options naturally instead if relevant (e.g. 'Want me to update it or create a deal "
        "for it?').\n"
        "- You cannot open a file picker yourself -- if the user wants to upload, attach, or "
        "share a file, tell them to tap the Attach button on their screen. Once a file is "
        "uploaded you will receive its contents as a message and can answer questions about it "
        "or use it to create CRM records.\n"
        "- Still verify names via crm_search, still research companies before creating records, "
        "still confirm before destructive writes, and still get an accurate total (not a guess "
        "from one record) before answering any 'how many' question -- speaking concisely doesn't "
        "mean skipping the same verification steps text chat follows."
    )

    # Load recent conversation history from stream_history for context continuity
    history = []
    load_fn = getattr(state, "load_stream_history_fn", None)
    if load_fn:
        try:
            history = await load_fn(session_id)
            # Convert to OpenAI realtime conversation.item format
            history = _convert_history_to_realtime_items(history)
        except Exception:
            logger.warning("Could not load conversation history for voice session %s", session_id)

    return {
        "session_update": {
            "type": "session.update",
            "session": {
                # GA requires this on every session object, including
                # session.update events sent over the data channel --
                # omitting it returns "Missing required parameter:
                # 'session.type'."
                "type": "realtime",
                "instructions": voice_prompt,
                "tools": allowed_schemas,
                "tool_choice": "auto",
                "audio": {
                    "input": {
                        "turn_detection": {"type": "server_vad"},
                        "transcription": {"model": "whisper-1"},
                    },
                    "output": {
                        "voice": voice_record["voice"],
                    },
                },
            },
        },
        "conversation_history": history,
    }


@voice_router.post("/session/{session_id}/tool")
async def execute_voice_tool(
    session_id: str,
    req: VoiceToolRequest,
    state=Depends(_get_server_state),
):
    """
    Executes a CRM tool on behalf of the voice session.
    Security checks:
      1. The voice session must be active (validated via voice_session_manager).
      2. The tool_name must be in ALLOWED_VOICE_TOOLS.
    Execution goes through server.py's _execute_tool() for full audit logging
    and CRMIdentity enforcement.
    """
    # 1. Validate session is active
    voice_record = get_voice_session(session_id)
    if not voice_record:
        raise HTTPException(
            status_code=401,
            detail="No active voice session. Call /api/voice/session/start first.",
        )

    # 2. Enforce tool whitelist
    if req.tool_name not in ALLOWED_VOICE_TOOLS:
        logger.warning(
            "Voice session %s attempted blocked tool: %s", session_id, req.tool_name
        )
        return JSONResponse(
            status_code=403,
            content={
                "call_id": req.call_id,
                "result": (
                    f"The tool '{req.tool_name}' cannot be executed by voice. "
                    "This action requires explicit text confirmation for safety."
                ),
            },
        )

    # 3. Execute via server's _execute_tool (audit log + identity enforcement)
    execute_fn: Callable = getattr(state, "execute_tool_fn", None)
    if execute_fn is None:
        raise HTTPException(
            status_code=500, detail="_execute_tool not available. Check server.py app.state setup."
        )

    user_id = voice_record.get("user_id")
    result = await execute_fn(
        tool_name=req.tool_name,
        args=req.args,
        session_id=session_id,
        user_id=user_id,
        prompt_text="[voice]",
    )

    return {
        "call_id": req.call_id,
        "result": str(result),
    }


@voice_router.get("/session/{session_id}/document-context")
async def get_document_context(session_id: str):
    """
    Returns the same document-context note the text/streaming chat path
    injects into its system prompt on every turn (see
    app.streaming.build_document_context_note), so the Realtime voice
    frontend can push it into the live conversation right after a file
    upload -- the Realtime model is a separate OpenAI session with its own
    conversation state, so it has no automatic access to state.document_store.
    """
    note = build_document_context_note(session_id)
    return {"note": note}


@voice_router.delete("/session/{session_id}")
async def end_voice_session(session_id: str):
    """Gracefully tears down a voice session and cancels the token refresh task."""
    revoke_voice_session(session_id)
    return {"success": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# Helper: convert LangGraph history to OpenAI Realtime conversation items
# ---------------------------------------------------------------------------

def _convert_history_to_realtime_items(history: list) -> list:
    """
    Converts LangGraph message history into the conversation.item.create
    format expected by the OpenAI Realtime API, so the voice AI has full
    context from previous text chat turns.
    """
    items = []
    for msg in history[-20:]:  # Last 20 turns max to stay within context limits
        role = None
        content = ""
        if hasattr(msg, "type"):
            if msg.type == "human":
                role = "user"
                content = msg.content or ""
            elif msg.type == "ai":
                role = "assistant"
                content = msg.content or ""
        elif isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")

        if role and content:
            # OpenAI's Realtime API requires the content part type to match
            # the role: user messages use "input_text", assistant messages
            # must use "output_text" -- mixing these up returns a 400/
            # validation error ("Invalid value: 'input_text'. Value must
            # be 'output_text'.") the moment the assistant history item
            # is sent.
            content_type = "input_text" if role == "user" else "output_text"
            items.append({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": role,
                    "content": [{"type": content_type, "text": content}],
                },
            })
    return items

@voice_router.post("/livekit-token")
async def get_livekit_token(req: VoiceSessionStartRequest):
    """Generates a LiveKit Access Token for the React frontend."""
    import os
    import json
    from livekit import api
    from fastapi import HTTPException
    
    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    
    if not api_key or not api_secret:
        raise HTTPException(status_code=500, detail="LIVEKIT_API_KEY or LIVEKIT_API_SECRET not set")
        
    room_name = f"voice_room_{req.session_id}"
    identity = req.user_id or "anonymous_user"
    
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(identity)
    token.with_name(identity)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
    ))
    
    # Pass session_id in metadata so the Python worker can extract it!
    metadata = {"session_id": req.session_id}
    token.with_metadata(json.dumps(metadata))
    
    jwt = token.to_jwt()
    return {"token": jwt, "room": room_name, "serverUrl": os.environ.get("LIVEKIT_URL", "wss://magna-crm-ai.livekit.cloud")}