"""Binds a real Frappe CRM user's own credentials to a chat session."""
from fastapi import APIRouter, Form, HTTPException
from pydantic import BaseModel

from CRM_Unified.crm_client import crm_client

from .. import state

router = APIRouter()


class SessionIdentifyRequest(BaseModel):
    session_id: str
    crm_api_key: str
    crm_api_secret: str


@router.post("/api/session/identify")
async def identify_session(req: SessionIdentifyRequest):
    """Binds session_id to the Frappe CRM user owning crm_api_key/secret, so
    every CRM tool call in that session runs under Frappe's own permission engine."""
    try:
        identity = crm_client.resolve_identity(req.crm_api_key, req.crm_api_secret)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    state.session_identities[req.session_id] = identity
    return {"authenticated": True, "user": identity.user, "roles": identity.roles}


@router.post("/api/session/logout")
async def logout_session(session_id: str = Form(...)):
    """Unbinds a session's identity; it falls back to the shared service account."""
    state.session_identities.pop(session_id, None)
    return {"success": True}
