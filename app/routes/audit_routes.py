"""Read-only routes over the Postgres audit log."""
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import db.postgres_audit_log as audit_log

router = APIRouter()


@router.get("/api/audit/sessions")
def list_audit_sessions(since: str = None, limit: int = 100):
    """Session index: one row per session_id with turn count and first/last activity."""
    return {"sessions": audit_log.list_sessions(since=since, limit=limit)}


@router.get("/api/audit/sessions/{session_id}")
def get_audit_transcript(session_id: str):
    """Full ordered transcript for one session."""
    transcript = audit_log.get_transcript(session_id)
    if not transcript:
        raise HTTPException(status_code=404, detail=f"No audit log found for session '{session_id}'")
    return {"session_id": session_id, "turn_count": len(transcript), "transcript": transcript}


@router.get("/api/audit/export")
def export_audit_json(session_id: str = None):
    """Downloads the audit log as a .json file, one session or all sessions."""
    export_dir = "audit_exports"
    os.makedirs(export_dir, exist_ok=True)
    filename = f"audit_{session_id}.json" if session_id else "audit_all_sessions.json"
    path = os.path.join(export_dir, filename)
    audit_log.write_json_export(path, session_id=session_id)
    return FileResponse(path, media_type="application/json", filename=filename)
