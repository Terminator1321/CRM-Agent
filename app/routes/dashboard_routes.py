"""CRM dashboard route for the React frontend."""
import asyncio

from fastapi import APIRouter, HTTPException

from ..dashboard import build_dashboard
from ..logging_setup import logger

router = APIRouter()


@router.get("/api/dashboard")
async def dashboard():
    """Returns live CRM dashboard data."""
    try:
        return await asyncio.to_thread(build_dashboard)
    except (PermissionError, RuntimeError) as exc:
        logger.exception("Unable to build CRM dashboard")
        raise HTTPException(status_code=502, detail=f"CRM dashboard data is unavailable: {exc}") from exc
