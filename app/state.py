"""Mutable state shared across modules. Import this module (not its names)
so every reader sees updates made by lifespan.py / graph/nodes.py at runtime."""
import asyncio
from typing import Any, Dict, Optional

document_store: Dict[str, Dict[str, Any]] = {}
session_identities: Dict[str, Any] = {}

agent_graph = None
checkpoint_conn = None
bg_tasks = set()


def safe_create_task(coro):
    """Schedules coro and keeps a strong reference so it can't be GC'd mid-run."""
    task = asyncio.create_task(coro)
    bg_tasks.add(task)
    task.add_done_callback(bg_tasks.discard)
    return task
