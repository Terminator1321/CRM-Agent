"""Startup/shutdown hooks: Postgres audit-log schema + persistent LangGraph checkpointer."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from db.init_db import apply_schema

from . import state
from .graph.nodes import build_agent_graph
from .logging_setup import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup and once at shutdown. CRM tools load synchronously
    at import time (agent_setup.py); this only handles Postgres + the checkpointer."""
    try:
        apply_schema()
        logger.info("Postgres audit-log schema ready.")
    except Exception:
        logger.exception(
            "Could not apply Postgres schema -- check PGHOST/PGPORT/PGUSER/"
            "PGPASSWORD/PGDATABASE in .env. Audit logging will fail until this is fixed."
        )

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    # isolation_level=None = autocommit, so writes are immediately visible to
    # subsequent reads on the same connection.
    state.checkpoint_conn = await aiosqlite.connect("stream_history.sqlite", isolation_level=None)
    try:
        saver = AsyncSqliteSaver(state.checkpoint_conn)
        await saver.setup()
        state.agent_graph = build_agent_graph(checkpointer=saver)
        logger.info("AsyncSqliteSaver persistent memory ready.")
        yield
    finally:
        await state.checkpoint_conn.close()
        state.checkpoint_conn = None
