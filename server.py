"""Entrypoint: assembles the FastAPI app from the app/ package and runs it."""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import agent_setup, config, state, tool_exec
from app.graph.nodes import load_stream_history, save_stream_history
from app.lifespan import lifespan
from app.logging_setup import logger
from app.routes import audit_routes, chat_routes, dashboard_routes, document_routes, health_routes, session_routes
from app.streaming import stream_agent_turn
from Voice.voice_routes import voice_router
from Voice.ws_voice import register_voice_ws

app = FastAPI(title="MagmaAssistance Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_router)
app.include_router(dashboard_routes.router)
app.include_router(session_routes.router)
app.include_router(document_routes.router)
app.include_router(chat_routes.router)
app.include_router(audit_routes.router)
app.include_router(health_routes.router)

register_voice_ws(app, stream_agent_turn, agent_setup.assistant.tts, logger, load_stream_history, save_stream_history)

# Exposed on app.state so Voice/voice_routes.py can reach shared state without circular imports.
app.state.session_identities = state.session_identities
app.state.all_tools_list = agent_setup.ALL_TOOLS
app.state.execute_tool_fn = tool_exec.execute_tool
app.state.load_stream_history_fn = load_stream_history


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8050))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
