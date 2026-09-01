import asyncio
import base64
import os
import re
import shutil
import logging
import json
import sys
import ast
import httpx
import asyncio
from datetime import datetime, timezone
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, BaseMessage, trim_messages, RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any, Sequence, Dict, Union, Callable, Annotated, TypedDict, Literal
import requests
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from LLM.LLM import LLM
import db.postgres_audit_log as audit_log
from db.init_db import apply_schema
from storage import s3_storage
from CRM_Unified.crm_client import crm_client, CRMIdentity, use_identity

# Helper function to convert messages to dictionary format for the OpenAI API
def convert_message_to_dict(message):
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, ToolMessage):
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    elif isinstance(message, AIMessage):
        d = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            d["tool_calls"] = []
            for tc in message.tool_calls:
                d["tool_calls"].append({
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": json.dumps(tc.get("args") or {})
                    }
                })
        elif hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs:
            d["tool_calls"] = message.additional_kwargs["tool_calls"]
        return d
    elif isinstance(message, dict):
        return message
    else:
        role = getattr(message, "type", "user")
        if role == "ai":
            role = "assistant"
        return {"role": role, "content": getattr(message, "content", str(message))}

def _clean_schema_for_openai(schema: dict) -> dict:
    """Cleans up tool JSON schemas so OpenAI API does not throw 400 errors."""
    if not isinstance(schema, dict):
        return schema
    
    cleaned = schema.copy()
    # Remove fields that cause OpenAI strict parameter 400 validation failures
    cleaned.pop("additionalProperties", None)
    cleaned.pop("title", None)

    if "properties" in cleaned and isinstance(cleaned["properties"], dict):
        new_props = {}
        for prop_key, prop_val in cleaned["properties"].items():
            if isinstance(prop_val, dict):
                new_props[prop_key] = _clean_schema_for_openai(prop_val)
            else:
                new_props[prop_key] = prop_val
        cleaned["properties"] = new_props

    if "items" in cleaned and isinstance(cleaned["items"], dict):
        cleaned["items"] = _clean_schema_for_openai(cleaned["items"])

    return cleaned


class OpenAIChatModel(BaseChatModel):
    model_name: str
    temperature: float
    api_key: str
    base_url: str
    bound_tools: Optional[List[Any]] = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        api_messages = [convert_message_to_dict(msg) for msg in messages]
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        env_openai_key = os.environ.get("OPENAI_API_KEY")
        key = openrouter_key or self.api_key or env_openai_key

        is_openrouter = bool(openrouter_key) or (key and str(key).startswith("sk-or-v1-")) or "openrouter.ai" in str(self.base_url)

        if is_openrouter:
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8050",
                "X-Title": "MagmaAssistance",
            }
            target_url = "https://openrouter.ai/api/v1/chat/completions"
            model_name = self.model_name if "/" in self.model_name else f"openai/{self.model_name}"
        else:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            target_url = self.base_url
            model_name = self.model_name

        data = {
            "model": model_name,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": LLM_MAX_TOKENS,
        }

        if self.bound_tools:
            data["tools"] = self.bound_tools

        # If the model hits LLM_MAX_TOKENS mid-reply (finish_reason ==
        # "length"), ask it to continue from where it left off instead
        # of returning the truncated text as if it were the full
        # answer. Tool-call replies are never continued this way --
        # a tool call is either complete or it isn't, and re-prompting
        # risks a duplicate/garbled call.
        content = ""
        tool_calls = []
        message_data = {}
        for round_number in range(MAX_COMPLETION_ROUNDS + 1):
            try:
                response = requests.post(target_url, json=data, headers=headers, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
            except requests.exceptions.Timeout:
                logger.error(
                    "LLM API call timed out after %ss with no response (%s)",
                    LLM_REQUEST_TIMEOUT_SECONDS, target_url,
                )
                raise RuntimeError(
                    f"The AI model didn't respond within {int(LLM_REQUEST_TIMEOUT_SECONDS)}s. "
                    "Please try again."
                )

            if not response.ok:
                logger.error(f"LLM API Rejected Request ({response.status_code}): {response.text}")

            response.raise_for_status()
            res_json = response.json()

            choice = res_json["choices"][0]
            message_data = choice["message"]
            finish_reason = choice.get("finish_reason")

            piece = message_data.get("content") or ""
            content += piece

            if "tool_calls" in message_data:
                for tc in message_data["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except Exception:
                        args = {}
                    tool_calls.append({
                        "name": tc["function"]["name"],
                        "args": args,
                        "id": tc.get("id"),
                    })

            if finish_reason != "length" or tool_calls or round_number == MAX_COMPLETION_ROUNDS:
                if finish_reason == "length":
                    logger.warning(
                        "LLM reply still truncated after %d continuation round(s); "
                        "returning what we have.", round_number
                    )
                break

            # Continue the truncated reply: replay what the model said so
            # far as an assistant turn, then ask it to pick up exactly
            # where it stopped.
            data = dict(data)
            data["messages"] = api_messages + [
                {"role": "assistant", "content": piece},
                {"role": "user", "content": "Continue exactly where you left off. Do not repeat any text or restart the answer."},
            ]

        ai_message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _llm_type(self) -> str:
        return "openai-chat-model"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type[BaseModel], Callable, Any]],
        **kwargs: Any,
    ) -> "OpenAIChatModel":
        from langchain_core.utils.function_calling import convert_to_openai_tool
        
        formatted_tools = []
        for t in tools:
            formatted = convert_to_openai_tool(t)
            # Sanitize tool parameter schema to fix 400 Bad Request
            if "function" in formatted and "parameters" in formatted["function"]:
                formatted["function"]["parameters"] = _clean_schema_for_openai(
                    formatted["function"]["parameters"]
                )
            formatted_tools.append(formatted)

        return OpenAIChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            base_url=self.base_url,
            bound_tools=formatted_tools,
        )

# Add model property to LLM class before importing Main/VoiceAssistant
@property
def get_model(self):
    return OpenAIChatModel(
        model_name=self.model_name,
        temperature=self.temperature,
        api_key=self.api_key,
        base_url=self.base_url
    )

LLM.model = get_model

from Main import VoiceAssistant

from CRM_Unified.tools import CRM_TOOLS
from CRM_Unified.tool_rag import ToolRAG
from web.web_tool import WEB_TOOLS

ALL_TOOLS = [*CRM_TOOLS, *WEB_TOOLS]
ALL_REQUIRED_FIELDS: dict = {}
ALL_FIELD_PARSERS: dict = {}

# Configure logging
# Global log level: INFO for most modules; DEBUG for voice pipeline modules
# so STT/TTS timing, byte counts and WebSocket lifecycle are fully visible.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
for _voice_logger in ("ws_voice", "openai-stt", "openai-tts"):
    logging.getLogger(_voice_logger).setLevel(logging.DEBUG)

logger = logging.getLogger("agent-server")

# ---------------------------------------------------------------------
# LangSmith tracing (optional -- no-op if LANGCHAIN_API_KEY isn't set)
# ---------------------------------------------------------------------
# LangChain/LangGraph runnables (assistant.llm.model, text_chain,
# agent_graph) are auto-instrumented by LangSmith's callback handler the
# moment LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are present in
# the environment -- no code changes needed for those. This block just:
#   1. sets a sane default project name (so traces aren't dumped into
#      LangSmith's "default" project) without clobbering one you already
#      set in .env, and
#   2. logs plainly at startup whether tracing is actually on, so a
#      missing/typo'd key fails loud instead of silently not tracing.
# LLM.py's raw `requests` calls to OpenAI Vision (extract_po_data_from_
# document, extract_document_text, ask_about_document) do NOT go through
# LangChain, so they are NOT auto-traced -- see the @traceable decorators
# added on those functions instead, which report to the same project.
#
# Add to your .env to enable:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=ls__...
#   LANGCHAIN_PROJECT=magma-assistance      # optional, defaults below
#   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com   # optional
os.environ.setdefault("LANGCHAIN_PROJECT", "magma-assistance")
LANGSMITH_TRACING_ENABLED = (
    os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    and bool(os.environ.get("LANGCHAIN_API_KEY"))
)
if LANGSMITH_TRACING_ENABLED:
    logger.info(
        "LangSmith tracing ENABLED -- project='%s', endpoint='%s'",
        os.environ.get("LANGCHAIN_PROJECT"),
        os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    )
else:
    logger.info(
        "LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true and "
        "LANGCHAIN_API_KEY in .env to enable)."
    )

# `WHISPER_MODEL` now names an OpenAI hosted transcription model (STT),
# not a local Whisper checkpoint -- e.g. "gpt-4o-mini-transcribe" or
# "whisper-1". `TTS_VOICE` must be one of OpenAI's TTS voices (alloy,
# ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse).
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "gpt-4o-mini-transcribe")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
TTS_VOICE = os.environ.get("TTS_VOICE", "alloy")
# Neither OpenAI nor (especially) OpenRouter can be trusted to pick a
# sensible default completion length on their own -- OpenRouter in
# particular will silently cap some routed providers far below the
# model's real context window when `max_tokens` is omitted. Without an
# explicit value here, long replies get cut off mid-sentence with
# finish_reason="length" and no error, and the old code below wasn't
# checking finish_reason at all, so the truncated text just went out
# as if it were complete. Set high on purpose; MAX_COMPLETION_ROUNDS
# below is what actually stops a truncated reply from continuing
# forever.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
# Safety cap on how many times we'll ask the model to continue a reply
# that got cut off by the token limit above, so a pathological
# never-ending completion can't loop forever.
MAX_COMPLETION_ROUNDS = int(os.environ.get("LLM_MAX_COMPLETION_ROUNDS", "4"))
# Neither the sync (requests) nor the async (httpx) call to the LLM API
# had a timeout at all -- the streaming path was explicitly
# timeout=None. If the provider stalls mid-connection (dropped
# packets, an OpenRouter route hanging, etc.) the request just sits
# open forever: no exception, no response, nothing -- which is exactly
# what produces an endless "Thinking..." spinner on the frontend with
# zero output and no visible error. This caps how long we'll wait.
LLM_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "90"))

logger.info("Loading VoiceAssistant agent (STT=%s, LLM=%s)...", WHISPER_MODEL, LLM_MODEL)
assistant = VoiceAssistant(
    whisper_model=WHISPER_MODEL,
    llm_model=LLM_MODEL,
    tts_voice=TTS_VOICE,
    speak_replies=False,
)
text_chain = assistant.prompt | assistant.llm.model | StrOutputParser()


# With only ~10 CRM tools + a handful of web tools, we're well under this
# threshold today, so agent_node/stream_agent_turn bind ALL_TOOLS directly
# on every turn (simplest, most reliable path for a small tool set). Once
# the CRM tool list grows past this (e.g. more DocTypes, more integrations),
# retrieval kicks in automatically via tool_rag below -- no code change
# needed here, just add tools to CRM_Unified/tools.py.
TOOL_RAG_BYPASS_THRESHOLD = 100
TOOL_RAG_TOP_K = int(os.environ.get("TOOL_RAG_TOP_K", "3"))
TOOL_RAG_MIN_SCORE = float(os.environ.get("TOOL_RAG_MIN_SCORE", "0.25"))

tool_rag = None
if ALL_TOOLS:
    logger.info("Indexing %d CRM/web tool(s) for retrieval...", len(ALL_TOOLS))
    tool_rag = ToolRAG(ALL_TOOLS, top_k=TOOL_RAG_TOP_K, min_score=TOOL_RAG_MIN_SCORE)
tool_map = {tool.name: tool for tool in ALL_TOOLS}
logger.info("Loaded %d CRM/web tools.", len(ALL_TOOLS))

from fastapi import BackgroundTasks

# Python 3.11+ asyncio can garbage collect background tasks if no strong reference is kept.
# We store them here to prevent them from being killed mid-execution (e.g. while saving to SQLite).
_bg_tasks = set()
_checkpoint_conn = None  # global aiosqlite connection, used to force commits after saves

def safe_create_task(coro):
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at server startup and once at shutdown (FastAPI lifespan
    protocol). CRM tools are now loaded synchronously above at import
    time, so this only handles the Postgres audit-log schema."""

    # Creates sessions / audit_log / file_uploads in Postgres if they
    # don't exist yet (schema.sql is idempotent, so this is safe to run
    # on every startup, not just the first). Doesn't crash the server if
    # Postgres is misconfigured/unreachable -- it logs instead, so the
    # rest of the app still comes up; audit logging just won't work
    # until PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE are fixed in .env.
    try:
        apply_schema()
        logger.info("Postgres audit-log schema ready.")
    except Exception:
        logger.exception(
            "Could not apply Postgres schema -- check PGHOST/PGPORT/PGUSER/"
            "PGPASSWORD/PGDATABASE in .env. Audit logging will fail until this is fixed."
        )

    # Initialize AsyncSqliteSaver for persistent LangGraph memory
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    
    global agent_graph, _checkpoint_conn
    # isolation_level=None = autocommit mode so every write is immediately visible
    # to subsequent reads on the same connection (fixes WAL not-visible-to-self bug)
    _checkpoint_conn = await aiosqlite.connect("stream_history.sqlite", isolation_level=None)
    try:
        saver = AsyncSqliteSaver(_checkpoint_conn)
        await saver.setup()
        agent_graph = build_agent_graph(checkpointer=saver)
        logger.info("AsyncSqliteSaver persistent memory ready.")
        yield
    finally:
        await _checkpoint_conn.close()
        _checkpoint_conn = None


app = FastAPI(title="MagmaAssistance Backend", lifespan=lifespan)

# Allow CORS requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    # The browser API calls do not use cookies. Combining credentials with a
    # wildcard origin is needlessly fragile across browsers/proxies.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the WebRTC voice router
from Voice.voice_routes import voice_router
app.include_router(voice_router)


# ---------------------------------------------------------------------
# CRM dashboard
# ---------------------------------------------------------------------

_CLOSED_LEAD_STATUSES = {"lost", "junk", "unqualified", "converted"}
_CLOSED_TASK_STATUSES = {"done", "canceled", "cancelled"}
_TASK_PROGRESS = {"backlog": 0, "todo": 0, "in progress": 50, "done": 100, "canceled": 0, "cancelled": 0}


def _dashboard_records(doctype: str, fields: list, *, order_by: str = None, max_rows: int = 1000) -> list:
    """Read paginated CRM records for the dashboard without exposing the CRM API."""
    records = []
    for start in range(0, max_rows, 100):
        page = crm_client.get_list(doctype, fields=fields, order_by=order_by, limit=100, start=start) or []
        records.extend(page)
        if len(page) < 100:
            break
    return records


def _dashboard_datetime(value):
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _dashboard_time_ago(value) -> str:
    timestamp = _dashboard_datetime(value)
    if not timestamp:
        return "Recently"
    seconds = max(0, int((datetime.now(timezone.utc) - timestamp).total_seconds()))
    if seconds < 60:
        return "Just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 604800:
        return f"{seconds // 86400}d ago"
    return timestamp.strftime("%d %b")


def _dashboard_money(value) -> str:
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        amount = 0
    return f"₹{amount:,.0f}"


def _dashboard_number(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _dashboard_change(current: float, previous: float) -> tuple[str, str]:
    """Format a real period-over-period change for a dashboard stat card."""
    if previous == 0:
        if current == 0:
            return "0%", "up"
        return "+100%", "up"
    percent = round((current - previous) * 100 / previous)
    return f"{percent:+d}%", "up" if percent >= 0 else "down"


def _build_dashboard() -> dict:
    """Build a dashboard response exclusively from the connected CRM data."""
    leads = _dashboard_records(
        "CRM Lead",
        ["name", "lead_name", "first_name", "last_name", "organization", "lead_owner", "status", "email", "mobile_no", "phone", "website", "annual_revenue", "creation"],
        order_by="creation desc",
    )
    deals = _dashboard_records(
        "CRM Deal",
        ["name", "organization", "organization_name", "lead", "lead_name", "status", "deal_value", "expected_deal_value", "creation"],
        order_by="creation desc",
    )
    tasks = _dashboard_records(
        "CRM Task", ["name", "title", "status", "assigned_to", "creation", "modified"], order_by="modified desc",
    )
    notes = _dashboard_records(
        "FCRM Note", ["name", "title", "reference_doctype", "reference_docname", "owner", "creation"], order_by="creation desc",
    )
    calls = _dashboard_records(
        "CRM Call Log", ["name", "type", "status", "reference_doctype", "reference_docname", "owner", "creation"], order_by="creation desc",
    )
    users = _dashboard_records("User", ["name", "full_name", "email", "enabled"], max_rows=100)

    user_names = {
        user.get("name"): user.get("full_name") or user.get("email") or user.get("name")
        for user in users if user.get("name")
    }
    open_leads = [lead for lead in leads if str(lead.get("status") or "").casefold() not in _CLOSED_LEAD_STATUSES]
    active_deals = [deal for deal in deals if str(deal.get("status") or "").casefold() not in {"lost", "closed lost", "won", "closed won"}]
    active_tasks = [task for task in tasks if str(task.get("status") or "").casefold() not in _CLOSED_TASK_STATUSES]
    now = datetime.now(timezone.utc)
    current_cutoff = now.timestamp() - 30 * 86400
    previous_cutoff = now.timestamp() - 60 * 86400

    def in_period(record, start_timestamp, end_timestamp):
        created = _dashboard_datetime(record.get("creation"))
        if not created:
            return False
        timestamp = created.timestamp()
        return start_timestamp <= timestamp < end_timestamp

    current_open_leads = sum(in_period(lead, current_cutoff, now.timestamp()) for lead in open_leads)
    previous_open_leads = sum(in_period(lead, previous_cutoff, current_cutoff) for lead in open_leads)
    current_deals = sum(in_period(deal, current_cutoff, now.timestamp()) for deal in active_deals)
    previous_deals = sum(in_period(deal, previous_cutoff, current_cutoff) for deal in active_deals)
    current_tasks = sum(in_period(task, current_cutoff, now.timestamp()) for task in active_tasks)
    previous_tasks = sum(in_period(task, previous_cutoff, current_cutoff) for task in active_tasks)
    current_pipeline_value = sum(
        _dashboard_number(deal.get("deal_value") or deal.get("expected_deal_value"))
        for deal in active_deals if in_period(deal, current_cutoff, now.timestamp())
    )
    previous_pipeline_value = sum(
        _dashboard_number(deal.get("deal_value") or deal.get("expected_deal_value"))
        for deal in active_deals if in_period(deal, previous_cutoff, current_cutoff)
    )

    # Use the CRM's own creation timestamps and deal values, grouped by month.
    revenue_by_month = defaultdict(float)
    for deal in deals:
        created = _dashboard_datetime(deal.get("creation"))
        if not created:
            continue
        try:
            revenue_by_month[created.strftime("%Y-%m")] += _dashboard_number(deal.get("deal_value") or deal.get("expected_deal_value"))
        except (TypeError, ValueError):
            continue
    revenue_trend = [
        {"month": datetime.strptime(month, "%Y-%m").strftime("%b"), "value": round(value, 2)}
        for month, value in sorted(revenue_by_month.items())[-12:]
    ]

    stage_counts = Counter(str(deal.get("status") or "Unspecified") for deal in deals)
    pipeline_stages = [
        {"stage": stage, "count": count}
        for stage, count in sorted(stage_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    activities = []
    for note in notes:
        activities.append({
            "id": f"note:{note.get('name')}", "actor": user_names.get(note.get("owner"), note.get("owner") or "CRM user"),
            "action": "added a note to", "target": note.get("reference_docname") or note.get("title") or "a CRM record",
            "time": _dashboard_time_ago(note.get("creation")), "kind": "note", "_created": note.get("creation"),
        })
    for call in calls:
        activities.append({
            "id": f"call:{call.get('name')}", "actor": user_names.get(call.get("owner"), call.get("owner") or "CRM user"),
            "action": f"logged a {str(call.get('type') or 'call').casefold()} call for", "target": call.get("reference_docname") or "a CRM record",
            "time": _dashboard_time_ago(call.get("creation")), "kind": "call", "_created": call.get("creation"),
        })
    for task in tasks:
        activities.append({
            "id": f"task:{task.get('name')}", "actor": user_names.get(task.get("assigned_to"), task.get("assigned_to") or "Unassigned"),
            "action": f"has task {str(task.get('status') or 'Todo').casefold()}:", "target": task.get("title") or task.get("name"),
            "time": _dashboard_time_ago(task.get("modified") or task.get("creation")), "kind": "task", "_created": task.get("modified") or task.get("creation"),
        })
    activities.sort(key=lambda item: _dashboard_datetime(item.get("_created")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    activity_feed = [{key: value for key, value in item.items() if key != "_created"} for item in activities[:20]]

    lead_rows = []
    for lead in leads[:20]:
        name = lead.get("lead_name") or " ".join(filter(None, [lead.get("first_name"), lead.get("last_name")])) or lead.get("organization") or lead.get("name")
        evidence_fields = ("email", "mobile_no", "phone", "website", "organization", "annual_revenue")
        score = round(100 * sum(bool(lead.get(field)) for field in evidence_fields) / len(evidence_fields))
        lead_rows.append({
            "id": str(lead.get("name")), "name": name, "owner": user_names.get(lead.get("lead_owner"), lead.get("lead_owner") or "Unassigned"),
            "stage": lead.get("status") or "Unspecified", "value": _dashboard_money(lead.get("annual_revenue")), "score": score,
        })

    logged_user = crm_client.get_logged_user()
    if isinstance(logged_user, dict):
        logged_user = logged_user.get("user") or logged_user.get("message")
    agent_name = user_names.get(logged_user, logged_user or "CRM Agent")
    agent_user = next((user for user in users if user.get("name") == logged_user), None)
    my_tasks = [task for task in active_tasks if task.get("assigned_to") == logged_user]
    agent_tasks = [
        {"label": task.get("title") or task.get("name"), "progress": _TASK_PROGRESS.get(str(task.get("status") or "").casefold(), 0)}
        for task in my_tasks[:10]
    ]

    return {
        "stat_cards": [
            {"label": "Open Leads", "value": str(len(open_leads)), "delta": _dashboard_change(current_open_leads, previous_open_leads)[0], "trend": _dashboard_change(current_open_leads, previous_open_leads)[1], "accent": "signal"},
            {"label": "Active Deals", "value": str(len(active_deals)), "delta": _dashboard_change(current_deals, previous_deals)[0], "trend": _dashboard_change(current_deals, previous_deals)[1], "accent": "mint"},
            {"label": "Open Tasks", "value": str(len(active_tasks)), "delta": _dashboard_change(current_tasks, previous_tasks)[0], "trend": _dashboard_change(current_tasks, previous_tasks)[1], "accent": "amber"},
            {"label": "Pipeline Value", "value": _dashboard_money(sum(_dashboard_number(deal.get("deal_value") or deal.get("expected_deal_value")) for deal in active_deals)), "delta": _dashboard_change(current_pipeline_value, previous_pipeline_value)[0], "trend": _dashboard_change(current_pipeline_value, previous_pipeline_value)[1], "accent": "pulse"},
        ],
        "revenue_trend": revenue_trend,
        "pipeline_stages": pipeline_stages,
        "activity_feed": activity_feed,
        "leads": lead_rows,
        "agent": {"name": agent_name, "status": "Active" if not agent_user or agent_user.get("enabled") else "Inactive", "tasks": agent_tasks},
    }


@app.get("/api/dashboard")
async def dashboard():
    """Return live CRM dashboard data for the React frontend."""
    try:
        return await asyncio.to_thread(_build_dashboard)
    except (PermissionError, RuntimeError) as exc:
        logger.exception("Unable to build CRM dashboard")
        raise HTTPException(status_code=502, detail=f"CRM dashboard data is unavailable: {exc}") from exc



# ---------------------------------------------------------------------
# Memory: LangGraph state + checkpointer, keyed by session_id
# ---------------------------------------------------------------------
#
# Two layers, both persisted per session_id via a MemorySaver checkpointer
# (swap for SqliteSaver/Postgres if this needs to survive a restart):
#
# 1. SHORT-TERM MEMORY -- state["messages"], accumulated automatically by
#    the `add_messages` reducer. Previously each /api/chat call only sent
#    [SystemMessage, HumanMessage(text)] with zero memory of earlier turns
#    in the session; now the full conversation is kept in the checkpointer
#    and a trimmed window (MAX_HISTORY_MESSAGES) is sent to the LLM each
#    turn, so latency/cost stay flat as a session grows.
#
# 2. TASK-CONTEXT MEMORY -- state["current_task"], state["task_slots"],
#    state["pending_tool"], state["pending_missing"]. This replaces the old
#    global `_pending_actions` dict one-for-one for slot-filling (a
#    create/update tool call missing a required field opens a flow that
#    asks for each missing field across turns, exactly as before) and adds
#    on top of it: `current_task` is a short label kept alive across turns
#    of the SAME task and cleared only when a lightweight classifier
#    decides the user has switched topics. tool retrieval retrieval is queried
#    against "<task label>. <new message>" instead of the raw message
#    alone, which keeps retrieval accurate on short follow-ups ("what's
#    her phone number?") that wouldn't embed well on their own.
MAX_HISTORY_TOKENS = 60000  # Approximated: 1 token ~= 4 chars

def _approx_tokens(messages: list) -> int:
    return sum(len(str(m.content)) // 4 for m in messages)


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str           # Rolling memory summary
    current_task: Optional[str]
    task_slots: dict
    pending_tool: Optional[str]
    pending_missing: list  # ordered [(field, question), ...] still to collect
    session_id: str        # thread id, passed through so nodes can attribute audit_log entries
    user_id: Optional[str] # who prompted this turn, passed through for the same reason
    
    # Multi-Agent Workflow State
    intent_category: Optional[str]
    extracted_entities: dict
    research_context: Optional[str]
    crm_context: Optional[str]
    proposal: Optional[str]


# Values a model sometimes invents in place of a real answer when it
# doesn't actually know one, instead of leaving the field blank so
# slot-filling can ask. Treated as "still missing" so a required field
# (e.g. company, warehouse) can't be silently satisfied by a guess.
_PLACEHOLDER_VALUES = {
    "default", "n/a", "na", "none", "null", "unknown",
    "not specified", "not sure", "unspecified", "todo", "tbd", "-",
}


def _missing_fields(tool_name: str, args: dict) -> list:
    """Ordered (field, question) pairs required for `tool_name` that are
    absent, empty, or filled with a placeholder-like guess in `args`."""
    required = ALL_REQUIRED_FIELDS.get(tool_name, [])
    args = args or {}
    missing = []
    for field, question in required:
        value = args.get(field)
        if not value:
            missing.append((field, question))
        elif isinstance(value, str) and value.strip().lower() in _PLACEHOLDER_VALUES:
            missing.append((field, question))
    return missing


def _last_human_message(messages) -> Optional[str]:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return None


def _parse_json_loose(text: str) -> dict:
    """Classifier replies sometimes get ```json-fenced despite instructions
    not to -- strip that before parsing instead of failing outright."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def _flatten_scalar(value):
    """Some local models (llama3.2 and similar) occasionally wrap a plain
    scalar argument in a dict instead of passing it directly — e.g.
    {'name': 'Negotiation'} instead of just 'Negotiation' for a `stage:
    str` field. Pydantic then rejects the call outright with a
    string_type/int_type error and the whole tool call is lost. This
    unwraps that: single-key dicts use their one value, dicts with a
    recognizable wrapper key (value/name/text/input) use that key, and
    anything else falls back to a string representation rather than
    failing. Recurses in case of double-wrapping."""
    if isinstance(value, dict):
        if len(value) == 1:
            return _flatten_scalar(next(iter(value.values())))
        for key in ("value", "name", "text", "input"):
            if key in value:
                return _flatten_scalar(value[key])
        return str(value)
    return value


def _sanitize_tool_args(tool_name: str, args: dict) -> dict:
    if not args:
        return args

    tool = tool_map.get(tool_name)
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None) if schema else None
    if not fields:
        return args

    cleaned = dict(args)

    for field_name, field_info in fields.items():
        if field_name not in cleaned:
            continue

        value = cleaned[field_name]

        annotation = field_info.annotation
        inner_types = [
            t for t in getattr(annotation, "__args__", [annotation])
            if t is not type(None)
        ]

        expects_scalar = any(
            t in (str, int, float, bool)
            for t in inner_types
        )

        # Existing dict unwrapping
        if isinstance(value, dict) and expects_scalar:
            value = _flatten_scalar(value)
            cleaned[field_name] = value

        # NEW: remove empty strings for numeric fields
        if value == "":
            if int in inner_types or float in inner_types:
                cleaned.pop(field_name, None)

    return cleaned

# =====================================================================
# GENERAL-PURPOSE DOCUMENT UPLOAD (ANY PDF/IMAGE, NOT JUST POs)
# =====================================================================
# Separate from /api/upload-document above -- that endpoint's strict Purchase
# Order JSON extraction / auto-create-in-Frappe CRM flow is untouched.
# This lets a user upload any PDF/image and then just ask questions
# about it in normal chat (/api/chat, /query); the extracted text is
# stashed here per session_id and injected into the conversation once
# by generate_reply() below.
document_store: Dict[str, Dict[str, Any]] = {}  # session_id -> {filename, text, injected}

# session_id -> CRMIdentity: the real Frappe CRM user each chat session is
# acting as, resolved via /api/session/identify. generate_reply() binds
# this around the agent turn so every crm_search call for that turn
# runs with THAT PERSON'S OWN Frappe CRM credentials -- and is therefore
# gated by Frappe's own, built-in role/permission engine -- instead of
# the shared service account. A session with nothing here just keeps
# using the shared service account, same as before this was wired up.
session_identities: Dict[str, CRMIdentity] = {}

class SessionIdentifyRequest(BaseModel):
    session_id: str
    crm_api_key: str
    crm_api_secret: str


@app.post("/api/session/identify")
async def identify_session(req: SessionIdentifyRequest):
    """Bind a real Frappe CRM user to a chat session, via their own personal
    API key/secret (Frappe CRM: User menu -> My Settings -> API Access ->
    Generate Keys). Call this once when a person starts or resumes a
    session, before /api/chat. From then on, every CRM tool call in that
    session is made with their own credentials and enforced by Frappe's
    own permission checks -- no custom Frappe CRM app required."""
    try:
        identity = crm_client.resolve_identity(req.crm_api_key, req.crm_api_secret)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    session_identities[req.session_id] = identity
    return {"authenticated": True, "user": identity.user, "roles": identity.roles}


@app.post("/api/session/logout")
async def logout_session(session_id: str = Form(...)):
    """Unbind whatever identity was set for this session -- subsequent
    turns fall back to the shared service account until re-identified."""
    session_identities.pop(session_id, None)
    return {"success": True}


class GeneralDocumentUploadResponse(BaseModel):
    status: str
    filename: str
    message: str
    page_count: int
    extraction_method: str


@app.post("/api/upload-document", response_model=GeneralDocumentUploadResponse)
async def upload_general_document(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    user_id: str = Form("anonymous"),
):
    """Reads ANY PDF or image (for PDF/images), extracts its
    full text, and stores it against session_id so the user can ask
    follow-up questions about it in normal chat."""
    allowed_types = ["image/jpeg", "image/png", "application/pdf", "image/jpg"]

    if file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file.content_type}'. Please upload JPEG, PNG, or PDF."
        )

    try:
        file_bytes = await file.read()
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds the 10MB limit.")

        logger.info(f"File '{file.filename}' uploaded for session '{session_id}'. Extracting document text...")

        upload_meta = s3_storage.upload_file(
            file_bytes=file_bytes,
            original_filename=file.filename,
            content_type=file.content_type,
            upload_kind="general_document",
            session_id=session_id,
            user_id=user_id,
        )

        extraction = llm_ocr_engine.extract_document_text(file_bytes=file_bytes, mime_type=file.content_type)

        document_store[session_id] = {
            "filename": file.filename,
            "text": extraction["text"],
            "injected": False,
        }

        audit_log.record_file_upload(
            **upload_meta,
            extracted_metadata={
                "page_count": extraction["page_count"],
                "pages_read": extraction["pages_read"],
                "method": extraction["method"],
            },
            status="processed",
        )

        return GeneralDocumentUploadResponse(
            status="success",
            filename=file.filename,
            message=(
                f"Document read successfully ({extraction['pages_read']}/{extraction['page_count']} "
                f"page(s), method={extraction['method']}). You can now ask questions about it in chat."
            ),
            page_count=extraction["page_count"],
            extraction_method=extraction["method"],
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("Error handling document upload in /api/upload-document")
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(e)}")


async def _execute_tool(
    tool_name: str,
    args: dict,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    prompt_text: Optional[str] = None,
):
    """Async because MCP-sourced tools (CRM_Unified/mcp_server.py, loaded via
    CRM tools/mcp_tools.py) only implement `.ainvoke()`, not the sync
    `.invoke()`. This works transparently for the existing local
    CRM tools/*.py tools too — LangChain's BaseTool.ainvoke() runs a sync
    tool's normal invoke() under the hood when no native async
    implementation exists, so no other tool code needed to change.

    Every call is written to the Postgres audit log (session_id, tool,
    args, result, how long it took, and who prompted it) regardless of
    success — this is the one choke point all tool execution passes
    through, so it's the cheapest place to record "what actions did the
    agent actually take" for later review."""
    tool = tool_map.get(tool_name)
    if tool is None:
        result = f"Tool '{tool_name}' is not available."
        if session_id:
            audit_log.log_turn(
                session_id, "tool", result, tool_name=tool_name, tool_args=args,
                user_id=user_id, prompt_text=prompt_text, tool_status="not_found",
            )
        return result
    try:
        effective_args = _sanitize_tool_args(tool_name, args) or {}
        with audit_log.time_tool_call() as elapsed:
            result = await tool.ainvoke(effective_args)
        if session_id:
            audit_log.log_turn(
                session_id, "tool", str(result), tool_name=tool_name, tool_args=effective_args,
                user_id=user_id, prompt_text=prompt_text, tool_status="success",
                duration_ms=elapsed(),
            )
        return result
    except PermissionError as e:
        # Raised by crm_client when a bound per-user CRMIdentity (see
        # /api/session/identify) is denied by Frappe's own permission
        # engine -- surfaced as-is so the agent can tell the person why,
        # instead of a generic failure message.
        logger.warning("Tool '%s' denied by Frappe CRM permission check: %s", tool_name, e)
        failure = str(e)
        if session_id:
            audit_log.log_turn(
                session_id, "tool", failure, tool_name=tool_name, tool_args=args,
                user_id=user_id, prompt_text=prompt_text, tool_status="permission_denied",
                error_message=str(e),
            )
        return failure
    except Exception as e:
        logger.exception("Tool '%s' failed", tool_name)
        detail = str(e).strip()
        # RuntimeError from crm_client._request() already includes Frappe's
        # own message (e.g. "CRM request failed (417): Could not find
        # Status: Open") -- surface that instead of a generic string so the
        # agent (and the person) can actually see what went wrong and fix
        # the input, rather than just knowing *that* something failed.
        failure = f"'{tool_name}' failed: {detail}" if detail else f"'{tool_name}' failed to fetch CRM data right now."
        if session_id:
            audit_log.log_turn(
                session_id, "tool", failure, tool_name=tool_name, tool_args=args,
                user_id=user_id, prompt_text=prompt_text, tool_status="error",
                error_message=detail,
            )
        return failure


async def _stream_chat_completion(messages, tools=None):
    api_messages = [convert_message_to_dict(m) for m in messages]
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    env_openai_key = os.environ.get("OPENAI_API_KEY")
    key = openrouter_key or env_openai_key
    is_openrouter = bool(openrouter_key)
    if is_openrouter:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "http://localhost:8050", "X-Title": "MagmaAssistance"}
        url = "https://openrouter.ai/api/v1/chat/completions"
        model_name = LLM_MODEL if "/" in LLM_MODEL else f"openai/{LLM_MODEL}"
    else:
        headers = {"Authorization": f"Bearer {env_openai_key}", "Content-Type": "application/json"}
        url = "https://api.openai.com/v1/chat/completions"
        model_name = LLM_MODEL
    data = {"model": model_name, "messages": api_messages, "temperature": assistant.llm.temperature, "stream": True, "max_tokens": LLM_MAX_TOKENS}
    if tools:
        data["tools"] = tools
    tool_acc = {}
    content = ""
    finish_reason = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=60.0)) as client:
        async with client.stream("POST", url, json=data, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                finish_reason = choice.get("finish_reason") or finish_reason
                if delta.get("content"):
                    content += delta["content"]
                    yield {"type": "token", "text": delta["content"]}
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    entry = tool_acc.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        entry["name"] = fn["name"]
                    if fn.get("arguments"):
                        entry["arguments"] += fn["arguments"]
    tool_calls = []
    for idx in sorted(tool_acc):
        entry = tool_acc[idx]
        try:
            args = json.loads(entry["arguments"]) if entry["arguments"] else {}
        except Exception:
            args = {}
        tool_calls.append({"name": entry["name"], "args": args, "id": entry["id"] or f"call_{idx}"})
    yield {"type": "done", "content": content, "tool_calls": tool_calls, "finish_reason": finish_reason}


async def _stream_full_reply(call_messages, tools=None):
    """Wraps _stream_chat_completion with the same finish_reason=="length"
    continuation handling as OpenAIChatModel._generate, so streamed
    replies don't silently cut off mid-sentence either. Yields token
    events as they arrive, then a final {"type": "done", "content":
    ..., "tool_calls": ...} once the reply is actually complete (or
    MAX_COMPLETION_ROUNDS is hit). Tool-call replies are never
    continued -- same reasoning as the non-streaming path."""
    content = ""
    tool_calls = []
    messages = call_messages
    for round_number in range(MAX_COMPLETION_ROUNDS + 1):
        piece = ""
        async for event in _stream_chat_completion(messages, tools=tools):
            if event["type"] == "token":
                yield {"type": "token", "text": event["text"]}
            else:
                piece = event["content"]
                tool_calls = event["tool_calls"]
                finish_reason = event["finish_reason"]

        content += piece

        if finish_reason != "length" or tool_calls or round_number == MAX_COMPLETION_ROUNDS:
            if finish_reason == "length":
                logger.warning(
                    "Streamed LLM reply still truncated after %d continuation "
                    "round(s); returning what we have.", round_number
                )
            break

        messages = [*call_messages, AIMessage(content=piece), HumanMessage(
            content="Continue exactly where you left off. Do not repeat any text or restart the answer."
        )]

    yield {"type": "done", "content": content, "tool_calls": tool_calls}


async def stream_agent_turn(text, session_id=None, user_id=None, history=None, task_context=None):
    """Streaming twin of agent_node/generate_reply for the realtime voice
    WS path. Keeps its own `history` list (caller-owned, per-connection)
    instead of the LangGraph checkpointer, so the existing text-chat graph
    and its slot-filling flow are untouched. Yields token/tool_call/
    tool_result/done event dicts in call order.
    
    IMPORTANT: We track start_len so callers can extract only the NEW messages
    added this turn (the delta) for saving. The checkpointer uses add_messages
    which appends — so saving the full list would duplicate every message.
    """
    history = history if history is not None else []
    start_len = len(history)  # snapshot before we mutate
    history.append(HumanMessage(content=text))
    # Trim for LLM context window — this is a separate list, does NOT affect history
    trimmed = trim_messages(history, max_tokens=MAX_HISTORY_TOKENS, token_counter=_approx_tokens, strategy="last", include_system=False)

    candidate_tools = []
    if ALL_TOOLS:
        candidate_tools = list(ALL_TOOLS) if len(ALL_TOOLS) <= TOOL_RAG_BYPASS_THRESHOLD else (tool_rag.retrieve(text) if tool_rag else [])

    openai_tools = None
    if candidate_tools:
        from langchain_core.utils.function_calling import convert_to_openai_tool
        openai_tools = []
        for t in candidate_tools:
            formatted = convert_to_openai_tool(t)
            if "function" in formatted and "parameters" in formatted["function"]:
                formatted["function"]["parameters"] = _clean_schema_for_openai(formatted["function"]["parameters"])
            openai_tools.append(formatted)

    system_parts = [assistant.llm.system_prompt]
    if task_context:
        system_parts.append(f"\nCurrent task in progress: {task_context}.")
    call_messages = [SystemMessage(content="\n".join(system_parts)), *trimmed]

    max_rounds = 4
    try:
        for round_number in range(max_rounds + 1):
            content = ""
            tool_calls = []
            async for event in _stream_full_reply(call_messages, tools=openai_tools):
                if event["type"] == "token":
                    yield {"type": "token", "text": event["text"]}
                else:
                    content = event["content"]
                    tool_calls = event["tool_calls"]

            if not tool_calls:
                ai_msg = AIMessage(content=content)
                history.append(ai_msg)
                # Yield the delta (only newly added messages) so callers can save it
                yield {"type": "done", "text": content, "_delta": history[start_len:]}
                return

            if round_number == max_rounds:
                break

            ai_msg = AIMessage(content=content, tool_calls=tool_calls)
            call_messages.append(ai_msg)
            history.append(ai_msg)
            for tc in tool_calls:
                yield {"type": "tool_call", "name": tc["name"], "args": tc.get("args") or {}}
                result = await _execute_tool(tc["name"], tc.get("args") or {}, session_id=session_id, user_id=user_id, prompt_text=text)
                yield {"type": "tool_result", "name": tc["name"], "result": result}
                t_msg = ToolMessage(content=str(result), tool_call_id=tc["id"])
                call_messages.append(t_msg)
                history.append(t_msg)

        content = ""
        async for event in _stream_full_reply(call_messages, tools=None):
            if event["type"] == "token":
                yield {"type": "token", "text": event["text"]}
            else:
                content = event["content"]
        ai_msg = AIMessage(content=content)
        history.append(ai_msg)
        yield {"type": "done", "text": content, "_delta": history[start_len:]}
    except asyncio.CancelledError:
        # Clean up ONLY if we have a dangling AIMessage with tool_calls that lacks matching ToolMessages
        if history:
            last_msg = history[-1]
            if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                history.pop()
            elif isinstance(last_msg, ToolMessage):
                # Count how many ToolMessages we have at the end
                tool_msg_count = 0
                for msg in reversed(history):
                    if isinstance(msg, ToolMessage):
                        tool_msg_count += 1
                    else:
                        break
                
                # Check the AIMessage that preceded these ToolMessages
                if len(history) > tool_msg_count:
                    ai_msg = history[-(tool_msg_count + 1)]
                    if isinstance(ai_msg, AIMessage) and getattr(ai_msg, "tool_calls", None):
                        if len(ai_msg.tool_calls) != tool_msg_count:
                            # Incomplete tool execution sequence! Pop them all to prevent OpenAI 400 errors.
                            for _ in range(tool_msg_count + 1):
                                history.pop()
        raise


_FAKE_NAME_RE = re.compile(r'"name"\s*:\s*"(?P<name>[a-zA-Z_][\w\-.]*)"')
_FAKE_KV_STR_RE = re.compile(r'"(?P<key>[a-zA-Z_]\w*)"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"')
_FAKE_KV_NUM_RE = re.compile(r'"(?P<key>[a-zA-Z_]\w*)"\s*:\s*(?P<value>-?\d+(?:\.\d+)?)\b')


def _extract_fake_tool_call(content: str):
    """Some local models via Ollama (llama3.2 and similar) occasionally
    reply with a plain-text approximation of a tool call instead of using
    the real function-calling protocol, e.g.:
        {"name":"create_customer","parameters={"lead_id":"","customer_name":"Sujay"}}
    This is often broken well beyond a single typo — note the missing
    colon/closing quote around "parameters" above, which means the
    "parameters" value isn't even a validly nested object, so a strict
    brace-matching parse won't survive it either. Rather than trying to
    fully parse the structure, this pulls out the tool name, then scrapes
    any "key":"value" or "key":123 pairs found anywhere after it — good
    enough to recover the user's actual intent from what is essentially
    a hallucinated shape. Returns None if `content` doesn't look like an
    attempted tool call, or names a tool that doesn't exist.
    """
    text = (content or "").strip()
    if not text.startswith("{") or '"name"' not in text or "parameters" not in text.lower():
        return None

    name_match = _FAKE_NAME_RE.search(text)
    if not name_match:
        return None

    name = name_match.group("name")
    if name not in tool_map:
        return None

    # Only look at text after "parameters" so we don't re-capture "name"
    # itself as if it were an argument.
    params_idx = text.lower().find("parameters")
    tail = text[params_idx:] if params_idx != -1 else text

    args = {}
    for m in _FAKE_KV_STR_RE.finditer(tail):
        args.setdefault(m.group("key"), m.group("value"))
    for m in _FAKE_KV_NUM_RE.finditer(tail):
        key = m.group("key")
        if key in args:
            continue
        value = m.group("value")
        args[key] = float(value) if "." in value else int(value)

    if not args:
        return None

    return {"name": name, "args": args, "id": "fake-tool-call-0"}


_CLASSIFY_SYSTEM = (
    "You track whether a new user message continues the SAME task as before, "
    "or starts a NEW, unrelated task, for an CRM assistant.\n"
    "Reply with ONLY a compact JSON object, nothing else: "
    '{"same_task": true|false, "task_label": "<3-6 word label for the CURRENT task after this message>"}\n'
    "Rules:\n"
    "- A follow-up about the same record/order/lead/customer, or the user "
    "supplying info the assistant just asked for, is the SAME task.\n"
    "- A greeting, thanks, or closing remark right after a task is finished "
    "does NOT start a new task; keep same_task=true with the same label.\n"
    "- A request about a different customer/record/action/topic is a NEW task."
)


def intake_node(state: ChatState) -> dict:
    """If a slot-filling flow is open (pending_tool set), this turn's
    message is the answer to the next missing field — merge it in and
    either ask the next question or hand off to execute_pending once the
    record is complete. Otherwise, no-op and fall through to classify_task."""
    pending_tool = state.get("pending_tool")
    pending_missing = state.get("pending_missing") or []
    if not pending_tool or not pending_missing:
        return {}

    last_user_msg = _last_human_message(state["messages"]) or ""
    field, _question = pending_missing[0]
    parser = ALL_FIELD_PARSERS.get((pending_tool, field))
    value = parser(last_user_msg) if parser else last_user_msg.strip()

    slots = {**(state.get("task_slots") or {}), field: value}
    remaining = pending_missing[1:]

    if remaining:
        return {
            "task_slots": slots,
            "pending_missing": remaining,
            "messages": [AIMessage(content=remaining[0][1])],
        }

    # All required fields collected -- clear the queue so the router sends
    # this to execute_pending. pending_tool/task_slots stay set until the
    # tool call actually runs (execute_pending clears them after).
    return {"task_slots": slots, "pending_missing": []}


def _route_after_intake(state: ChatState) -> str:
    if state.get("pending_tool"):
        return END if state.get("pending_missing") else "execute_pending"
    return "classify_task"


def classify_task_node(state: ChatState) -> dict:
    """Task-context memory: keeps `current_task` alive across turns of the
    same task, and only resets it when the topic genuinely changes."""
    last_user_msg = _last_human_message(state["messages"])
    if last_user_msg is None:
        return {}

    current_task = state.get("current_task")
    if current_task is None:
        return {"current_task": last_user_msg[:60]}

    prompt = f"Active task: {current_task}\nNew user message: {last_user_msg}"
    try:
        resp = assistant.llm.model.invoke(
            [SystemMessage(content=_CLASSIFY_SYSTEM), HumanMessage(content=prompt)]
        )
        parsed = _parse_json_loose(resp.content)
        same_task = bool(parsed.get("same_task"))
        task_label = parsed.get("task_label") or current_task
    except Exception as exc:  # noqa: BLE001
        # Fail safe toward continuity rather than losing progress over a
        # transient classification error.
        logger.warning("Task classification failed (%s); assuming same task.", exc)
        same_task, task_label = True, current_task

    return {"current_task": task_label}


def _plain_reply(history, task_context: Optional[str] = None) -> str:
    """Used when no CRM tool is needed for this turn (plain chat, or
    Q&A about an uploaded document). Unlike text_chain.invoke(), which
    only ever sees the single latest message, this sends the full
    trimmed history -- including any injected '[System note: the user
    uploaded a document...]' context from generate_reply() -- so a
    follow-up like 'solve problem 3 from that PDF' actually has the
    document content available instead of being answered blind."""
    system_parts = [assistant.llm.system_prompt, f"Current date: {datetime.now().astimezone():%Y-%m-%d}."]
    if task_context:
        system_parts.append(f"\nCurrent task in progress: {task_context}.")
    call_messages = [SystemMessage(content="\n".join(system_parts)), *history]
    response = assistant.llm.model.invoke(call_messages)
    return response.content


def _is_unqualified_approval(message: str) -> bool:
    """Recognise an approval without treating a correction as consent."""
    text = (message or "").strip().lower()
    if not text:
        return False
    if re.search(r"\b(no|wrong|incorrect|change|correct|instead|but|except|update|modify|add)\b", text):
        return False
    return bool(re.search(r"\b(yes|approve|approved|proceed|continue|go ahead|create)\b", text))

def _build_fallback_chart(results: list, query_lower: str) -> Optional[str]:
    """Generic, zero-hardcode fallback chart generator: parses ANY list of records
    returned by crm_search, detects category/label keys and numeric/status keys,
    and dynamically builds bar, line, or pie chart specs."""
    if not results:
        return None

    IGNORED_KEYS = {"name", "docstatus", "idx", "owner", "creation", "modified"}

    for _tc, raw_result in reversed(results):
        raw_text = str(raw_result)
        try:
            cleaned = raw_text
            if cleaned.startswith("[{'type': 'text', 'text':"):
                parsed_wrapper = ast.literal_eval(cleaned)
                if isinstance(parsed_wrapper, list) and len(parsed_wrapper) > 0 and 'text' in parsed_wrapper[0]:
                    cleaned = parsed_wrapper[0]['text']

            data = ast.literal_eval(cleaned)

            if not (isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict)):
                continue

            first_row = data[0]
            avail_keys = [k for k in first_row.keys() if k.lower() not in IGNORED_KEYS]

            if not avail_keys:
                continue

            cat_keys = []
            num_keys = []
            for k in avail_keys:
                val = first_row.get(k)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    num_keys.append(k)
                elif isinstance(val, str) and not val.replace(".", "", 1).isdigit():
                    cat_keys.append(k)
                elif val is not None:
                    try:
                        float(str(val))
                        num_keys.append(k)
                    except ValueError:
                        cat_keys.append(k)

            # Prioritize standard CRM label fields over generic text
            priority_cat = [
                "customer", "customer_name", "supplier", "supplier_name",
                "item_code", "item_name", "production_item", "warehouse",
                "territory", "item_group", "posting_date", "transaction_date"
            ]
            best_cat = next((k for p in priority_cat for k in cat_keys if k.lower() == p or p in k.lower()), cat_keys[0] if cat_keys else None)

            # Prioritize standard CRM numeric fields over generic numbers
            priority_num = [
                "grand_total", "total", "net_total", "amount",
                "qty", "produced_qty", "stock_qty", "rate", "valuation_rate"
            ]
            best_num = next((k for p in priority_num for k in num_keys if k.lower() == p or p in k.lower()), num_keys[0] if num_keys else None)

            # If all rows belong to a single category (e.g. single customer query like "West View Software Ltd."),
            # or if date fields are present, switch category key to date for a Timeline Line Chart!
            date_key = next((k for k in avail_keys if any(dw in k.lower() for dw in ["date", "posting", "transaction"])), None)
            if best_cat and date_key:
                unique_cats = {str(r.get(best_cat) or "").strip() for r in data}
                if len(unique_cats) <= 1:
                    best_cat = date_key

            # Strategy 1: Numerical aggregation (Category key + Numeric key) -> Bar or Line Chart
            if best_cat and best_num:
                label_key = best_cat
                value_key = best_num

                totals: dict[str, float] = {}
                for row in data:
                    label = str(row.get(label_key) or "Unknown").strip()
                    try:
                        num_val = float(row.get(value_key) or 0)
                    except (ValueError, TypeError):
                        num_val = 0.0
                    totals[label] = totals.get(label, 0.0) + num_val

                if totals:
                    is_timeline = any(w in label_key.lower() for w in ["date", "month", "year", "time", "day"])
                    if is_timeline:
                        sorted_items = sorted(totals.items(), key=lambda x: x[0])
                        x_axis = [item[0] for item in sorted_items]
                        series_data = [round(item[1], 2) for item in sorted_items]
                    else:
                        x_axis = list(totals.keys())
                        series_data = [round(v, 2) for v in totals.values()]

                    label_title = label_key.replace("_", " ").title()
                    value_title = value_key.replace("_", " ").title()
                    chart_type = "line" if is_timeline else "bar"

                    chart_spec = {
                        "type": chart_type,
                        "title": f"{value_title} Over Time" if is_timeline else f"{value_title} by {label_title}",
                        "xAxis": x_axis,
                        "series": [{"name": value_title, "data": series_data}]
                    }
                    return f"\n\n```chart\n{json.dumps(chart_spec, indent=2)}\n```"

            # Strategy 2: Categorical Frequency Breakdown (Status/Type key) -> Pie Chart
            if cat_keys:
                status_key = next((k for k in cat_keys if "status" in k.lower() or "group" in k.lower() or "type" in k.lower()), cat_keys[0])
                counts: dict[str, int] = {}
                for row in data:
                    cat_val = str(row.get(status_key) or "Unknown").strip()
                    counts[cat_val] = counts.get(cat_val, 0) + 1

                if counts:
                    title_name = status_key.replace("_", " ").title()
                    chart_spec = {
                        "type": "pie",
                        "title": f"{title_name} Distribution",
                        "labels": list(counts.keys()),
                        "values": list(counts.values())
                    }
                    return f"\n\n```chart\n{json.dumps(chart_spec, indent=2)}\n```"

        except Exception as e:
            logger.debug("Generic fallback chart parsing skipped: %s", e)
            continue

    return None



# --- 1. Summarize Node (Rolling Memory) ---
def summarize_node(state: ChatState) -> dict:
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    
    # If messages exceed threshold
    if len(messages) > 10:
        # Keep last 4 messages, summarize the rest
        to_summarize = messages[:-4]
        
        prompt = (
            f"Summarize the following conversation history. "
            f"Include any important entities, facts, or context.\n"
            f"Previous Summary: {summary}\n"
            f"New Messages: {to_summarize}"
        )
        new_summary_msg = assistant.llm.model.invoke(prompt)
        new_summary = new_summary_msg.content
        
        # Remove old messages from state
        delete_messages = [RemoveMessage(id=m.id) for m in to_summarize if getattr(m, 'id', None)]
        return {"summary": new_summary, "messages": delete_messages}
    return {}

# --- 2. Supervisor Node ---
from pydantic import BaseModel
class IntentOutput(BaseModel):
    category: Literal["chitchat", "crm_query", "crm_write", "web_search"]
    record_type: Optional[str]
    entities: Dict[str, str]

def supervisor_node(state: ChatState) -> dict:
    last_user_msg = _last_human_message(state["messages"]) or ""
    summary = state.get("summary", "")
    
    context_msg = f"Context: {summary}\n" if summary else ""
    
    from LLM.LLM import INTENT_SYSTEM_PROMPT
    llm_intent = assistant.llm.model.with_structured_output(IntentOutput)
    
    intent = llm_intent.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=context_msg + last_user_msg)
    ])
    
    logger.info("Intent classified: category=%s, entities=%s", intent.category, intent.entities)
    
    return {
        "intent_category": intent.category,
        "extracted_entities": intent.entities
    }

def route_from_supervisor(state: ChatState) -> str:
    cat = state.get("intent_category")
    if cat in ("crm_write", "web_search"):
        return "web_research_node"
    else:
        return "general_node"

# --- 3. Web Research Node ---
async def web_research_node(state: ChatState) -> dict:
    from LLM.LLM import RESEARCH_SYSTEM_PROMPT
    entities = state.get("extracted_entities", {})
    last_user_msg = _last_human_message(state["messages"]) or ""
    
    web_tools = [t for t in ALL_TOOLS if t.name in ("web_search", "web_company_search", "web_fetch_page")]
    search_llm = assistant.llm.model.bind_tools(web_tools)
    
    call_messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT), 
        HumanMessage(content=f"Research these entities: {entities}. Original request: {last_user_msg}")
    ]
    
    for _ in range(2):
        response = search_llm.invoke(call_messages)
        if not response.tool_calls:
            break
            
        call_messages.append(response)
        for tc in response.tool_calls:
            result = await _execute_tool(
                tc["name"], tc.get("args") or {},
                session_id=state.get("session_id"), user_id=state.get("user_id"),
                prompt_text=last_user_msg
            )
            call_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            
    final_response = search_llm.invoke(call_messages)
    
    if state.get("intent_category") == "web_search":
        return {"messages": [AIMessage(content=final_response.content)], "research_context": final_response.content}
    return {"research_context": final_response.content}

def route_after_research(state: ChatState) -> str:
    if state.get("intent_category") == "web_search":
        return END
    return "crm_context_node"

# --- 4. CRM Context Node ---
async def crm_context_node(state: ChatState) -> dict:
    entities = state.get("extracted_entities", {})
    if not entities:
        return {"crm_context": "No entities to lookup"}
    crm_tools = [t for t in ALL_TOOLS if t.name.startswith("crm_")]
    context_llm = assistant.llm.model.bind_tools(crm_tools)
    prompt = (
        "You are an internal Frappe CRM researcher. Query the CRM to gather relevant context "
        "for the user's request. Use metadata when field names are uncertain. "
        f"Entities: {entities}\nReturn a concise summary of what you found."
    )
    call_messages = [SystemMessage(content=prompt), HumanMessage(content="Gather internal CRM context.")]
    response = context_llm.invoke(call_messages)
    if response.tool_calls:
        call_messages.append(response)
        for tc in response.tool_calls:
            result = await _execute_tool(tc["name"], tc.get("args") or {}, session_id=state.get("session_id"), user_id=state.get("user_id"))
            call_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        final = context_llm.invoke(call_messages)
        return {"crm_context": final.content}
    return {"crm_context": response.content}

# --- 5. CRM Proposal Node ---
def proposal_node(state: ChatState) -> dict:
    from LLM.LLM import PROPOSAL_SYSTEM_PROMPT
    entities = state.get("extracted_entities", {})
    research = state.get("research_context", "")
    crm_ctx = state.get("crm_context", "")
    prop_response = assistant.llm.model.invoke([
        SystemMessage(content=PROPOSAL_SYSTEM_PROMPT),
        HumanMessage(content=f"Entities: {entities}\nWeb Research: {research}\nCRM Context: {crm_ctx}")
    ])
    return {"messages": [AIMessage(content=prop_response.content)], "proposal": prop_response.content}

# --- 6. General Node (Fallback / Chitchat / CRM Query) ---
from langchain_core.messages import trim_messages
async def general_node(state: ChatState) -> dict:
    last_user_msg = _last_human_message(state["messages"]) or ""
    task_context = state.get("current_task")
    
    history = trim_messages(
        state["messages"],
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter=_approx_tokens,
        strategy="last",
        include_system=False,
        start_on="human"
    )
    
    candidate_tools = list(ALL_TOOLS)
    
    intent_cat = state.get("intent_category")
    if intent_cat in ("chitchat", "crm_query"):
        _WEB = {"web_search", "web_fetch_page", "web_crawl", "web_company_search"}
        candidate_tools = [t for t in candidate_tools if t.name not in _WEB]
        
    llm_with_tools = assistant.llm.model.bind_tools(candidate_tools)
    import datetime
    current_date_str = datetime.date.today().strftime("%Y-%m-%d")
    system_parts = [
        assistant.llm.system_prompt,
        f"\nCURRENT SYSTEM DATE: {current_date_str}",
        "\nSENIOR CRM & BUSINESS INTELLIGENCE DIRECTIVE:",
        "- You are an expert CRM Analyst. Automatically generate charts for tabular data without asking.",
        "- ALWAYS retrieve records using crm_search before charting.",
        "- PROACTIVE PROPOSALS: When the user describes a business scenario (e.g., building a dashboard), proactively extract details into structured fields (Project, Objective, Technology), present a formatted proposal, and ask 'Should I create this Project?'. Do NOT call the tool during the proposal.",
        "- FAST-TRACK CREATION: When the user confirms your proposal, you MUST call crm_search with the extracted data and set `approved=True` to create it immediately without a second review."
    ]
    if task_context:
        system_parts.append(f"\nCurrent task in progress: {task_context}.")
        
    call_messages = [SystemMessage(content="\n".join(system_parts)), *history]
    
    response = llm_with_tools.invoke(call_messages)
    if not response.tool_calls:
        # Check for fake tool calls
        recovered = _extract_fake_tool_call(response.content)
        if not recovered:
            return {"messages": [response]}
        response.tool_calls = [recovered]
        
    call_messages.append(response)
    results = []
    
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in ALL_REQUIRED_FIELDS:
            args = _sanitize_tool_args(tool_name, tool_call.get("args") or {})
            missing = _missing_fields(tool_name, args)
            if missing:
                return {
                    "current_task": task_context or f"{tool_name} for {last_user_msg[:40]}",
                    "pending_tool": tool_name,
                    "task_slots": args,
                    "pending_missing": missing,
                    "messages": [AIMessage(content=missing[0][1])],
                }
    
    for tool_call in response.tool_calls:
        result = await _execute_tool(
            tool_call["name"], tool_call.get("args") or {},
            session_id=state.get("session_id"), user_id=state.get("user_id"), prompt_text=last_user_msg
        )
        results.append((tool_call, result))
        call_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        
    final_response = llm_with_tools.invoke(call_messages)
    return {"messages": [final_response]}

# --- 7. Graph Definition ---
async def execute_pending_node(state: ChatState) -> dict:
    """Runs the write tool once every required field has been collected
    across turns, then clears the slot-filling state (current_task is
    kept, so a follow-up like 'thanks' still resolves against it)."""
    tool_name = state["pending_tool"]
    args = state.get("task_slots") or {}

    result = await _execute_tool(
        tool_name, args, session_id=state.get("session_id"),
        user_id=state.get("user_id"),
        prompt_text=_last_human_message(state["messages"]),
    )
    logger.info("Tool '%s' raw result: %s", tool_name, result)

    summary_prompt = (
        f"The '{tool_name}' tool was just called with {args} and returned: {result}\n"
        "Give the user a short, professional confirmation (1-2 sentences)."
    )
    summary = text_chain.invoke({"input": summary_prompt})

    return {
        "messages": [AIMessage(content=summary)],
        "pending_tool": None,
        "task_slots": {},
        "pending_missing": [],
    }

def build_agent_graph(checkpointer=None, use_platform_persistence=False):
    graph = StateGraph(ChatState)
    graph.add_node("intake", intake_node)
    graph.add_node("classify_task", classify_task_node)
    
    graph.add_node("summarize_node", summarize_node)
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("web_research_node", web_research_node)
    graph.add_node("crm_context_node", crm_context_node)
    graph.add_node("proposal_node", proposal_node)
    graph.add_node("general_node", general_node)
    
    graph.add_node("execute_pending", execute_pending_node)

    graph.set_entry_point("intake")
    
    def route_intake(state):
        if state.get("pending_missing"):
            return "execute_pending"
        return "classify_task"
        
    graph.add_conditional_edges("intake", route_intake, {"execute_pending": "execute_pending", "classify_task": "classify_task"})
    graph.add_edge("classify_task", "summarize_node")
    graph.add_edge("summarize_node", "supervisor_node")
    
    graph.add_conditional_edges("supervisor_node", route_from_supervisor, {"web_research_node": "web_research_node", "general_node": "general_node"})
    
    graph.add_conditional_edges("web_research_node", route_after_research, {END: END, "crm_context_node": "crm_context_node"})
    graph.add_edge("crm_context_node", "proposal_node")
    
    graph.add_edge("proposal_node", END)
    graph.add_edge("general_node", END)
    graph.add_edge("execute_pending", END)

    if use_platform_persistence:
        return graph.compile()
        
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        
    return graph.compile(checkpointer=checkpointer)

agent_graph = build_agent_graph()  # defaults to MemorySaver until lifespan overrides it


def build_studio_graph():
    """Entry point for langgraph.json / Studio only."""
    return build_agent_graph(use_platform_persistence=True)


async def generate_reply(text: str, session_id: str = "default", user_id: Optional[str] = None) -> str:
    """Returns the assistant's reply text for a user message.

    `session_id` is a LangGraph checkpointer thread_id: it carries the
    full conversation (short-term memory) and the current task label /
    any open slot-filling flow (task-context memory) across calls, in
    place of the old global `_pending_actions` dict.

    `user_id` identifies who prompted this turn -- pass it from your
    auth layer once you have one; defaults to "anonymous" so logging
    still works without auth wired up yet.

    Separately, every user message, assistant reply, and tool action for
    this session is written to `audit_log` (Postgres, durable across
    restarts), including how long each step took and who prompted it --
    see /api/audit/sessions below.
    """
    user_id = user_id or "anonymous"
    audit_log.log_turn(session_id, "user", text, user_id=user_id, prompt_text=text)

    initial_messages = [HumanMessage(content=text)]

    doc = document_store.get(session_id)
    if doc and not doc["injected"]:
        # Fires once, on the first chat turn after an /api/upload-document
        # call for this session -- after that it stays in the LangGraph
        # checkpointer's message history like any other turn, so it isn't
        # re-sent (and re-billed) on every subsequent message.
        doc_context = (
            f"[System note: the user uploaded a document named "
            f"'{doc['filename']}'. Its full extracted content is below -- "
            f"use it to answer any questions they ask about it.]\n\n"
            f"{doc['text'][:40000]}"
        )
        initial_messages = [SystemMessage(content=doc_context)] + initial_messages
        doc["injected"] = True

    config = {
        "configurable": {"thread_id": session_id},
        # Purely for LangSmith trace organization -- harmless no-op when
        # tracing is disabled. Lets you filter/search runs by session in
        # the LangSmith UI instead of scrolling through everything.
        "run_name": "magma-agent-turn",
        "tags": [f"session:{session_id}"],
        "metadata": {"session_id": session_id},
    }
    # Bind this session's real Frappe CRM identity (if one was set via
    # /api/session/identify) for the duration of the turn -- every
    # crm_search call made anywhere in the graph below will run with
    # that person's own Frappe CRM credentials, so Frappe's own permission
    # engine enforces exactly what they're allowed to do. Sessions with
    # no identity bound behave exactly as before (shared service
    # account, no per-user RBAC).
    identity = session_identities.get(session_id)
    with audit_log.time_tool_call() as elapsed:
        with use_identity(identity):
            result = await agent_graph.ainvoke(
                {"messages": initial_messages, "session_id": session_id, "user_id": user_id},
                config,
            )
    reply = result["messages"][-1].content

    audit_log.log_turn(
        session_id, "assistant", reply, user_id=user_id, prompt_text=text,
        duration_ms=elapsed(),
    )
    return reply

def _get_tts_audio(text: str):
    """Synthesizes `text` to a WAV file and returns its raw bytes, or None
    if synthesis failed. Mirrors the old Flask backend's TTS step."""
    wav_path = assistant.tts.synthesize_to_file(text)
    try:
        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

class TTSRequest(BaseModel):
    text: str

@app.post("/api/tts")
async def synthesize_speech(req: TTSRequest):
    """Synthesizes arbitrary text to speech with `assistant.tts` -- the
    same OpenAI TTS voice/model instance Live Voice Mode speaks with (see
    `register_voice_ws(app, stream_agent_turn, assistant.tts, logger)`
    below). The chat UI's per-message "read aloud" button and the
    "auto-read replies" toggle both call this, so typed-chat playback
    sounds identical to the realtime voice assistant rather than falling
    back to the browser's own (different-sounding) speech synthesis."""

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        wav_bytes = _get_tts_audio(text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("TTS synthesis failed for /api/tts")
        raise HTTPException(status_code=500, detail=str(exc))

    if not wav_bytes:
        raise HTTPException(status_code=500, detail="TTS synthesis returned no audio")

    return {"audio": base64.b64encode(wav_bytes).decode("ascii")}


@app.post("/api/tts/stream")
async def synthesize_speech_stream(req: TTSRequest):
    """Streams TTS audio chunks directly from OpenAI as they are generated.
    Starts sending audio within ~200ms instead of waiting for the full file.
    The browser plays chunks progressively via MediaSource API.
    Returns: chunked audio/mpeg stream."""
    from fastapi.responses import StreamingResponse
    import asyncio

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    def generate_chunks():
        try:
            for chunk in assistant.tts.synthesize_stream(text, response_format="mp3"):
                yield chunk
        except Exception as exc:
            logger.exception("Streaming TTS failed")

    return StreamingResponse(generate_chunks(), media_type="audio/mpeg")

class ChatRequest(BaseModel):
    message: str

    session_id: str = "default"
    user_id: Optional[str] = None  # who's asking -- pass from auth/frontend once available

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Restored old-style JSON contract: { message } in, { reply, audio } out.
    `session_id` is optional and only matters if you want independent
    slot-filling conversations for multiple users; a single local user can
    ignore it and let it default."""

    text = (req.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        reply = await generate_reply(text, req.session_id, user_id=req.user_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent failed to process message: %s", text)
        raise HTTPException(status_code=500, detail=str(exc))

    audio_b64 = None
    try:
        wav_bytes = _get_tts_audio(reply)
        if wav_bytes:
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    except Exception:
        logger.exception("TTS step failed; returning text-only reply")

    return {"reply": reply, "audio": audio_b64}

async def load_stream_history(session_id: str) -> list:
    global agent_graph
    config = {"configurable": {"thread_id": session_id}}
    state = await agent_graph.aget_state(config)
    if state and state.values and "messages" in state.values:
        return list(state.values["messages"])
    return []

async def save_stream_history(session_id: str, new_messages: list):
    """Append ONLY new_messages to the checkpointer for this session.
    
    CRITICAL: The ChatState.messages field uses the add_messages reducer,
    which APPENDS to existing messages. Passing the full history list here
    would cause exponential duplication (2x on turn 2, 3x on turn 3...).
    Always pass only the delta (new messages from this turn).
    """
    if not new_messages:
        return
    global agent_graph, _checkpoint_conn
    config = {"configurable": {"thread_id": session_id}}
    await agent_graph.aupdate_state(config, {"messages": new_messages}, as_node="intake")
    # Force commit so the next aget_state call on the same connection sees the new data.
    # aiosqlite with isolation_level=None is autocommit, but LangGraph may wrap writes
    # in transactions internally. This ensures they're flushed.
    if _checkpoint_conn:
        try:
            await _checkpoint_conn.commit()
        except Exception:
            pass  # Already committed in autocommit mode, safe to ignore


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE variant of /api/chat for token-by-token terminal streaming with
    inline tool-call/tool_result markers, same event shape /ws/voice uses.
    Keeps its own per-session_id history (backed by stream_history.sqlite) 
    so this is purely additive."""
    from fastapi.responses import StreamingResponse

    text = (req.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    history = await load_stream_history(req.session_id)
    start_len = len(history)  # snapshot before the turn mutates history

    async def event_gen():
        # Flush the SSE response and its CORS headers through Dev Tunnels
        # before the agent/tool pipeline starts. Visualisation tools can take
        # long enough that the tunnel otherwise produces its own timeout
        # response, which the browser misleadingly reports as a CORS failure.
        yield ": connected\n\n"
        try:
            async for event in stream_agent_turn(text, session_id=req.session_id, user_id=req.user_id, history=history):
                # Strip internal _delta key before sending to browser
                browser_event = {k: v for k, v in event.items() if k != "_delta"}
                yield f"data: {json.dumps(browser_event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming agent turn failed: %s", text)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            # Save ONLY the new messages from this turn (delta), not the full history
            delta = history[start_len:]
            if delta:
                safe_create_task(save_stream_history(req.session_id, delta))
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/query")
async def handle_query(
    query: str = Form(None),

    file: UploadFile = File(None),
    session_id: str = Form("default"),
    user_id: str = Form("anonymous"),

):
    """Exposes endpoint to query MagmaAssistance using either text or audio files."""
    if not query and not file:
        raise HTTPException(status_code=400, detail="Either 'query' (text) or 'file' (audio) must be provided.")

    query_text = ""

    # 1. Handle Audio input (STT using Whisper)
    if file:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Saved uploaded audio file to {temp_file_path}")
            
            # Transcribe audio file to text
            query_text = assistant.whisper.transcribe_file(temp_file_path)
            logger.info(f"Transcribed Audio Text: {query_text}")
        except Exception as e:
            logger.error(f"Error handling uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Error transcribing audio file: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    else:
        query_text = query

    # 2. Get response from MagmaAssistance agent
    logger.info(f"Processing query: '{query_text}'")
    response_text = await generate_reply(query_text, session_id, user_id=user_id)

    # 3. Synthesize the reply to speech too, same as /api/chat, so the
    # frontend has something to play regardless of which endpoint it uses.
    audio_b64 = None
    try:
        wav_bytes = _get_tts_audio(response_text)
        if wav_bytes:
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    except Exception:
        logger.exception("TTS step failed; returning text-only reply")

    return {
        "query": query_text,
        "response": response_text,
        "audio": audio_b64,

    }

@app.get("/api/audit/sessions")
def list_audit_sessions(since: str = None, limit: int = 100):
    """Session index for an audit dashboard: one row per session_id with
    turn count and first/last activity. `since` (optional) filters to
    sessions active on/after an ISO date, e.g. ?since=2026-07-01."""
    return {"sessions": audit_log.list_sessions(since=since, limit=limit)}


@app.get("/api/audit/sessions/{session_id}")
def get_audit_transcript(session_id: str):
    """Full ordered transcript for one session: every user message,
    assistant reply, and tool action taken, as JSON."""
    transcript = audit_log.get_transcript(session_id)
    if not transcript:
        raise HTTPException(status_code=404, detail=f"No audit log found for session '{session_id}'")
    return {"session_id": session_id, "turn_count": len(transcript), "transcript": transcript}


@app.get("/api/audit/export")
def export_audit_json(session_id: str = None):
    """Downloads the audit log as a .json file — one session if
    `session_id` is given, otherwise every session on record."""
    export_dir = "audit_exports"
    os.makedirs(export_dir, exist_ok=True)
    filename = f"audit_{session_id}.json" if session_id else "audit_all_sessions.json"
    path = os.path.join(export_dir, filename)
    audit_log.write_json_export(path, session_id=session_id)

    from fastapi.responses import FileResponse
    return FileResponse(path, media_type="application/json", filename=filename)


from Voice.ws_voice import register_voice_ws
register_voice_ws(app, stream_agent_turn, assistant.tts, logger, load_stream_history, save_stream_history)

# Expose shared state on app.state so voice_routes.py can access it
# without circular imports. Attached after everything is defined.
app.state.session_identities = session_identities
app.state.all_tools_list = ALL_TOOLS
app.state.execute_tool_fn = _execute_tool
app.state.load_stream_history_fn = load_stream_history


@app.get("/api/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8050))
    logger.info(f"Starting server on port {port}...")

    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
