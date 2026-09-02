"""Streaming chat completion for the SSE/voice path, independent of the LangGraph checkpointer."""
import asyncio
import json
import os
import re

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages

from . import agent_setup, config, state, tool_exec
from .history import approx_tokens
from .llm_model import _clean_schema_for_openai, convert_message_to_dict
from .logging_setup import logger

_FAKE_NAME_RE = re.compile(r'"name"\s*:\s*"(?P<name>[a-zA-Z_][\w\-.]*)"')
_FAKE_KV_STR_RE = re.compile(r'"(?P<key>[a-zA-Z_]\w*)"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"')
_FAKE_KV_NUM_RE = re.compile(r'"(?P<key>[a-zA-Z_]\w*)"\s*:\s*(?P<value>-?\d+(?:\.\d+)?)\b')


def extract_fake_tool_call(content: str):
    """Recovers a tool call from a local model's plain-text approximation of one."""
    text = (content or "").strip()
    if not text.startswith("{") or '"name"' not in text or "parameters" not in text.lower():
        return None

    name_match = _FAKE_NAME_RE.search(text)
    if not name_match:
        return None

    name = name_match.group("name")
    if name not in agent_setup.tool_map:
        return None

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


async def _stream_chat_completion(messages, tools=None):
    """Yields token/done events for one raw streamed completion call."""
    api_messages = [convert_message_to_dict(m) for m in messages]
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    env_openai_key = os.environ.get("OPENAI_API_KEY")
    key = openrouter_key or env_openai_key
    is_openrouter = bool(openrouter_key)
    if is_openrouter:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "http://localhost:8050", "X-Title": "MagmaAssistance"}
        url = "https://openrouter.ai/api/v1/chat/completions"
        model_name = config.LLM_MODEL if "/" in config.LLM_MODEL else f"openai/{config.LLM_MODEL}"
    else:
        headers = {"Authorization": f"Bearer {env_openai_key}", "Content-Type": "application/json"}
        url = "https://api.openai.com/v1/chat/completions"
        model_name = config.LLM_MODEL
    data = {"model": model_name, "messages": api_messages, "temperature": agent_setup.assistant.llm.temperature, "stream": True, "max_tokens": config.LLM_MAX_TOKENS}
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
    """Continues a streamed reply across rounds until finish_reason != 'length'."""
    content = ""
    tool_calls = []
    messages = call_messages
    for round_number in range(config.MAX_COMPLETION_ROUNDS + 1):
        piece = ""
        async for event in _stream_chat_completion(messages, tools=tools):
            if event["type"] == "token":
                yield {"type": "token", "text": event["text"]}
            else:
                piece = event["content"]
                tool_calls = event["tool_calls"]
                finish_reason = event["finish_reason"]

        content += piece

        if finish_reason != "length" or tool_calls or round_number == config.MAX_COMPLETION_ROUNDS:
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


def build_document_context_note(session_id, query=None):
    # Retrieval-based system-prompt note for the session's uploaded document
    if not session_id or not agent_setup.doc_rag.has_document(session_id):
        return None

    filename = agent_setup.doc_rag.get_filename(session_id)
    chunks = agent_setup.doc_rag.retrieve(session_id, query) if query else []

    instructions = (
        f"\n[System note: the user has an uploaded document named "
        f"'{filename}' available for this entire conversation. Relevant "
        f"excerpts retrieved for the current turn are below (if any). Use "
        f"them to answer questions about the document, now or later in this "
        f"conversation. Also treat retrieved excerpts as a possible source "
        f"of CRM record data: if the user's message (or the document's own "
        f"content) asks you to create or update a Lead, Deal, Organization, "
        f"or Contact, pull the relevant fields out of the excerpts yourself "
        f"and proceed via crm_create/crm_update exactly as you would for "
        f"any other request -- run crm_research_company first if it names a "
        f"company, show what you found for confirmation, then create. If a "
        f"Deal or other record refers to an Organization (or other linked "
        f"record) that doesn't exist yet, this follows the same "
        f"missing-dependency flow described earlier: crm_create will come "
        f"back with invalid_link_fields/creatable=true, so ask the user to "
        f"confirm creating that Organization first, create it once they "
        f"confirm, then retry the original create with the same value -- "
        f"never invent data that isn't actually present in the retrieved "
        f"excerpts or the conversation. If the excerpts below don't cover "
        f"what's needed, ask the user for it instead of guessing.]"
    )

    if not chunks:
        return instructions

    excerpts = "\n\n---\n\n".join(chunks)
    return f"{instructions}\n\n{excerpts}"


async def stream_agent_turn(text, session_id=None, user_id=None, history=None, task_context=None):
    """Streaming twin of general_node/generate_reply for the realtime voice WS
    path. Keeps its own caller-owned history list instead of the LangGraph
    checkpointer. Yields token/tool_call/tool_result/done event dicts."""
    history = history if history is not None else []
    start_len = len(history)
    history.append(HumanMessage(content=text))
    trimmed = trim_messages(history, max_tokens=config.MAX_HISTORY_TOKENS, token_counter=approx_tokens, strategy="last", include_system=False)

    candidate_tools = []
    if agent_setup.ALL_TOOLS:
        candidate_tools = list(agent_setup.ALL_TOOLS) if len(agent_setup.ALL_TOOLS) <= config.TOOL_RAG_BYPASS_THRESHOLD else (agent_setup.tool_rag.retrieve(text) if agent_setup.tool_rag else [])

    openai_tools = None
    if candidate_tools:
        from langchain_core.utils.function_calling import convert_to_openai_tool
        openai_tools = []
        for t in candidate_tools:
            formatted = convert_to_openai_tool(t)
            if "function" in formatted and "parameters" in formatted["function"]:
                formatted["function"]["parameters"] = _clean_schema_for_openai(formatted["function"]["parameters"])
            openai_tools.append(formatted)

    system_parts = [agent_setup.assistant.llm.system_prompt]
    if task_context:
        system_parts.append(f"\nCurrent task in progress: {task_context}.")

    doc_note = build_document_context_note(session_id, query=text)
    if doc_note:
        system_parts.append(doc_note)

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
                yield {"type": "done", "text": content, "_delta": history[start_len:]}
                return

            if round_number == max_rounds:
                break

            ai_msg = AIMessage(content=content, tool_calls=tool_calls)
            call_messages.append(ai_msg)
            history.append(ai_msg)
            for tc in tool_calls:
                yield {"type": "tool_call", "name": tc["name"], "args": tc.get("args") or {}}
                result = await tool_exec.execute_tool(tc["name"], tc.get("args") or {}, session_id=session_id, user_id=user_id, prompt_text=text)
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
        # Drop a dangling AIMessage(tool_calls) or an incomplete ToolMessage run so a
        # cancelled turn never leaves OpenAI a message sequence it will 400 on next time.
        if history:
            last_msg = history[-1]
            if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                history.pop()
            elif isinstance(last_msg, ToolMessage):
                tool_msg_count = 0
                for msg in reversed(history):
                    if isinstance(msg, ToolMessage):
                        tool_msg_count += 1
                    else:
                        break

                if len(history) > tool_msg_count:
                    ai_msg = history[-(tool_msg_count + 1)]
                    if isinstance(ai_msg, AIMessage) and getattr(ai_msg, "tool_calls", None):
                        if len(ai_msg.tool_calls) != tool_msg_count:
                            for _ in range(tool_msg_count + 1):
                                history.pop()
        raise