"""LangGraph nodes, graph assembly, and the two entrypoints (generate_reply,
load/save_stream_history) that drive a text-chat turn."""
import datetime as dt
from typing import Dict, Literal, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from CRM_Unified.crm_client import use_identity
import db.postgres_audit_log as audit_log

from .. import agent_setup, config, state, tool_exec
from ..history import approx_tokens
from ..llm_model import OpenAIChatModel
from ..logging_setup import logger
from ..streaming import build_document_context_note, extract_fake_tool_call
from .chat_state import ChatState, CLASSIFY_SYSTEM, last_human_message, parse_json_loose


def intake_node(state_in: ChatState) -> dict:
    """If a slot-filling flow is open, treats this turn's message as the
    answer to the next missing field."""
    pending_tool = state_in.get("pending_tool")
    pending_missing = state_in.get("pending_missing") or []
    if not pending_tool or not pending_missing:
        return {}

    last_user_msg = last_human_message(state_in["messages"]) or ""
    field, _question = pending_missing[0]
    parser = agent_setup.ALL_FIELD_PARSERS.get((pending_tool, field))
    value = parser(last_user_msg) if parser else last_user_msg.strip()

    slots = {**(state_in.get("task_slots") or {}), field: value}
    remaining = pending_missing[1:]

    if remaining:
        return {
            "task_slots": slots,
            "pending_missing": remaining,
            "messages": [AIMessage(content=remaining[0][1])],
        }

    return {"task_slots": slots, "pending_missing": []}


def _route_after_intake(state_in: ChatState) -> str:
    if state_in.get("pending_tool"):
        return END if state_in.get("pending_missing") else "execute_pending"
    return "classify_task"


def classify_task_node(state_in: ChatState) -> dict:
    """Keeps current_task alive across turns of the same task; resets it
    only when the topic genuinely changes."""
    last_user_msg = last_human_message(state_in["messages"])
    if last_user_msg is None:
        return {}

    current_task = state_in.get("current_task")
    if current_task is None:
        return {"current_task": last_user_msg[:60]}

    prompt = f"Active task: {current_task}\nNew user message: {last_user_msg}"
    try:
        resp = agent_setup.assistant.llm.model.invoke(
            [SystemMessage(content=CLASSIFY_SYSTEM), HumanMessage(content=prompt)]
        )
        parsed = parse_json_loose(resp.content)
        task_label = parsed.get("task_label") or current_task
    except Exception as exc:
        logger.warning("Task classification failed (%s); assuming same task.", exc)
        task_label = current_task

    return {"current_task": task_label}


def _plain_reply(history, task_context: Optional[str] = None) -> str:
    """Used when no CRM tool is needed for this turn, sending the full
    trimmed history so injected document context is still visible."""
    system_parts = [agent_setup.assistant.llm.system_prompt, f"Current date: {dt.datetime.now().astimezone():%Y-%m-%d}."]
    if task_context:
        system_parts.append(f"\nCurrent task in progress: {task_context}.")
    call_messages = [SystemMessage(content="\n".join(system_parts)), *history]
    response = agent_setup.assistant.llm.model.invoke(call_messages)
    return response.content


def summarize_node(state_in: ChatState) -> dict:
    """Rolls older messages into a running summary once history grows past 10 messages."""
    messages = state_in.get("messages", [])
    summary = state_in.get("summary", "")

    if len(messages) > 10:
        to_summarize = messages[:-4]

        prompt = (
            f"Summarize the following conversation history. "
            f"Include any important entities, facts, or context.\n"
            f"Previous Summary: {summary}\n"
            f"New Messages: {to_summarize}"
        )
        new_summary_msg = agent_setup.assistant.llm.model.invoke(prompt)
        new_summary = new_summary_msg.content

        delete_messages = [RemoveMessage(id=m.id) for m in to_summarize if getattr(m, 'id', None)]
        return {"summary": new_summary, "messages": delete_messages}
    return {}


class IntentOutput(BaseModel):
    category: Literal["chitchat", "crm_query", "crm_write", "web_search"]
    record_type: Optional[str] = None
    entities: Dict[str, Union[str, int, float, bool, None]] = {}


def supervisor_node(state_in: ChatState) -> dict:
    """Classifies the turn's intent and extracts entities for the downstream nodes."""
    last_user_msg = last_human_message(state_in["messages"]) or ""
    summary = state_in.get("summary", "")

    context_msg = f"Context: {summary}\n" if summary else ""

    from LLM.LLM import INTENT_SYSTEM_PROMPT
    llm_intent = agent_setup.assistant.llm.model.with_structured_output(IntentOutput)

    intent = llm_intent.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=context_msg + last_user_msg)
    ])

    logger.info("Intent classified: category=%s, entities=%s", intent.category, intent.entities)

    return {
        "intent_category": intent.category,
        "extracted_entities": intent.entities
    }


def route_from_supervisor(state_in: ChatState) -> str:
    cat = state_in.get("intent_category")
    if cat in ("crm_write", "web_search"):
        return "web_research_node"
    return "general_node"


async def web_research_node(state_in: ChatState) -> dict:
    """Runs web-search tools to research the entities the supervisor extracted."""
    from LLM.LLM import RESEARCH_SYSTEM_PROMPT
    entities = state_in.get("extracted_entities", {})
    last_user_msg = last_human_message(state_in["messages"]) or ""

    web_tools = [t for t in agent_setup.ALL_TOOLS if t.name in ("web_search", "web_company_search", "web_fetch_page")]
    search_llm = agent_setup.assistant.llm.model.bind_tools(web_tools)

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
            result = await tool_exec.execute_tool(
                tc["name"], tc.get("args") or {},
                session_id=state_in.get("session_id"), user_id=state_in.get("user_id"),
                prompt_text=last_user_msg
            )
            call_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    final_response = search_llm.invoke(call_messages)

    if state_in.get("intent_category") == "web_search":
        return {"messages": [AIMessage(content=final_response.content)], "research_context": final_response.content}
    return {"research_context": final_response.content}


def route_after_research(state_in: ChatState) -> str:
    if state_in.get("intent_category") == "web_search":
        return END
    return "crm_context_node"


async def crm_context_node(state_in: ChatState) -> dict:
    """Looks up internal CRM context for the entities the supervisor extracted."""
    entities = state_in.get("extracted_entities", {})
    if not entities:
        return {"crm_context": "No entities to lookup"}
    crm_tools = [t for t in agent_setup.ALL_TOOLS if t.name.startswith("crm_")]
    context_llm = agent_setup.assistant.llm.model.bind_tools(crm_tools)
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
            result = await tool_exec.execute_tool(tc["name"], tc.get("args") or {}, session_id=state_in.get("session_id"), user_id=state_in.get("user_id"))
            call_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        final = context_llm.invoke(call_messages)
        return {"crm_context": final.content}
    return {"crm_context": response.content}


def proposal_node(state_in: ChatState) -> dict:
    """Drafts a create/update proposal from the research + CRM context gathered so far."""
    from LLM.LLM import PROPOSAL_SYSTEM_PROMPT
    entities = state_in.get("extracted_entities", {})
    research = state_in.get("research_context", "")
    crm_ctx = state_in.get("crm_context", "")
    prop_response = agent_setup.assistant.llm.model.invoke([
        SystemMessage(content=PROPOSAL_SYSTEM_PROMPT),
        HumanMessage(content=f"Entities: {entities}\nWeb Research: {research}\nCRM Context: {crm_ctx}")
    ])
    return {"messages": [AIMessage(content=prop_response.content)], "proposal": prop_response.content}


async def general_node(state_in: ChatState) -> dict:
    """Fallback/chitchat/CRM-query node: binds all tools, loops tool-call
    rounds until the model returns plain text."""
    last_user_msg = last_human_message(state_in["messages"]) or ""
    task_context = state_in.get("current_task")

    history = trim_messages(
        state_in["messages"],
        max_tokens=config.MAX_HISTORY_TOKENS,
        token_counter=approx_tokens,
        strategy="last",
        include_system=False,
        start_on="human"
    )

    candidate_tools = list(agent_setup.ALL_TOOLS)

    intent_cat = state_in.get("intent_category")
    if intent_cat in ("chitchat", "crm_query"):
        _WEB = {"web_search", "web_fetch_page", "web_crawl", "web_company_search"}
        candidate_tools = [t for t in candidate_tools if t.name not in _WEB]

    llm_with_tools = agent_setup.assistant.llm.model.bind_tools(candidate_tools)
    current_date_str = dt.date.today().strftime("%Y-%m-%d")
    system_parts = [
        agent_setup.assistant.llm.system_prompt,
        f"\nCURRENT SYSTEM DATE: {current_date_str}",
        "\nSENIOR CRM & BUSINESS INTELLIGENCE DIRECTIVE:",
        "- You are an expert CRM Analyst. Automatically generate charts for tabular data without asking.",
        "- ALWAYS retrieve records using crm_search before charting.",
        "- PROACTIVE PROPOSALS: When the user describes a business scenario (e.g., building a dashboard), proactively extract details into structured fields (Project, Objective, Technology), present a formatted proposal, and ask 'Should I create this Project?'. Do NOT call the tool during the proposal.",
        "- FAST-TRACK CREATION: When the user confirms your proposal, you MUST call crm_search with the extracted data and set `approved=True` to create it immediately without a second review."
    ]
    if task_context:
        system_parts.append(f"\nCurrent task in progress: {task_context}.")

    doc_note = build_document_context_note(state_in.get("session_id"))
    if doc_note:
        system_parts.append(doc_note)

    call_messages = [SystemMessage(content="\n".join(system_parts)), *history]

    response = llm_with_tools.invoke(call_messages)

    MAX_TOOL_ROUNDS = 4
    for round_number in range(MAX_TOOL_ROUNDS):
        if not response.tool_calls:
            recovered = extract_fake_tool_call(response.content)
            if not recovered:
                if not (response.content or "").strip():
                    return {"messages": [AIMessage(
                        content="I wasn't able to put together a response to that -- could you rephrase or try again?"
                    )]}
                return {"messages": [response]}
            response.tool_calls = [recovered]

        call_messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in agent_setup.ALL_REQUIRED_FIELDS:
                args = tool_exec.sanitize_tool_args(tool_name, tool_call.get("args") or {})
                missing = tool_exec.missing_fields(tool_name, args)
                if missing:
                    return {
                        "current_task": task_context or f"{tool_name} for {last_user_msg[:40]}",
                        "pending_tool": tool_name,
                        "task_slots": args,
                        "pending_missing": missing,
                        "messages": [AIMessage(content=missing[0][1])],
                    }

        for tool_call in response.tool_calls:
            result = await tool_exec.execute_tool(
                tool_call["name"], tool_call.get("args") or {},
                session_id=state_in.get("session_id"), user_id=state_in.get("user_id"), prompt_text=last_user_msg
            )
            call_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        response = llm_with_tools.invoke(call_messages)

    if response.tool_calls or not (response.content or "").strip():
        call_messages.append(SystemMessage(
            content="Stop calling tools now and answer the user directly with what you've found so far."
        ))
        response = OpenAIChatModel(
            model_name=llm_with_tools.model_name,
            temperature=llm_with_tools.temperature,
            api_key=llm_with_tools.api_key,
            base_url=llm_with_tools.base_url,
        ).invoke(call_messages)
        if not (response.content or "").strip():
            response = AIMessage(content="I gathered some information but couldn't finish putting together a full answer -- could you try asking again?")

    return {"messages": [response]}


async def execute_pending_node(state_in: ChatState) -> dict:
    """Runs the write tool once every required field has been collected
    across turns, then clears the slot-filling state."""
    tool_name = state_in["pending_tool"]
    args = state_in.get("task_slots") or {}

    result = await tool_exec.execute_tool(
        tool_name, args, session_id=state_in.get("session_id"),
        user_id=state_in.get("user_id"),
        prompt_text=last_human_message(state_in["messages"]),
    )
    logger.info("Tool '%s' raw result: %s", tool_name, result)

    summary_prompt = (
        f"The '{tool_name}' tool was just called with {args} and returned: {result}\n"
        "Give the user a short, professional confirmation (1-2 sentences)."
    )
    summary = agent_setup.text_chain.invoke({"input": summary_prompt})

    return {
        "messages": [AIMessage(content=summary)],
        "pending_tool": None,
        "task_slots": {},
        "pending_missing": [],
    }


def build_agent_graph(checkpointer=None, use_platform_persistence=False):
    """Assembles the intake -> classify -> supervisor -> (research/CRM) -> reply graph."""
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

    def route_intake(state_in):
        if state_in.get("pending_missing"):
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


state.agent_graph = build_agent_graph()  # default MemorySaver graph, until lifespan swaps it in


def build_studio_graph():
    """Entry point for langgraph.json / Studio only."""
    return build_agent_graph(use_platform_persistence=True)


async def generate_reply(text: str, session_id: str = "default", user_id: Optional[str] = None) -> str:
    """Runs one full agent turn for session_id and returns the reply text.
    session_id is the LangGraph checkpointer thread_id, carrying short-term
    (messages) and task-context (current_task/task_slots/pending_*) memory."""
    user_id = user_id or "anonymous"
    audit_log.log_turn(session_id, "user", text, user_id=user_id, prompt_text=text)

    initial_messages = [HumanMessage(content=text)]

    config_dict = {
        "configurable": {"thread_id": session_id},
        "run_name": "magma-agent-turn",
        "tags": [f"session:{session_id}"],
        "metadata": {"session_id": session_id},
    }
    identity = state.session_identities.get(session_id)
    with audit_log.time_tool_call() as elapsed:
        with use_identity(identity):
            result = await state.agent_graph.ainvoke(
                {"messages": initial_messages, "session_id": session_id, "user_id": user_id},
                config_dict,
            )
    reply = result["messages"][-1].content

    audit_log.log_turn(
        session_id, "assistant", reply, user_id=user_id, prompt_text=text,
        duration_ms=elapsed(),
    )
    return reply


async def load_stream_history(session_id: str) -> list:
    """Returns the checkpointed message history for session_id."""
    config_dict = {"configurable": {"thread_id": session_id}}
    graph_state = await state.agent_graph.aget_state(config_dict)
    if graph_state and graph_state.values and "messages" in graph_state.values:
        return list(graph_state.values["messages"])
    return []


async def save_stream_history(session_id: str, new_messages: list):
    """Appends ONLY the new messages from this turn to the checkpointer --
    add_messages is a reducer, so passing the full history would duplicate it."""
    if not new_messages:
        return
    config_dict = {"configurable": {"thread_id": session_id}}
    await state.agent_graph.aupdate_state(config_dict, {"messages": new_messages}, as_node="intake")
    if state.checkpoint_conn:
        try:
            await state.checkpoint_conn.commit()
        except Exception:
            pass