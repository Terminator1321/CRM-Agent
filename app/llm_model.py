"""OpenAI/OpenRouter-backed BaseChatModel, and message/schema conversion helpers."""
import json
import os
from typing import Any, List, Optional, Sequence, Union

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from . import config
from .logging_setup import logger


def convert_message_to_dict(message):
    """Converts a LangChain message object into an OpenAI chat-completion dict."""
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
    """Strips fields from a tool JSON schema that make OpenAI's API 400."""
    if not isinstance(schema, dict):
        return schema

    cleaned = schema.copy()
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

    for union_key in ("anyOf", "oneOf", "allOf"):
        if union_key in cleaned and isinstance(cleaned[union_key], list):
            cleaned[union_key] = [
                _clean_schema_for_openai(branch) if isinstance(branch, dict) else branch
                for branch in cleaned[union_key]
            ]

    return cleaned


class OpenAIChatModel(BaseChatModel):
    """Talks to OpenAI or OpenRouter directly, with tool-choice forwarding and
    finish_reason=="length" continuation handling."""

    model_name: str
    temperature: float
    api_key: str
    base_url: str
    bound_tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None

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
            "max_tokens": config.LLM_MAX_TOKENS,
        }

        if self.bound_tools:
            data["tools"] = self.bound_tools
            if self.tool_choice is not None:
                tc = self.tool_choice
                if isinstance(tc, str) and tc not in ("auto", "none", "required", "any"):
                    tc = {"type": "function", "function": {"name": tc}}
                elif tc == "any":
                    tc = "required"
                data["tool_choice"] = tc

        content = ""
        tool_calls = []
        message_data = {}
        for round_number in range(config.MAX_COMPLETION_ROUNDS + 1):
            try:
                response = requests.post(target_url, json=data, headers=headers, timeout=config.LLM_REQUEST_TIMEOUT_SECONDS)
            except requests.exceptions.Timeout:
                logger.error(
                    "LLM API call timed out after %ss with no response (%s)",
                    config.LLM_REQUEST_TIMEOUT_SECONDS, target_url,
                )
                raise RuntimeError(
                    f"The AI model didn't respond within {int(config.LLM_REQUEST_TIMEOUT_SECONDS)}s. "
                    "Please try again."
                )

            if not response.ok:
                logger.error(f"LLM API Rejected Request ({response.status_code}): {response.text}")
                try:
                    error_body = response.json()
                    error_detail = (
                        error_body.get("error", {}).get("message")
                        if isinstance(error_body.get("error"), dict)
                        else error_body.get("error") or response.text
                    )
                except Exception:
                    error_detail = response.text
                raise RuntimeError(
                    f"OpenAI API rejected the request ({response.status_code}): "
                    f"{str(error_detail)[:500]}"
                )

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

            if finish_reason != "length" or tool_calls or round_number == config.MAX_COMPLETION_ROUNDS:
                if finish_reason == "length":
                    logger.warning(
                        "LLM reply still truncated after %d continuation round(s); "
                        "returning what we have.", round_number
                    )
                break

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
        tools: Sequence[Union[dict, type[BaseModel], Any]],
        **kwargs: Any,
    ) -> "OpenAIChatModel":
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted_tools = []
        for t in tools:
            formatted = convert_to_openai_tool(t)
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
            tool_choice=kwargs.get("tool_choice"),
        )
