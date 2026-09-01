"""Sanitizes tool call args and runs every tool call through one audited path."""
import db.postgres_audit_log as audit_log

from . import agent_setup
from .logging_setup import logger

# Values a model sometimes invents instead of leaving a field blank.
_PLACEHOLDER_VALUES = {
    "default", "n/a", "na", "none", "null", "unknown",
    "not specified", "not sure", "unspecified", "todo", "tbd", "-",
}


def missing_fields(tool_name: str, args: dict) -> list:
    """Ordered (field, question) pairs required for tool_name that are absent,
    empty, or filled with a placeholder-like guess."""
    required = agent_setup.ALL_REQUIRED_FIELDS.get(tool_name, [])
    args = args or {}
    missing = []
    for field, question in required:
        value = args.get(field)
        if not value:
            missing.append((field, question))
        elif isinstance(value, str) and value.strip().lower() in _PLACEHOLDER_VALUES:
            missing.append((field, question))
    return missing


def _flatten_scalar(value):
    """Unwraps a scalar arg that a model wrapped in a single-key dict."""
    if isinstance(value, dict):
        if len(value) == 1:
            return _flatten_scalar(next(iter(value.values())))
        for key in ("value", "name", "text", "input"):
            if key in value:
                return _flatten_scalar(value[key])
        return str(value)
    return value


def sanitize_tool_args(tool_name: str, args: dict) -> dict:
    """Unwraps mis-shaped scalar args and drops empty strings for numeric fields."""
    if not args:
        return args

    tool = agent_setup.tool_map.get(tool_name)
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

        expects_scalar = any(t in (str, int, float, bool) for t in inner_types)

        if isinstance(value, dict) and expects_scalar:
            value = _flatten_scalar(value)
            cleaned[field_name] = value

        if value == "":
            if int in inner_types or float in inner_types:
                cleaned.pop(field_name, None)

    return cleaned


async def execute_tool(
    tool_name: str,
    args: dict,
    session_id: str = None,
    user_id: str = None,
    prompt_text: str = None,
):
    """Runs one tool call and writes it to the audit log regardless of outcome."""
    tool = agent_setup.tool_map.get(tool_name)
    if tool is None:
        result = f"Tool '{tool_name}' is not available."
        if session_id:
            audit_log.log_turn(
                session_id, "tool", result, tool_name=tool_name, tool_args=args,
                user_id=user_id, prompt_text=prompt_text, tool_status="not_found",
            )
        return result
    try:
        effective_args = sanitize_tool_args(tool_name, args) or {}
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
        failure = f"'{tool_name}' failed: {detail}" if detail else f"'{tool_name}' failed to fetch CRM data right now."
        if session_id:
            audit_log.log_turn(
                session_id, "tool", failure, tool_name=tool_name, tool_args=args,
                user_id=user_id, prompt_text=prompt_text, tool_status="error",
                error_message=detail,
            )
        return failure
