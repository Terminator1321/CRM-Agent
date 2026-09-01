"""LangGraph state schema plus small helpers used by multiple nodes."""
import ast
import json
import re
from typing import Annotated, Any, Dict, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages

from ..logging_setup import logger


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    current_task: Optional[str]
    task_slots: dict
    pending_tool: Optional[str]
    pending_missing: list
    session_id: str
    user_id: Optional[str]

    intent_category: Optional[str]
    extracted_entities: dict
    research_context: Optional[str]
    crm_context: Optional[str]
    proposal: Optional[str]


def last_human_message(messages) -> Optional[str]:
    """Returns the content of the most recent HumanMessage, if any."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return None


def parse_json_loose(text: str) -> dict:
    """Parses a classifier reply as JSON, stripping any ```json fence first."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


CLASSIFY_SYSTEM = (
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


def is_unqualified_approval(message: str) -> bool:
    """Recognises an approval without treating a correction as consent."""
    text = (message or "").strip().lower()
    if not text:
        return False
    if re.search(r"\b(no|wrong|incorrect|change|correct|instead|but|except|update|modify|add)\b", text):
        return False
    return bool(re.search(r"\b(yes|approve|approved|proceed|continue|go ahead|create)\b", text))


def build_fallback_chart(results: list, query_lower: str) -> Optional[str]:
    """Generic fallback chart generator: detects category/numeric keys in any
    list of records returned by a tool and builds a bar/line/pie chart spec."""
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

            priority_cat = [
                "customer", "customer_name", "supplier", "supplier_name",
                "item_code", "item_name", "production_item", "warehouse",
                "territory", "item_group", "posting_date", "transaction_date"
            ]
            best_cat = next((k for p in priority_cat for k in cat_keys if k.lower() == p or p in k.lower()), cat_keys[0] if cat_keys else None)

            priority_num = [
                "grand_total", "total", "net_total", "amount",
                "qty", "produced_qty", "stock_qty", "rate", "valuation_rate"
            ]
            best_num = next((k for p in priority_num for k in num_keys if k.lower() == p or p in k.lower()), num_keys[0] if num_keys else None)

            date_key = next((k for k in avail_keys if any(dw in k.lower() for dw in ["date", "posting", "transaction"])), None)
            if best_cat and date_key:
                unique_cats = {str(r.get(best_cat) or "").strip() for r in data}
                if len(unique_cats) <= 1:
                    best_cat = date_key

            if best_cat and best_num:
                label_key = best_cat
                value_key = best_num

                totals: Dict[str, float] = {}
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

            if cat_keys:
                status_key = next((k for k in cat_keys if "status" in k.lower() or "group" in k.lower() or "type" in k.lower()), cat_keys[0])
                counts: Dict[str, int] = {}
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
