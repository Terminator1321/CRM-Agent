"""Reads CRM Lead/Deal/Task/Note/Call records and shapes them for the dashboard UI."""
from collections import Counter, defaultdict
from datetime import datetime, timezone

from CRM_Unified.crm_client import crm_client

_CLOSED_LEAD_STATUSES = {"lost", "junk", "unqualified", "converted"}
_CLOSED_TASK_STATUSES = {"done", "canceled", "cancelled"}
_TASK_PROGRESS = {"backlog": 0, "todo": 0, "in progress": 50, "done": 100, "canceled": 0, "cancelled": 0}


def _dashboard_records(doctype: str, fields: list, *, order_by: str = None, max_rows: int = 1000) -> list:
    """Reads paginated CRM records for the dashboard."""
    records = []
    for start in range(0, max_rows, 100):
        page = crm_client.get_list(doctype, fields=fields, order_by=order_by, limit=100, start=start) or []
        records.extend(page)
        if len(page) < 100:
            break
    return records


def _dashboard_datetime(value):
    """Parses a CRM timestamp string into a timezone-aware datetime."""
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
    """Formats a timestamp as a relative 'x ago' string."""
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
    """Formats a numeric value as an INR currency string."""
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        amount = 0
    return f"₹{amount:,.0f}"


def _dashboard_number(value) -> float:
    """Coerces a value to float, defaulting to 0.0."""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _dashboard_change(current: float, previous: float) -> tuple[str, str]:
    """Formats a period-over-period percent change and its direction."""
    if previous == 0:
        if current == 0:
            return "0%", "up"
        return "+100%", "up"
    percent = round((current - previous) * 100 / previous)
    return f"{percent:+d}%", "up" if percent >= 0 else "down"


def build_dashboard() -> dict:
    """Builds the full dashboard response from live CRM data."""
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
