"""Agent-facing tools that delegate to the existing Frappe CRM API.

The agent does not duplicate CRM business logic. Frappe CRM remains the source
of truth for validation, permissions, DocTypes and specialized CRM operations.
"""
import json
import re
from typing import Optional
from langchain_core.tools import tool
from .crm_client import crm_client
from web.web_tool import web_search, web_company_search, web_company_extract, web_fetch_page

# Guardrail only: it prevents the generic resource gateway from becoming a
# gateway into unrelated Frappe/system DocTypes. Fields remain dynamic.
ALLOWED_CRM_DOCTYPES = {
    "CRM Organization", "CRM Lead", "CRM Deal", "Contact", "CRM Task",
    "FCRM Note", "CRM Call Log", "CRM Notification", "CRM Contacts",
    # Read-only reference/picklist doctypes -- safe to expose for lookup so
    # the agent can check valid Link field values (e.g. Lead/Deal status)
    # before create/update instead of guessing.
    "CRM Lead Status", "CRM Deal Status", "CRM Lead Source",
    "CRM Industry", "CRM Territory", "CRM Lost Reason",
}
ALIASES = {
    "company": "CRM Organization", "companies": "CRM Organization", "organization": "CRM Organization", "organizations": "CRM Organization",
    "lead": "CRM Lead", "leads": "CRM Lead", "deal": "CRM Deal", "deals": "CRM Deal",
    "contact": "Contact", "contacts": "Contact", "task": "CRM Task", "tasks": "CRM Task",
    "note": "FCRM Note", "notes": "FCRM Note", "call": "CRM Call Log", "calls": "CRM Call Log",
    "lead status": "CRM Lead Status", "lead statuses": "CRM Lead Status",
    "deal status": "CRM Deal Status", "deal statuses": "CRM Deal Status",
}

# Fixed workflow enumerations -- their option sets are curated by whoever
# configured the CRM (e.g. sales stages) and should never be silently
# extended by the agent. A bad value here is always a "pick from
# valid_options" situation, never a "create it for me" situation.
ENUM_LIKE_DOCTYPES = {"CRM Lead Status", "CRM Deal Status"}
# Everything else in ALLOWED_CRM_DOCTYPES that a Link field can point to
# (CRM Organization, CRM Industry, CRM Territory, CRM Lead Source,
# CRM Lost Reason) is open master data -- fine to create on the fly once
# the user has explicitly confirmed they want that new record to exist.

def resolve_doctype(value: str) -> str:
    value = (value or "").strip()
    dt = ALIASES.get(value.casefold(), value)
    if dt not in ALLOWED_CRM_DOCTYPES:
        raise ValueError(f"Unsupported CRM DocType: {value}. Allowed: {', '.join(sorted(ALLOWED_CRM_DOCTYPES))}")
    return dt

def out(value):
    return json.dumps(value, ensure_ascii=False, default=str)

def _normalize_activity_reference(dt: str, data: dict):
    """Normalize the reference fields shared by CRM timeline records.

    Frappe CRM uses ``reference_docname`` (not the generic Frappe-looking
    ``reference_name``) and its CRM doctypes are prefixed (``CRM Deal``, not
    ``Deal``).  Models and API clients commonly use the shorter spellings, so
    accept those at the boundary and turn them into the exact live values.
    """
    if dt not in {"FCRM Note", "CRM Call Log", "CRM Task"}:
        return data, None

    normalized = dict(data or {})
    legacy_name = normalized.pop("reference_name", None)
    docname = normalized.get("reference_docname")
    if legacy_name not in (None, ""):
        if docname not in (None, "") and str(docname) != str(legacy_name):
            return normalized, {
                "error": "invalid_activity_reference",
                "doctype": dt,
                "message": "reference_name and reference_docname disagree. Supply only reference_docname.",
            }
        normalized["reference_docname"] = legacy_name

    reference_dt = normalized.get("reference_doctype")
    if reference_dt not in (None, ""):
        try:
            normalized["reference_doctype"] = resolve_doctype(str(reference_dt))
        except ValueError:
            return normalized, {
                "error": "invalid_activity_reference",
                "doctype": dt,
                "reference_doctype": reference_dt,
                "message": "Use a CRM document type such as CRM Deal, CRM Lead, Contact, CRM Organization, or CRM Task.",
            }

    has_dt = normalized.get("reference_doctype") not in (None, "")
    has_name = normalized.get("reference_docname") not in (None, "")
    if has_dt != has_name:
        return normalized, {
            "error": "invalid_activity_reference",
            "doctype": dt,
            "message": "reference_doctype and reference_docname must be supplied together.",
        }
    if has_dt:
        try:
            crm_client.get_doc(normalized["reference_doctype"], normalized["reference_docname"])
        except Exception as exc:
            return normalized, {
                "error": "invalid_activity_reference",
                "doctype": dt,
                "reference_doctype": normalized["reference_doctype"],
                "reference_docname": normalized["reference_docname"],
                "message": f"Referenced document could not be found or read: {exc}",
            }
    return normalized, None

def _task_assignee_options(search_text: str = ""):
    """Return enabled CRM users ranked by their active CRM Task workload."""
    try:
        users = crm_client.get_list(
            "User", fields=["name", "full_name", "email", "enabled"],
            filters={"enabled": 1}, limit=100,
        ) or []
        tasks = []
        start = 0
        # Read all task pages (up to 10,000 records) so the recommendation is
        # based on workload, not merely the first REST page.
        while start < 10000:
            page = crm_client.get_list(
                "CRM Task", fields=["assigned_to", "status"], limit=100, start=start,
            ) or []
            tasks.extend(page)
            if len(page) < 100:
                break
            start += 100
    except Exception as exc:
        return {"error": "task_assignment_lookup_failed", "message": str(exc)}

    active_counts = {}
    for task in tasks:
        if task.get("assigned_to") and task.get("status") not in {"Done", "Canceled"}:
            assignee = task["assigned_to"]
            active_counts[assignee] = active_counts.get(assignee, 0) + 1

    query = (search_text or "").strip().casefold()
    options = []
    for user in users:
        name = str(user.get("name") or "")
        full_name = str(user.get("full_name") or "")
        email = str(user.get("email") or "")
        if query and query not in " ".join((name, full_name, email)).casefold():
            continue
        options.append({
            "user": name,
            "full_name": full_name or name,
            "email": email or name,
            "active_task_count": active_counts.get(name, 0),
        })
    options.sort(key=lambda option: (option["active_task_count"], option["full_name"].casefold(), option["user"].casefold()))
    return {
        "count": len(options),
        "recommended": options[0] if options else None,
        "options": options[:50],
        "definition": "Active tasks exclude Done and Canceled.",
    }

def _validate_task_assignee(data: dict, *, required: bool):
    assignee = (data or {}).get("assigned_to")
    if assignee in (None, ""):
        if not required:
            return None
        return {
            "error": "task_assignee_required",
            "message": "Choose an assignee before creating a task. Call crm_task_assignment_options to get the available users ranked by active task count, then obtain the user's confirmation.",
        }
    choices = _task_assignee_options(str(assignee))
    if choices.get("error"):
        return choices
    exact_match = next((option for option in choices["options"] if option["user"].casefold() == str(assignee).casefold()), None)
    if exact_match:
        return None
    return {
        "error": "invalid_task_assignee",
        "assigned_to": assignee,
        "available_matches": choices["options"],
        "message": f"'{assignee}' is not an enabled CRM user. Ask whether the user wants to create an account; do not create one without explicit confirmation and the person's email and first name.",
    }

def _validate_link_fields(dt: str, data: dict):
    """Pre-checks Link-type fields in `data` against live CRM field
    metadata before writing, so a bad value (e.g. status='Open' when the
    real options are New/Contacted/..., or organization='BJP' when no such
    CRM Organization record exists yet) is caught with the actual valid
    options in the same tool call, instead of a raw Frappe
    LinkValidationError after a wasted round trip.

    Only checks Link fields whose target doctype is itself one we're
    allowed to query (see ALLOWED_CRM_DOCTYPES) -- fields linking to
    doctypes outside that set (User, Salutation, Gender, ...) are left for
    Frappe to validate as before. Returns None if nothing looks wrong (or
    it couldn't be checked), else a dict describing the problem(s).
    """
    try:
        fields = crm_client.get_fields(dt)
    except Exception:
        return None  # metadata unavailable -- fall back to normal write + Frappe's own error
    field_by_name = {f.get("fieldname"): f for f in (fields or [])}
    problems = []
    for key, value in (data or {}).items():
        if value in (None, ""):
            continue
        f = field_by_name.get(key)
        if not f or f.get("fieldtype") != "Link":
            continue
        linked_dt = f.get("options")
        if not linked_dt or linked_dt not in ALLOWED_CRM_DOCTYPES:
            continue
        try:
            existing = crm_client.get_list(linked_dt, fields=["name"], limit=100)
        except Exception:
            continue
        valid_names = {str(r.get("name", "")).casefold() for r in (existing or [])}
        if str(value).casefold() not in valid_names:
            options_list = sorted({r.get("name") for r in (existing or []) if r.get("name")})
            creatable = linked_dt not in ENUM_LIKE_DOCTYPES
            if creatable:
                hint = (
                    f"'{value}' is not an existing {linked_dt} record. This is open reference "
                    f"data, not a fixed status list -- ask the user to confirm before creating "
                    f"a new {linked_dt} record named '{value}' via crm_create, then retry this "
                    f"call with the same value."
                )
            else:
                hint = (
                    f"'{value}' is not a valid {linked_dt} option. This is a fixed workflow "
                    f"status list -- do NOT create a new one. Pick the closest match from "
                    f"valid_options and retry, or ask the user to choose if none fit."
                )
            problems.append({
                "field": key,
                "value": value,
                "linked_doctype": linked_dt,
                "valid_options": options_list[:50],
                "creatable": creatable,
                "hint": hint,
            })
    if problems:
        return {"error": "invalid_link_fields", "doctype": dt, "problems": problems}
    return None

@tool
def crm_metadata(doctype: str) -> str:
    """Return live fields and filterable fields from the CRM application."""
    dt = resolve_doctype(doctype)
    return out({"doctype": dt, "fields": crm_client.get_fields(dt), "filterable_fields": crm_client.get_filterable_fields(dt)})

@tool
def crm_search(doctype: str, filters: Optional[dict] = None, search_text: Optional[str] = None,
               fields: Optional[list] = None, order_by: Optional[str] = None, limit: int = 20, start: int = 0) -> str:
    """Search CRM records using the standard Frappe resource API."""
    dt = resolve_doctype(doctype)
    filters = filters or {}
    or_filters = None
    if search_text:
        available = {f.get("fieldname") for f in crm_client.get_fields(dt)}
        candidates = [f for f in ("name", "organization_name", "lead_name", "first_name", "last_name", "email_id", "email", "phone", "mobile_no") if f in available]
        or_filters = [[f, "like", f"%{search_text}%"] for f in candidates]
    rows = crm_client.get_list(dt, fields=fields, filters=filters, or_filters=or_filters, order_by=order_by, limit=limit, start=start)
    return out({"doctype": dt, "count": len(rows or []), "records": rows or []})

@tool
def crm_get(doctype: str, name: str) -> str:
    """Read one CRM document by name."""
    return out(crm_client.get_doc(resolve_doctype(doctype), name))

@tool
def crm_create(doctype: str, data: dict) -> str:
    """Create a CRM document. The CRM application performs validation and permissions.

    Link fields (e.g. status, organization) are pre-checked against live CRM
    data before writing; if a value doesn't match an existing record, this
    returns the actual valid options instead of failing at the Frappe API.

    Notes use doctype ``FCRM Note``, calls use ``CRM Call Log``, and tasks use
    ``CRM Task``. To attach any of them to a CRM record, send ``reference_doctype`` and
    ``reference_docname`` together; short aliases such as ``deal`` and the
    legacy key ``reference_name`` are accepted and normalized. CRM Call Log
    additionally requires ``from``, ``to``, ``type`` (Incoming/Outgoing), and
    ``status``.
    """
    if not data: return "Record data is required."
    dt = resolve_doctype(doctype)
    data, reference_problem = _normalize_activity_reference(dt, data)
    if reference_problem:
        return out(reference_problem)
    if dt == "CRM Task":
        assignee_problem = _validate_task_assignee(data, required=True)
        if assignee_problem:
            return out(assignee_problem)
    problem = _validate_link_fields(dt, data)
    if problem:
        return out(problem)
    return out(crm_client.create_doc(dt, data))

@tool
def crm_update(doctype: str, name: str, data: dict) -> str:
    """Update a CRM document. The CRM application performs validation and permissions.

    Link fields (e.g. status, organization) are pre-checked against live CRM
    data before writing; if a value doesn't match an existing record, this
    returns the actual valid options instead of failing at the Frappe API.
    """
    if not data: return "Update data is required."
    dt = resolve_doctype(doctype)
    data, reference_problem = _normalize_activity_reference(dt, data)
    if reference_problem:
        return out(reference_problem)
    if dt == "CRM Task":
        assignee_problem = _validate_task_assignee(data, required="assigned_to" in data)
        if assignee_problem:
            return out(assignee_problem)
    problem = _validate_link_fields(dt, data)
    if problem:
        return out(problem)
    return out(crm_client.update_doc(dt, name, data))

@tool
def crm_delete(doctype: str, name: str) -> str:
    """Delete a CRM document subject to the CRM application's permissions."""
    return out(crm_client.delete_doc(resolve_doctype(doctype), name))

@tool
def crm_task_assignment_options(search_text: str = "") -> str:
    """List enabled CRM users for task assignment, ranked by fewest active tasks.

    Call this before creating every CRM Task. If the user did not name an
    assignee, present the recommended user and obtain confirmation. If they
    named someone, use search_text to find them. A no-match means the person
    is not an enabled CRM user; ask before beginning the new-user process.
    """
    return out(_task_assignee_options(search_text))

@tool
def crm_create_user(email: str, first_name: str, last_name: str = "", send_welcome_email: bool = True) -> str:
    """Create a basic enabled CRM user account for task assignment.

    Use only after the user explicitly confirms creating the account and
    confirms the email, name, and whether to send the welcome email. This
    intentionally does not set roles or other privileged account settings.
    """
    email = (email or "").strip()
    first_name = (first_name or "").strip()
    if not first_name or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return out({"error": "invalid_new_user_details", "message": "A first name and a valid email are required to create a CRM user."})
    result = crm_client.create_doc("User", {
        "email": email,
        "first_name": first_name,
        "last_name": (last_name or "").strip(),
        "send_welcome_email": bool(send_welcome_email),
    })
    return out({key: result.get(key) for key in ("name", "email", "full_name", "enabled") if key in result})

@tool
def crm_linked_records(doctype: str, name: str) -> str:
    """Use Frappe CRM's existing linked-document implementation."""
    dt = resolve_doctype(doctype)
    return out(crm_client.call_method("crm.api.doc.get_linked_docs_of_document", {"doctype": dt, "docname": name}))

@tool
def crm_activities(name: str) -> str:
    """Use Frappe CRM's existing activity timeline implementation for a Lead/Deal."""
    return out(crm_client.call_method("crm.api.activities.get_activities", {"name": name}))

@tool
def crm_contact_action(action: str, contact: str = "", field: str = "", value: str = "") -> str:
    """Call existing CRM contact helpers: linked deals, add contact email/phone, set primary, or search emails."""
    action = (action or "").strip().lower()
    methods = {
        "linked_deals": ("crm.api.contact.get_linked_deals", {"contact": contact}),
        "add": ("crm.api.contact.create_new", {"contact": contact, "field": field, "value": value}),
        "set_primary": ("crm.api.contact.set_as_primary", {"contact": contact, "field": field, "value": value}),
        "search_emails": ("crm.api.contact.search_emails", {"txt": value}),
    }
    if action not in methods: return "Unknown action. Use linked_deals, add, set_primary, or search_emails."
    return out(crm_client.call_method(*methods[action]))

@tool
def crm_research_company(company_name: str) -> str:
    """Research a company with the existing web tools and return CRM-ready evidence.

    This tool does not write to CRM. It finds an official website, extracts
    contact/profile information, and performs targeted searches for industry,
    headquarters and contact details. Missing data is never guessed.
    """
    name = (company_name or "").strip()
    if not name: return "Company name is required."
    candidate = web_company_search.invoke({"company_name": name})
    candidate_text = str(candidate)
    import re
    urls = re.findall(r"https?://[^\s)\]}>,]+", candidate_text)
    official = urls[0].rstrip(".,") if urls else None
    # web_company_search puts domain-verified candidates first and marks
    # anything else with a "⚠️ domain doesn't obviously match" flag right
    # after its URL -- check whether that flag applies to the URL we
    # picked, so a mismatched company (e.g. an aggregator page about an
    # unrelated same-ish-named business) isn't silently presented as fact.
    confidence = "unverified"
    if official:
        url_pos = candidate_text.find(official)
        following_text = candidate_text[url_pos:url_pos + len(official) + 150] if url_pos != -1 else ""
        confidence = "unverified" if "⚠️" in following_text else "verified"
    extraction = web_company_extract.invoke({"url": official, "company_name": name}) if official else "NOT FOUND"
    contact_page = web_fetch_page.invoke({"url": official.rstrip("/") + "/contact"}) if official else "NOT FOUND"
    context = web_search.invoke({"query": f"{name} industry headquarters contact email phone", "max_results": 5})
    instruction = "Use only evidence-backed values. Review before crm_create/crm_update."
    if confidence == "unverified" or not official:
        instruction = (
            "LOW CONFIDENCE: no candidate site's domain clearly matched the company name -- "
            "this may be the wrong company entirely (e.g. an aggregator page about an unrelated "
            "similarly-named business). Do not present this as confirmed fact. Show the raw "
            "candidates to the user and ask them to identify the right one, or ask for the "
            "company's website/city/industry to narrow the search, before using any of these details."
        )
    return out({"company_name": name, "official_website": official or "NOT FOUND", "confidence": confidence,
                "website_extraction": extraction, "contact_page": contact_page, "supporting_search": context,
                "instruction": instruction})

CRM_TOOLS = [crm_metadata, crm_search, crm_get, crm_create, crm_update, crm_delete,
             crm_task_assignment_options, crm_create_user, crm_linked_records, crm_activities,
             crm_contact_action, crm_research_company]
