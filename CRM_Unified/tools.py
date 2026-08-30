"""Agent-facing tools that delegate to the existing Frappe CRM API.

The agent does not duplicate CRM business logic. Frappe CRM remains the source
of truth for validation, permissions, DocTypes and specialized CRM operations.
"""
import json
from typing import Optional
from langchain_core.tools import tool
from .crm_client import crm_client
from web.web_tool import web_search, web_company_search, web_company_extract, web_fetch_page

# Guardrail only: it prevents the generic resource gateway from becoming a
# gateway into unrelated Frappe/system DocTypes. Fields remain dynamic.
ALLOWED_CRM_DOCTYPES = {
    "CRM Organization", "CRM Lead", "CRM Deal", "Contact", "CRM Task",
    "FCRM Note", "CRM Call Log", "CRM Notification", "CRM Contacts",
}
ALIASES = {
    "company": "CRM Organization", "companies": "CRM Organization", "organization": "CRM Organization", "organizations": "CRM Organization",
    "lead": "CRM Lead", "leads": "CRM Lead", "deal": "CRM Deal", "deals": "CRM Deal",
    "contact": "Contact", "contacts": "Contact", "task": "CRM Task", "tasks": "CRM Task",
    "note": "FCRM Note", "notes": "FCRM Note", "call": "CRM Call Log", "calls": "CRM Call Log",
}

def resolve_doctype(value: str) -> str:
    value = (value or "").strip()
    dt = ALIASES.get(value.casefold(), value)
    if dt not in ALLOWED_CRM_DOCTYPES:
        raise ValueError(f"Unsupported CRM DocType: {value}. Allowed: {', '.join(sorted(ALLOWED_CRM_DOCTYPES))}")
    return dt

def out(value):
    return json.dumps(value, ensure_ascii=False, default=str)

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
    """Create a CRM document. The CRM application performs validation and permissions."""
    if not data: return "Record data is required."
    return out(crm_client.create_doc(resolve_doctype(doctype), data))

@tool
def crm_update(doctype: str, name: str, data: dict) -> str:
    """Update a CRM document. The CRM application performs validation and permissions."""
    if not data: return "Update data is required."
    return out(crm_client.update_doc(resolve_doctype(doctype), name, data))

@tool
def crm_delete(doctype: str, name: str) -> str:
    """Delete a CRM document subject to the CRM application's permissions."""
    return out(crm_client.delete_doc(resolve_doctype(doctype), name))

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
    extraction = web_company_extract.invoke({"url": official, "company_name": name}) if official else "NOT FOUND"
    contact_page = web_fetch_page.invoke({"url": official.rstrip("/") + "/contact"}) if official else "NOT FOUND"
    context = web_search.invoke({"query": f"{name} industry headquarters contact email phone", "max_results": 5})
    return out({"company_name": name, "official_website": official or "NOT FOUND", "website_extraction": extraction,
                "contact_page": contact_page, "supporting_search": context,
                "instruction": "Use only evidence-backed values. Review before crm_create/crm_update."})

CRM_TOOLS = [crm_metadata, crm_search, crm_get, crm_create, crm_update, crm_delete,
             crm_linked_records, crm_activities, crm_contact_action, crm_research_company]
