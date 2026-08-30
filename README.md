# MagmaAssistance — Frappe CRM Agent

Standalone AI agent for a separate Frappe CRM instance. The agent exposes CRM
tools and the existing web tools; it does not depend on the legacy ERP tool
stack.

## CRM tools
- `crm_metadata` — live CRM fields/filterable fields
- `crm_search` — dynamic record search
- `crm_get` — read a record
- `crm_create` — create a record
- `crm_update` — update a record
- `crm_delete` — delete a record
- `crm_linked_records` — existing CRM relationship API
- `crm_activities` — existing CRM activity API
- `crm_contact_action` — existing CRM contact helpers
- `crm_research_company` — web-based company research; does not write CRM

## Frappe authentication

The CRM is a Frappe application. API credentials are credentials of a Frappe
User, not a separate CRM API-key service.

Preferred:
- `CRM_URL`
- `CRM_API_KEY`
- `CRM_API_SECRET`

If API-key generation is not available in the CRM UI, the client also supports
Frappe's normal session login:
- `CRM_USERNAME`
- `CRM_PASSWORD`

Use a dedicated Frappe user with only the permissions the agent needs.

## MCP
`CRM_Unified/mcp_server.py` is a standalone stdio MCP server and
`CRM_Unified/mcp_client.py` is a standalone discovery client. They can be copied
into another Python project.

## Run
Install the dependencies from `requirements.txt`, configure `.env`, then use
the project's normal entrypoint.
