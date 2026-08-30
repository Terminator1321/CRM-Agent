# Standalone CRM MCP

Copy `mcp_server.py` and the small client into another Python project, install `mcp requests python-dotenv`, and set `CRM_URL`, `CRM_API_KEY`, and `CRM_API_SECRET`.

The server exposes the same CRM tools used by MagmaAssistance and delegates CRM behavior to the existing Frappe CRM API.
