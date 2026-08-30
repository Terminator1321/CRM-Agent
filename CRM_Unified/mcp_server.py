"""Copy-paste standalone MCP server for the separate Frappe CRM.

Run: python -m CRM_Unified.mcp_server
Requires: mcp, requests, python-dotenv
"""
from mcp.server.fastmcp import FastMCP
from .tools import CRM_TOOLS

mcp = FastMCP("Magma CRM")
for _tool in CRM_TOOLS:
    mcp.add_tool(_tool.func, name=_tool.name, description=_tool.description)

if __name__ == "__main__":
    mcp.run(transport="stdio")
