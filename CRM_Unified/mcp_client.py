"""Standalone copy-paste MCP client for the Frappe CRM MCP server.

Set CRM_MCP_SERVER to the path of mcp_server.py, then run this file to
discover tools. The `call_tool` helper can be imported by another app.
"""
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_tool(tool_name: str, arguments: dict | None = None):
    server = os.getenv("CRM_MCP_SERVER", "CRM_Unified/mcp_server.py")
    params = StdioServerParameters(
        command=os.getenv("PYTHON", "python"),
        args=[server],
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments or {})

async def list_tools():
    server = os.getenv("CRM_MCP_SERVER", "CRM_Unified/mcp_server.py")
    params = StdioServerParameters(command=os.getenv("PYTHON", "python"), args=[server], env=os.environ.copy())
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools

async def main():
    for tool in await list_tools():
        print(f"{tool.name}: {tool.description}")

if __name__ == "__main__":
    asyncio.run(main())
