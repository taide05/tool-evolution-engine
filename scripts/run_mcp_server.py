"""Standalone MCP server entry point — runs the memory bridge over stdio transport.

Usage:
    python scripts/run_mcp_server.py

Connect from Claude Code with:
    {
      "mcpServers": {
        "tool-evolution-memory": {
          "command": "python",
          "args": ["scripts/run_mcp_server.py"],
          "cwd": "D:/tool-evolution-engine"
        }
      }
    }
"""
import asyncio
import sys
sys.path.insert(0, "src")

from tool_evolution.utils.database import get_connection, init_db
from tool_evolution.governance.mcp_bridge import MCPBridge, mcp, set_bridge


async def main():
    conn = await get_connection()
    await init_db(conn)
    bridge = MCPBridge(conn)
    set_bridge(bridge)
    await mcp.run_stdio_async()  # mcp>=1.0 新 API（run_stdio 已移除——I#7 冒烟实测）


if __name__ == "__main__":
    asyncio.run(main())
