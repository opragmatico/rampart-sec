"""The MCP adapter for external agents, through a real MCP client session (the SDK's memory transport) with
HTTP replayed from cassettes: it offers the same tools the agent calls, and answers with the same envelope."""

from __future__ import annotations

import json

from mcp.shared.memory import create_connected_server_and_client_session

from rampart.tools.core import TOOL_NAMES, close_deps
from rampart.tools.langchain_tools import langchain_tools
from rampart.tools.mcp_server import create_mcp_server


async def test_external_agents_see_the_agents_tools_and_get_the_same_envelope():
    try:
        async with create_connected_server_and_client_session(create_mcp_server()) as session:
            listed = (await session.list_tools()).tools
            result = await session.call_tool("get_cve_details", {"cve_id": "CVE-2026-85706"})
    finally:
        await close_deps()

    assert tuple(t.name for t in listed) == TOOL_NAMES
    # both adapters read the same functions: identical descriptions and arguments
    for mcp_tool, lc_tool in zip(listed, langchain_tools(), strict=True):
        assert mcp_tool.description == lc_tool.description
        assert set(mcp_tool.inputSchema["properties"]) == set(lc_tool.args)
    assert result.isError is False
    env = json.loads(result.content[0].text)
    assert env["status"] == "ok" and env["data"]["cvss"]["base_score"] == 10.0
