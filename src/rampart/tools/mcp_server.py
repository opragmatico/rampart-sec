"""The tools over MCP, for agents outside Rampart (Claude Desktop, IDEs, other teams' agents).

An outward-facing adapter: the API mounts `create_mcp_server().streamable_http_app()` at `/mcp` behind its
bearer token. Rampart's own agent does not use it; it calls the same functions in-process through
`rampart.tools.langchain_tools`. FastMCP takes each tool's schema and description from its function in
`rampart.tools.core`, as the LangChain adapter does.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .core import TOOL_DEFINITIONS

INSTRUCTIONS = """Rampart vulnerability-triage tools. Typical flow: get_cve_details -> get_exploit_intel ->
query_private_database (which systems run the affected software, exposure, controls, policy SLA) ->
assess_urgency -> open_ticket (requires operator approval). Every tool returns {"status": "ok"|"error", ...}."""


def create_mcp_server() -> FastMCP:
    """A FastMCP server with the seven tools registered. Each instance owns its own HTTP session manager."""
    server = FastMCP(
        "rampart",
        instructions=INSTRUCTIONS,
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        # The bearer-token middleware of the ASGI app is the access control; the SDK's Host-header check
        # would otherwise reject requests addressed to the container's service name.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    for name, fn in TOOL_DEFINITIONS.items():
        server.tool(name=name)(fn)
    return server
