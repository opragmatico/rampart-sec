"""Connect to the Rampart tools as LangChain tools.

- `mode="inmemory"`: the MCP client session talks to the server object inside this process through
  the SDK's memory transport. No port, no subprocess. Still real MCP (schemas, framing, isError).
- `mode="http"`: a running service, authenticated with a bearer token.
"""

from __future__ import annotations

from contextlib import AsyncExitStack

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from ..settings import get_settings
from .server import TOOL_NAMES

EXPECTED_TOOLS = frozenset(TOOL_NAMES)


class RampartTools:
    """Connection to the tools. `async with RampartTools() as conn:` then `conn.tools` / `conn.by_name(...)`.
    The connection must be closed by the task that opened it (the in-memory transport is anyio-based)."""

    def __init__(
        self, mode: str = "inmemory", url: str | None = None, token: str | None = None, server: FastMCP | None = None
    ):
        self.mode = mode
        s = get_settings()
        self.url = url or s.mcp_url
        self.token = token or (s.auth_token.get_secret_value() if s.auth_token else None)
        self.server = server
        self.tools: list[BaseTool] = []
        self._stack: AsyncExitStack | None = None

    async def open(self) -> list[BaseTool]:
        self._stack = AsyncExitStack()
        if self.mode == "inmemory":
            from .server import get_mcp

            server = self.server or get_mcp()
            session = await self._stack.enter_async_context(create_connected_server_and_client_session(server))
            self.tools = await load_mcp_tools(session)
        elif self.mode == "http":
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            client = MultiServerMCPClient(
                {"rampart": {"url": self.url, "transport": "streamable_http", "headers": headers}}
            )
            self.tools = await client.get_tools()
        else:
            raise ValueError("mode must be 'inmemory' or 'http'")
        names = {t.name for t in self.tools}
        missing = EXPECTED_TOOLS - names
        if missing:
            raise RuntimeError(f"MCP server is missing tools: {sorted(missing)}; loaded {sorted(names)}")
        return self.tools

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        if self.mode == "inmemory":
            from .server import close_deps

            await close_deps()

    async def __aenter__(self) -> RampartTools:
        await self.open()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def by_name(self, name: str) -> BaseTool:
        return next(t for t in self.tools if t.name == name)
