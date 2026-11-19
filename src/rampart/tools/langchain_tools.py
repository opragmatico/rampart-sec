"""The tools as LangChain tools, for the agent: called in-process, with no protocol between the agent and the
functions.

Each tool takes its name, description and argument schema from its definition in `rampart.tools.core` (the
same source the MCP adapter reads), and calls the function registered under its name at call time, so
`override_tool` reaches it without changing what the model is offered. The result is the envelope as JSON,
the same text an MCP client receives.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.tools import StructuredTool

from .core import TOOL_DEFINITIONS, TOOL_FUNCTIONS, close_deps


def _tool(name: str, fn: Callable[..., Awaitable[dict]]) -> StructuredTool:
    template = StructuredTool.from_function(coroutine=fn, name=name)

    async def call(**kwargs: Any) -> str:
        return json.dumps(await TOOL_FUNCTIONS[name](**kwargs))

    return StructuredTool(name=name, description=template.description, args_schema=template.args_schema, coroutine=call)


def langchain_tools() -> list[StructuredTool]:
    """The seven tools, in their registry order."""
    return [_tool(name, fn) for name, fn in TOOL_DEFINITIONS.items()]


@asynccontextmanager
async def open_tools() -> AsyncIterator[list[StructuredTool]]:
    """The tools for one owner (the API lifespan, a CLI command, a test). On exit it closes the clients the
    functions share (HTTP, the corpus store), which Qdrant local mode needs before another owner opens."""
    try:
        yield langchain_tools()
    finally:
        await close_deps()
