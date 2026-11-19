"""The agent with its real tools: how the CLI, the API and the live evaluations run it.

    async with open_triage() as rp:
        result = await rp.run("Triage CVE-2026-85706 for our environment.")

The tools are the in-process functions, opened for this owner; the runner gets them by injection. In Qdrant
local mode the output guard reads the corpus through the store the tools already hold, because local mode
allows one client per path; with a Qdrant server the runner opens its own.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from .agent.runner import Rampart
from .settings import Settings, get_settings
from .tools.core import deps
from .tools.langchain_tools import open_tools


@asynccontextmanager
async def open_triage(settings: Settings | None = None) -> AsyncIterator[Rampart]:
    settings = settings or get_settings()
    async with open_tools() as tools:
        store = None if settings.qdrant_url else deps().store
        async with Rampart(settings, tools, store=store) as rp:
            yield rp
