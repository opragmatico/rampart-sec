"""Stand-ins for the model-backed parts and the MCP tools, shared by the guard-node, graph and agent tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from rampart.tools.models import Envelope

USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}


class StubClassifier:
    """Stands in for LlmClassifier: answers as told and records what it was asked."""

    model = "stub"

    def __init__(self, hit: bool = False):
        self.hit, self.calls = hit, []

    async def classify(self, text: str, config=None) -> dict:
        self.calls.append(text)
        verdict = "INJECTION" if self.hit else "SAFE"
        return {"hit": self.hit, "verdict": verdict, "reason": "stub", "error": None}


class StubJudge:
    """Stands in for LlmJudge: answers as told and records what it was asked."""

    model = "stub"

    def __init__(self, verdict: str = "SAFE"):
        self.verdict, self.calls = verdict, []

    async def judge_answer(self, text: str, config=None) -> dict:
        self.calls.append(text)
        return {"verdict": self.verdict, "reasons": ["stub"], "error": None}


# ── the agent's chat model ────────────────────────────────────────────────────
class ScriptedChatModel(BaseChatModel):
    """Stands in for the agent's chat model: replies from a script, in order, and records every call with the
    tools bound, the forced tool choice and the messages it saw. `bind_tools` returns a copy that shares the
    script and the record. A reply may be an AIMessage, an exception to raise, or a function of the messages."""

    replies: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)
    bound: tuple = ()
    tool_choice: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.model_copy(update={"bound": tuple(t.name for t in tools), "tool_choice": tool_choice})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append({"bound": self.bound, "tool_choice": self.tool_choice, "messages": list(messages)})
        if not self.replies:
            raise AssertionError(f"the script has no reply left for call {len(self.calls)}")
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(messages)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(name: str, id: str | None = None, **args) -> dict:
    return {"name": name, "args": args, "id": id or f"{name}-call"}


def ai(*calls: dict, content: str = "", usage: dict | None = USAGE) -> AIMessage:
    """A model reply making `calls`, with usage metadata the cost accounting reads."""
    meta = {"usage_metadata": usage, "response_metadata": {"model_name": "gpt-4o-mini"}} if usage else {}
    return AIMessage(content=content, tool_calls=list(calls), **meta)


def final(summary: str, id: str = "final-call", **fields) -> AIMessage:
    """A model reply calling final_answer with a structured answer."""
    return ai(call("final_answer", id=id, summary=summary, **fields))


# ── tools ─────────────────────────────────────────────────────────────────────
class CveArgs(BaseModel):
    cve_id: str


class QueryArgs(BaseModel):
    query: str
    system: str | None = None


class NoArgs(BaseModel):
    pass


class TicketArgs(BaseModel):
    cve_id: str
    system: str
    priority: str
    due_date: str
    summary: str
    owner_team: str | None = None


def envelope_tool(
    name: str, args_schema: type[BaseModel], data: dict | Callable[[dict], dict], calls: list
) -> StructuredTool:
    """A tool that records its arguments and answers with an ok envelope, as the MCP tools do."""

    async def run(**kwargs) -> str:
        calls.append(kwargs)
        payload = data(kwargs) if callable(data) else data
        return json.dumps(Envelope.ok(name, payload).model_dump())

    return StructuredTool.from_function(coroutine=run, name=name, description=f"stub {name}", args_schema=args_schema)


def failing_tool(name: str, args_schema: type[BaseModel], message: str) -> StructuredTool:
    """A tool that fails the way an MCP tool does when the server reports isError."""

    async def run(**kwargs) -> str:
        raise ToolException(message)

    return StructuredTool.from_function(coroutine=run, name=name, description=f"stub {name}", args_schema=args_schema)


class TriageTools:
    """The seven tool names with stub behaviour and a record of every call, by tool name."""

    def __init__(self) -> None:
        self.calls: dict[str, list[dict]] = {}
        self.tickets_opened = 0

    def _calls(self, name: str) -> list:
        return self.calls.setdefault(name, [])

    def _ticket(self, args: dict) -> dict:
        self.tickets_opened += 1
        return {"id": f"RMP-{self.tickets_opened:04d}", "created_now": True} | args

    def all(self) -> list[StructuredTool]:
        return [
            envelope_tool("get_cve_details", CveArgs, {"cvss": {"base_score": 10.0}}, self._calls("get_cve_details")),
            envelope_tool("get_exploit_intel", CveArgs, {"epss": {"epss": 0.97}}, self._calls("get_exploit_intel")),
            envelope_tool("search_advisories", QueryArgs, {"results": []}, self._calls("search_advisories")),
            envelope_tool("assess_urgency", CveArgs, {"urgency": "high"}, self._calls("assess_urgency")),
            envelope_tool(
                "query_private_database", QueryArgs, {"answer": "Yes."}, self._calls("query_private_database")
            ),
            envelope_tool("list_systems", NoArgs, {"systems": ["data-platform"]}, self._calls("list_systems")),
            envelope_tool("open_ticket", TicketArgs, self._ticket, self._calls("open_ticket")),
        ]
