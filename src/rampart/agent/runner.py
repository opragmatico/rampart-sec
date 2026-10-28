"""Run the guarded agent: one entry point for the API, the CLI, the evaluations and the tests.

A run answers one question and belongs to a thread (the conversation). Runs are checkpointed, so a run
paused for ticket approval can be resumed later, and every run is written to the audit store with its
guardrail log, tools, tokens, cost estimate and latency per node.

The runner takes its tools as LangChain tools and does not own their connection: the caller opens the MCP
session (in-process or over HTTP) and passes `tools` (and, in Qdrant local mode, the corpus store the tools
already hold, which the output guard reads). That keeps one owner per connection, and lets a test pass stubs.

Per thread, the runner keeps three rules:

- runs on one thread never overlap; a second request waits for the first;
- a thread paused for approval takes no new question (`ApprovalPending`), and a thread that is not paused
  cannot be resumed (`NothingToResume`);
- each run starts with fresh evidence (guardrail log, tools that answered, retrieved chunks), so the output
  guard judges a follow-up question only on what ran for it, while the conversation itself carries over.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import aiosqlite
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, Overwrite

from ..audit import AuditStore
from ..guardrails.nodes import Guards, parse_envelope, tool_text
from ..logging import bind, get_logger, unbind
from ..rag.store import CorpusStore
from ..settings import Settings, get_settings
from .graph import RECURSION_LIMIT, build_triage_graph
from .nodes import TICKET_TOOL, error_result, load_prompt
from .state import FINAL_ANSWER_TOOL

log = get_logger(__name__)

# USD per 1M tokens (input, output), for the estimate in the audit log; the longest matching prefix wins
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00)}


class ApprovalPending(RuntimeError):
    """A new question was sent to a thread that is waiting for an approval decision."""


class NothingToResume(LookupError):
    """`resume` was called on a thread that is not paused for approval."""


@dataclass
class RunResult:
    """One call to `run` or `resume`. The audit row covers the whole run, across a pause; `latency_ms`,
    `usage` and `cost_usd` here cover this call only."""

    run_id: str
    thread_id: str
    query: str
    answer: str
    state: dict[str, Any]
    guardrail_log: list[dict]
    tools_called: list[str]
    tool_calls: list[dict]
    node_latency_ms: dict[str, int]
    latency_ms: int
    interrupted: bool = False
    interrupt: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def route(self) -> str:
        return self.guardrail_log[0].get("route", "?") if self.guardrail_log else "?"

    @property
    def output_log(self) -> dict | None:
        return next((e for e in reversed(self.guardrail_log) if e.get("stage") == "output_guard"), None)

    @property
    def withheld(self) -> bool:
        return bool(self.state.get("withheld"))

    @property
    def status(self) -> str:
        """One word for the outcome: delivered | blocked | withheld | paused | failed."""
        if self.error and not self.state.get("delivered"):
            return "failed"
        if self.interrupted and not self.state.get("delivered"):
            return "paused"
        if self.route == "blocked":
            return "blocked"
        if self.withheld:
            return "withheld"
        return "delivered"

    def path(self) -> list[str]:
        """The graph nodes this call ran, in first-run order."""
        return list(self.node_latency_ms)


def close_unanswered_calls(messages: Sequence[BaseMessage]) -> list[ToolMessage]:
    """Results for tool calls in the history that never got one, which happens only when a run stopped
    between the model's turn and its tool step (a crash, the recursion limit). Without them the model API
    rejects the thread's next request."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    return [
        error_result(call, "runtime", "NOT_RUN", "This call never ran: the run that made it ended first.")
        for m in messages
        for call in getattr(m, "tool_calls", None) or []
        if call["id"] not in answered
    ]


def _cost(model: str, usage: dict) -> float:
    prefix = max((p for p in PRICES if model.startswith(p)), key=len, default=None)
    if prefix is None:
        return 0.0
    price_in, price_out = PRICES[prefix]
    return (usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out) / 1e6


class _ThreadLocks:
    """One lock per thread id, dropped when nobody holds or waits for it."""

    def __init__(self) -> None:
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def hold(self, thread_id: str) -> AsyncIterator[None]:
        lock, users = self._locks.get(thread_id, (asyncio.Lock(), 0))
        self._locks[thread_id] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            lock, users = self._locks[thread_id]
            if users == 1:
                del self._locks[thread_id]
            else:
                self._locks[thread_id] = (lock, users - 1)


def agent_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.agent_model,
        temperature=0,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )


class Rampart:
    """Owns the graph, the checkpointer and the audit store. `async with Rampart(settings, tools) as rp:`."""

    def __init__(
        self,
        settings: Settings | None = None,
        tools: Sequence[BaseTool] = (),
        *,
        store: CorpusStore | None = None,
        guards: Guards | None = None,
        model: BaseChatModel | None = None,
        checkpointer=None,
        company: str | None = None,
    ):
        self.settings = settings or get_settings()
        self.tools = list(tools)
        self.company = company or self.settings.company
        self.audit = AuditStore(self.settings.audit_db)
        self.store, self.guards, self.model, self.checkpointer = store, guards, model, checkpointer
        self.graph = None
        self.prompt_version = "unknown"
        self._stack: AsyncExitStack | None = None
        self._threads = _ThreadLocks()

    async def open(self) -> Rampart:
        if self.graph is not None:
            return self
        self._stack = AsyncExitStack()
        if self.store is None and self.guards is None and self.settings.qdrant_url:
            self.store = CorpusStore(self.settings)  # a Qdrant server takes any number of clients
            self._stack.callback(self.store.close)
        if self.guards is None:
            self.guards = Guards(self.settings, store=self.store)
        if self.guards.store is None:
            log.warning("runner.no_corpus_store", detail="the output guard cannot check verbatim overlap")
        self.model = self.model or agent_model(self.settings)
        if self.checkpointer is None:
            conn = await self._stack.enter_async_context(aiosqlite.connect(str(self.settings.checkpoints_db)))
            self.checkpointer = AsyncSqliteSaver(conn)
            await self.checkpointer.setup()
        prompt, self.prompt_version = load_prompt()
        self.graph = build_triage_graph(
            self.guards, self.tools, self.model, prompt=prompt, company=self.company, checkpointer=self.checkpointer
        )
        return self

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self.graph = None

    async def __aenter__(self) -> Rampart:
        return await self.open()

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ── running ─────────────────────────────────────────────────────────────
    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    async def pending_approval(self, thread_id: str) -> dict[str, Any] | None:
        """The approval request a thread is paused on, or None when nothing waits for a decision."""
        snapshot = await self.graph.aget_state(self._config(thread_id))
        return next((i.value for i in snapshot.interrupts), None)

    async def run(self, query: str, thread_id: str | None = None) -> RunResult:
        thread_id = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
        async with self._threads.hold(thread_id):
            snapshot = await self.graph.aget_state(self._config(thread_id))
            if snapshot.interrupts:
                raise ApprovalPending(f"Thread {thread_id} is waiting for an approval decision; resume it first.")
            run_id = f"run-{uuid.uuid4().hex[:10]}"
            closing = close_unanswered_calls(snapshot.values.get("messages", []))
            inputs = {
                "messages": [*closing, HumanMessage(content=query, id=f"{run_id}:query")],
                "user_query": query,
                "run_id": run_id,
                # this run's evidence starts empty; the conversation in `messages` carries over
                "guardrail_log": Overwrite([]),
                "tools_called": Overwrite([]),
                "retrieved_chunks": Overwrite([]),
                "tool_calls_made": 0,
                "agent_turns": 0,
                "forced_final": False,
                "approvals": [],
                "draft": "",
                "final": None,
                "final_call_id": None,
                "delivered": "",
                "withheld": False,
            }
            return await self._execute(inputs, thread_id, run_id, query)

    async def resume(self, thread_id: str, decision: dict[str, Any] | bool) -> RunResult:
        """Continue a paused run with the operator's decision on the tickets it proposed."""
        async with self._threads.hold(thread_id):
            snapshot = await self.graph.aget_state(self._config(thread_id))
            if not snapshot.interrupts:
                raise NothingToResume(f"Thread {thread_id} is not paused for approval.")
            if isinstance(decision, bool):
                decision = {"approve_all": decision}
            values = snapshot.values
            return await self._execute(
                Command(resume=decision),
                thread_id,
                values.get("run_id") or "run-unknown",
                values.get("user_query", ""),
                resumed=True,
            )

    async def _execute(self, inputs: Any, thread_id: str, run_id: str, query: str, resumed: bool = False) -> RunResult:
        usage_cb = UsageMetadataCallbackHandler()
        config = self._config(thread_id) | {"recursion_limit": RECURSION_LIMIT, "callbacks": [usage_cb]}
        bind(run_id=run_id, thread_id=thread_id)
        try:
            log.info("run.start", query_preview=query[:80], resumed=resumed)
            node_latency: dict[str, int] = {}
            t_run = t_node = time.monotonic()
            interrupt_payload: dict[str, Any] | None = None
            error = None
            try:
                async for update in self.graph.astream(inputs, config=config, stream_mode="updates"):
                    now = time.monotonic()
                    for node, payload in update.items():
                        ms = int((now - t_node) * 1000)
                        if node == "__interrupt__":
                            interrupt_payload = payload[0].value if payload else {}
                            node_latency["approval(interrupt)"] = ms
                        else:
                            node_latency[node] = node_latency.get(node, 0) + ms
                            log.info("node.done", node=node, ms=ms)
                    t_node = now
            except Exception as e:  # noqa: BLE001 - a failed run is reported and audited, never raised
                error = f"{type(e).__name__}: {e}"
                log.error("run.error", error=error)
            state = dict((await self.graph.aget_state(self._config(thread_id))).values)
            result = self._result(
                state, run_id, thread_id, query, node_latency, t_run, interrupt_payload, usage_cb, error
            )
            self._audit(result, resumed)
            log.info(
                "run.end",
                status=result.status,
                route=result.route,
                ms=result.latency_ms,
                cost_usd=result.cost_usd,
                tools=result.tools_called,
            )
            return result
        finally:
            unbind("run_id", "thread_id")

    @staticmethod
    def _result(state, run_id, thread_id, query, node_latency, t_run, interrupt_payload, usage_cb, error) -> RunResult:
        messages = state.get("messages", [])
        start = next((i for i, m in enumerate(messages) if getattr(m, "id", None) == f"{run_id}:query"), -1)
        this_run = messages[start + 1 :]
        if state.get("delivered"):
            answer = state["delivered"]
        elif interrupt_payload is not None:
            n = len(interrupt_payload.get("tickets", []))
            answer = f"⏸️ Paused for operator approval: {n} ticket(s) requested. Resume with a decision."
        else:  # the exception text stays in the audit log; the analyst gets the run id to look it up
            why = error.split(":", 1)[0] if error else "no answer was produced"
            answer = (
                f"⚠️ The triage could not be completed ({why}). Nothing was delivered; the run is recorded in the "
                f"audit log as {run_id}."
            )
        usage = dict(usage_cb.usage_metadata)
        return RunResult(
            run_id=run_id,
            thread_id=thread_id,
            query=query,
            answer=answer,
            state=state,
            guardrail_log=list(state.get("guardrail_log", [])),
            tools_called=sorted(set(state.get("tools_called", []))),
            tool_calls=[
                c for m in this_run for c in getattr(m, "tool_calls", None) or [] if c["name"] != FINAL_ANSWER_TOOL
            ],
            node_latency_ms=node_latency,
            latency_ms=int((time.monotonic() - t_run) * 1000),
            interrupted=interrupt_payload is not None,
            interrupt=interrupt_payload,
            usage=usage,
            cost_usd=round(sum(_cost(model, u) for model, u in usage.items()), 6),
            error=error,
        )

    def _audit(self, r: RunResult, resumed: bool) -> None:
        """One row per run. A resumed run adds this call's tokens, cost and latency to the row of its first
        part, so the row covers the whole run."""
        messages = r.state.get("messages", [])
        start = next((i for i, m in enumerate(messages) if getattr(m, "id", None) == f"{r.run_id}:query"), -1)
        tickets = []
        for m in messages[start + 1 :]:
            if isinstance(m, ToolMessage) and m.name == TICKET_TOOL:
                env = parse_envelope(tool_text(m))
                if env and env.get("status") == "ok":
                    tickets.append((env.get("data") or {}).get("id"))
        out = r.output_log or {}
        row: dict[str, Any] = {
            "run_id": r.run_id,
            "thread_id": r.thread_id,
            "query_sha256": hashlib.sha256(r.query.encode()).hexdigest(),
            "query_preview": r.query[:120],
            "route": r.route,
            "tools_called": r.tools_called,
            "guardrail_log": r.guardrail_log,
            "prompt_version": self.prompt_version,
            "agent_model": self.settings.agent_model,
            "judge_model": self.settings.judge_model,
            "input_tokens": sum(u.get("input_tokens", 0) for u in r.usage.values()),
            "output_tokens": sum(u.get("output_tokens", 0) for u in r.usage.values()),
            "cost_usd": r.cost_usd,
            "latency_ms": r.latency_ms,
            "node_latency": dict(r.node_latency_ms),
            "verdict": out.get("judge") or ("INTERRUPTED" if r.interrupted else None),
            "withheld": int(r.withheld),
            "tickets": tickets,
            "error": r.error,
        }
        try:
            previous = self.audit.get(r.run_id) if resumed else None
            if previous:
                row["ts"] = previous["ts"]  # the run started then
                for k in ("input_tokens", "output_tokens", "latency_ms"):
                    row[k] += previous.get(k) or 0
                row["cost_usd"] = round(row["cost_usd"] + (previous.get("cost_usd") or 0), 6)
                earlier = previous.get("node_latency")
                for node, ms in (earlier if isinstance(earlier, dict) else json.loads(earlier or "{}")).items():
                    row["node_latency"][node] = row["node_latency"].get(node, 0) + int(ms)
            self.audit.record(**row)
        except Exception as e:  # noqa: BLE001 - the audit store must never take a run down with it
            log.warning("audit.write_failed", error=str(e))
