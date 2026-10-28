"""The agent's own nodes: the model turn, the tool step and the approval step.

`rampart.agent.graph` wires them between the guard nodes. Each is built by a factory that closes over its
dependencies (the chat model, the tools), so a test can build them with a scripted model and stub tools.

- **agent**: one model turn. The model sees the versioned system prompt and the conversation, and may call
  any tool, or `final_answer` with the structured answer as its arguments. Past either budget it is bound
  to `final_answer` alone, so a run cannot loop. A reply in plain text is converted into `final_answer` by
  one more, constrained call, so the output guard always reads fields.
- **tools**: runs the turn's tool calls concurrently and never raises. A failing call, an unknown tool,
  a `final_answer` batched with other calls, or `open_ticket` (which only the approval step may run) each
  come back as an error envelope the model can read.
- **approval**: the only path to `open_ticket`. When the analyst asked for action the run pauses with
  `interrupt()` until an operator decides; otherwise the proposal is declined by policy without pausing.
  The turn's other tool calls run after the decision, so a resume (which re-runs the node from its start)
  runs them once.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import ValidationError

from ..llm import aretry
from ..logging import get_logger
from ..tools.models import Envelope
from .state import FINAL_ANSWER_TOOL, FinalAnswer, GraphState

log = get_logger(__name__)

TICKET_TOOL = "open_ticket"
APPROVAL_TOOLS = frozenset({TICKET_TOOL})  # tools with a side effect: only the approval step runs them
TOOL_BUDGET = 24  # tool calls per run
TURN_BUDGET = 7  # agent turns per run; past either budget the model may only call final_answer

PROMPT_PATH = Path(__file__).parent / "prompts" / "agent_system.md"
_PROMPT_VERSION = re.compile(r"<!--\s*version:\s*(.+?)\s*-->\s*")

BUDGET_NOTE = (
    "Budget reached ({made} tool calls, {turns} turns): call final_answer now with the evidence you have, "
    "mark anything unverified as unknown, and say what is missing."
)
STRUCTURE_NOTE = "Deliver exactly this answer by calling final_answer, with the citations kept in each field."
PREMATURE_FINAL = (
    "final_answer ignored: it was called together with other tools. Wait for their results, then call "
    "final_answer alone."
)
NOT_REQUESTED = (
    "No ticket was opened: tickets are opened only when the analyst asks for them, and this was a triage "
    "question. Report the recommended ticket in the answer instead."
)
HOW_TO_RESUME = (
    "Resume the thread with {'approve_all': true|false}, or {'approved': [true, false, ...]} with one flag per "
    "ticket in order; an optional 'reason' is shown to the agent for the declined tickets."
)

# A ticket is a side effect, so the agent may propose one only when the analyst asked for action. This is
# enforced here, not only in the prompt: on a plain triage question a proposal is declined by policy.
ACTION_RE = re.compile(
    r"\b(open|create|file|raise|log|submit|issue)\b[^.?!]{0,60}\bticket|\bticket[^.?!]{0,40}\b(open|create|file|raise)"
    r"|\b(take action|act on it|go ahead and (open|create|file))\b",
    re.IGNORECASE,
)


def wants_action(query: str) -> bool:
    """Whether the analyst's question asks for a ticket to be opened."""
    return bool(ACTION_RE.search(query or ""))


# ── the prompt and the final-answer tool ──────────────────────────────────────
def load_prompt(path: Path = PROMPT_PATH) -> tuple[str, str]:
    """The system prompt template and its version, read from its `<!-- version: ... -->` header, which is
    not sent to the model."""
    text = path.read_text(encoding="utf-8")
    m = _PROMPT_VERSION.match(text)
    return (text[m.end() :], m.group(1)) if m else (text, "unknown")


def render_prompt(template: str, *, company: str, today: date) -> str:
    return template.replace("{company}", company).replace("{today}", today.isoformat())


def final_answer_tool() -> StructuredTool:
    """The structured answer, offered to the model as a tool. The graph answers the call itself (the output
    guard a final one, the tool step a premature one); it is never executed."""

    def _never_executed(**kwargs: Any) -> str:
        return "recorded"

    return StructuredTool.from_function(
        func=_never_executed,
        name=FINAL_ANSWER_TOOL,
        args_schema=FinalAnswer,
        description="Deliver the final, structured triage answer. Call exactly once, after the evidence is "
        "gathered, with every field cited.",
    )


# ── tool results the runtime writes itself ────────────────────────────────────
def error_result(call: ToolCall, source: str, code: str, message: str) -> ToolMessage:
    """An error envelope answering `call`, in the shape every tool returns."""
    envelope = Envelope.fail(source, code, message)
    return ToolMessage(
        content=json.dumps(envelope.model_dump()), tool_call_id=call["id"], name=call["name"], status="error"
    )


def _brief(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:600]


async def _invoke(tool: BaseTool, call: ToolCall, config) -> ToolMessage:
    """Run one tool call. Invoked with the call itself, a tool returns a ToolMessage answering it."""
    try:
        result = await tool.ainvoke({"type": "tool_call", **call}, config=config)
    except ValidationError as e:
        return error_result(call, "runtime", "INVALID_INPUT", _brief(e))
    except Exception as e:  # noqa: BLE001 - a failing tool is a result the model can read, never a crashed run
        log.warning("tool.failed", tool=call["name"], error=_brief(e))
        return error_result(call, "runtime", "TOOL_ERROR", _brief(e))
    if isinstance(result, ToolMessage):
        return result
    content = result if isinstance(result, (str, list)) else str(result)
    return ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])


async def _run_call(call: ToolCall, runnable: dict[str, BaseTool], config) -> ToolMessage:
    name = call["name"]
    if name == FINAL_ANSWER_TOOL:
        return error_result(call, "runtime", "PREMATURE_FINAL", PREMATURE_FINAL)
    if name in APPROVAL_TOOLS:  # unreachable through the router; refused here too, as defence in depth
        return error_result(call, "runtime", "APPROVAL_REQUIRED", f"{name} runs only through the approval step.")
    tool = runnable.get(name)
    if tool is None:
        return error_result(
            call, "runtime", "UNKNOWN_TOOL", f"There is no tool named '{name}'. Available: {sorted(runnable)}."
        )
    return await _invoke(tool, call, config)


def _tool_calls(message: BaseMessage) -> list[ToolCall]:
    return list(getattr(message, "tool_calls", None) or [])


# ── the agent turn ────────────────────────────────────────────────────────────
def _final_fields(call: ToolCall) -> dict[str, Any]:
    try:
        answer = FinalAnswer.model_validate(call["args"])
    except ValidationError as e:
        raw = json.dumps(call["args"], indent=1, ensure_ascii=False, default=str)
        return {"final": None, "draft": f"(final answer failed validation: {e})\n\n{raw}", "final_call_id": call["id"]}
    return {"final": answer.model_dump(), "draft": answer.render(), "final_call_id": call["id"]}


def make_agent_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    prompt: str,
    company: str,
    today: Callable[[], date] = date.today,
):
    final_tool = final_answer_tool()
    llm = model.bind_tools([*tools, final_tool])
    # Past a budget the model may only call final_answer: the graph cannot loop on a model that keeps asking
    # for tools, and a run always ends with an answer the output guard can check.
    llm_final_only = model.bind_tools([final_tool], tool_choice=FINAL_ANSWER_TOOL)

    async def agent(state: GraphState, config=None) -> dict:
        made, turns = state.get("tool_calls_made", 0), state.get("agent_turns", 0)
        system = SystemMessage(content=render_prompt(prompt, company=company, today=today()))
        msgs: list[BaseMessage] = [system, *state["messages"]]
        forced = made >= TOOL_BUDGET or turns >= TURN_BUDGET
        if forced:
            msgs.append(SystemMessage(content=BUDGET_NOTE.format(made=made, turns=turns)))
        bound = llm_final_only if forced else llm
        resp: AIMessage = await aretry(lambda: bound.ainvoke(msgs, config=config), what="agent")
        new: list[AIMessage] = [resp]
        calls = [c for c in _tool_calls(resp) if c["name"] != FINAL_ANSWER_TOOL]
        final_call = next((c for c in _tool_calls(resp) if c["name"] == FINAL_ANSWER_TOOL), None)
        if not calls and final_call is None and not forced:
            # A plain-text ending is converted into the structured answer with one constrained call, so the
            # output guard reads fields. The plain reply stays in the history.
            structure = [*msgs, resp, SystemMessage(content=STRUCTURE_NOTE)]
            resp2: AIMessage = await aretry(
                lambda: llm_final_only.ainvoke(structure, config=config), what="agent.structure"
            )
            new.append(resp2)
            final_call = next((c for c in _tool_calls(resp2) if c["name"] == FINAL_ANSWER_TOOL), None)
        out: dict[str, Any] = {
            "messages": new,
            "tool_calls_made": made + len(calls),
            "agent_turns": turns + 1,
            "forced_final": forced,
        }
        if final_call is not None and not calls:
            out |= _final_fields(final_call)
        elif not calls:  # no structured answer even after the constrained call: check the plain text
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            out |= {"final": None, "draft": content, "final_call_id": None}
        # a final_answer batched with other calls is answered as premature by the tool or approval step
        return out

    return agent


# ── the tool step ─────────────────────────────────────────────────────────────
def make_tools_node(tools: Sequence[BaseTool]):
    runnable = {t.name: t for t in tools if t.name not in APPROVAL_TOOLS}

    async def tools_node(state: GraphState, config=None) -> dict:
        calls = _tool_calls(state["messages"][-1])
        return {"messages": list(await asyncio.gather(*(_run_call(c, runnable, config) for c in calls)))}

    return tools_node


# ── the approval step ─────────────────────────────────────────────────────────
def decisions(decision: Any, n: int) -> tuple[list[bool], str]:
    """One flag per proposed ticket, and the operator's reason. Only a literal `True` approves: a missing
    flag, a malformed decision or any other value declines."""
    if isinstance(decision, bool):
        return [decision] * n, ""
    if isinstance(decision, dict):
        reason = str(decision.get("reason") or "").strip()
        approved = decision.get("approved")
        if isinstance(approved, list):
            flags = [a is True for a in approved[:n]]
            return flags + [False] * (n - len(flags)), reason
        if "approve_all" in decision:
            return [decision["approve_all"] is True] * n, reason
    return [False] * n, "The decision was not understood, so every ticket was declined."


def make_approval_node(tools: Sequence[BaseTool]):
    ticket_tool = next((t for t in tools if t.name == TICKET_TOOL), None)
    runnable = {t.name: t for t in tools if t.name not in APPROVAL_TOOLS}

    async def approval(state: GraphState, config=None) -> dict:
        calls = _tool_calls(state["messages"][-1])
        tickets = [c for c in calls if c["name"] in APPROVAL_TOOLS]
        by_policy = not wants_action(state.get("user_query", ""))
        if by_policy:
            flags, reason = [False] * len(tickets), ""
        else:
            # Pauses the run. On resume LangGraph re-runs this node from its start and interrupt() returns the
            # operator's decision, so nothing above this line may have a side effect.
            decision = interrupt(
                {
                    "type": "ticket_approval",
                    "run_id": state.get("run_id"),
                    "tickets": [c["args"] for c in tickets],
                    "how_to_resume": HOW_TO_RESUME,
                }
            )
            flags, reason = decisions(decision, len(tickets))

        results: dict[str, ToolMessage] = {}
        approvals: list[dict] = []
        for call, approved in zip(tickets, flags, strict=True):
            if by_policy:
                results[call["id"]] = error_result(call, "tickets", "NOT_REQUESTED", NOT_REQUESTED)
            elif not approved:
                message = f"The operator declined this ticket. {reason}".strip()
                results[call["id"]] = error_result(call, "tickets", "DECLINED", message)
            elif ticket_tool is None:
                results[call["id"]] = error_result(call, "runtime", "UNKNOWN_TOOL", f"{TICKET_TOOL} is not available.")
            else:  # one at a time: the ticket store allocates ids in sequence
                results[call["id"]] = await _invoke(ticket_tool, call, config)
            entry = {"args": call["args"], "approved": approved, "decided_by": "policy" if by_policy else "operator"}
            approvals.append(entry | ({"reason": reason} if reason and not approved else {}))

        others = [c for c in calls if c["name"] not in APPROVAL_TOOLS]
        for call, message in zip(
            others, await asyncio.gather(*(_run_call(c, runnable, config) for c in others)), strict=True
        ):
            results[call["id"]] = message

        log_entry = {
            "stage": "approval",
            "requested": len(tickets),
            "approved": sum(flags),
            "auto_declined": by_policy,
            "systems": [c["args"].get("system") for c in tickets],
        }
        return {
            "messages": [results[c["id"]] for c in calls],
            "approvals": [*(state.get("approvals") or []), *approvals],
            "guardrail_log": [log_entry],
        }

    return approval
