"""The triage workflow as a LangGraph graph: the guard nodes around the agent, its tools and the approval step.

    START -> input_guard --blocked--> refuse -> END
                  |
                  +--agent--> agent --tool calls----> tools -----> tool_guard --+
                               ^  |                                             |
                               |  +--open_ticket---> approval --> tool_guard --+
                               |  |   (pauses for an operator)                  |
                               |  +--final answer--> output_guard -> END       |
                               +------------------------------------------------+

`build_graph` is the wiring. The guard nodes are the real ones from `rampart.guardrails.nodes`; the agent,
tools and approval nodes are arguments, so a test can script them. `build_triage_graph` passes the real ones
from `rampart.agent.nodes`. `GraphState`, in `rampart.agent.state`, is the contract between all of them,
and docs/AGENT_FLOWS.md draws every path.

`open_ticket` never reaches the tool step: a turn that proposes a ticket is routed to the approval step,
which runs it only after an operator approves.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from ..guardrails.nodes import Guards
from .nodes import APPROVAL_TOOLS, make_agent_node, make_approval_node, make_tools_node
from .state import FINAL_ANSWER_TOOL, GraphState

__all__ = ["FINAL_ANSWER_TOOL", "RECURSION_LIMIT", "GraphState", "build_graph", "build_triage_graph"]

RECURSION_LIMIT = 40  # graph steps per run; the agent's budgets end a run well before this

Node = Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]]


async def _agent_not_built(state: GraphState) -> dict:
    raise NotImplementedError("the agent node is not built yet: pass agent= to build_graph")


async def _tools_not_built(state: GraphState) -> dict:
    raise NotImplementedError("the tools node is not built yet: pass tools= to build_graph")


async def _approval_not_built(state: GraphState) -> dict:
    raise NotImplementedError("the approval node is not built yet: pass approval= to build_graph")


def route_after_input(state: GraphState) -> str:
    return "refuse" if state["route"] == "blocked" else "agent"


def route_after_agent(state: GraphState) -> str:
    """No tool call, or `final_answer` alone, ends the run through the output guard. A turn that proposes a
    ticket goes to the approval step, with its other calls; any other turn goes to the tool step, which also
    answers a `final_answer` batched with other calls as premature."""
    names = {c["name"] for c in getattr(state["messages"][-1], "tool_calls", None) or []}
    if not names - {FINAL_ANSWER_TOOL}:
        return "output_guard"
    if names & APPROVAL_TOOLS:
        return "approval"
    return "tools"


def build_graph(
    guards: Guards,
    *,
    agent: Node = _agent_not_built,
    tools: Node = _tools_not_built,
    approval: Node = _approval_not_built,
    checkpointer=None,
):
    g = StateGraph(GraphState)
    g.add_node("input_guard", guards.input_guard)
    g.add_node("refuse", guards.refuse)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("approval", approval)
    g.add_node("tool_guard", guards.tool_guard)
    g.add_node("output_guard", guards.output_guard)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_input, ["refuse", "agent"])
    g.add_edge("refuse", END)
    g.add_conditional_edges("agent", route_after_agent, ["tools", "approval", "output_guard"])
    g.add_edge("tools", "tool_guard")
    g.add_edge("approval", "tool_guard")
    g.add_edge("tool_guard", "agent")
    g.add_edge("output_guard", END)
    return g.compile(checkpointer=checkpointer)


def build_triage_graph(
    guards: Guards,
    tools: Sequence[BaseTool],
    model: BaseChatModel,
    *,
    prompt: str,
    company: str,
    checkpointer=None,
    today: Callable[[], date] = date.today,
):
    """The graph with the real agent, tool and approval nodes. A checkpointer is what makes the approval
    pause resumable."""
    return build_graph(
        guards,
        agent=make_agent_node(model, tools, prompt=prompt, company=company, today=today),
        tools=make_tools_node(tools),
        approval=make_approval_node(tools),
        checkpointer=checkpointer,
    )
