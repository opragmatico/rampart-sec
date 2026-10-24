"""The triage workflow as a LangGraph graph: a stub that wires the guard nodes around the agent.

    START -> input_guard --blocked--> refuse -> END
                  |
                  +--------agent--> agent --tool calls--> tools -> tool_guard --+
                                      ^  |                                       |
                                      |  +--final answer--> output_guard -> END  |
                                      +------------------------------------------+

The guard nodes are the real ones from `rampart.guardrails.nodes`. `agent` and `tools` are
placeholders until the agent lands (model, system prompt, MCP tools, the approval interrupt before
`open_ticket`, budgets, checkpoints); `build_graph` takes them as arguments so a test can script them.

`GraphState` is the contract between the nodes. The agent node finishes a run by calling the
`final_answer` tool and writing `draft`, `final` and `final_call_id`; the output guard answers that
tool call and delivers the checked text.
"""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ..guardrails.nodes import Guards

FINAL_ANSWER_TOOL = "final_answer"

Node = Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]]


class GraphState(TypedDict, total=False):
    # set by the caller
    user_query: str
    # add_messages appends, and replaces a message whose id is already present (how the tool guard
    # swaps in a redacted ToolMessage)
    messages: Annotated[list[BaseMessage], add_messages]
    # written by input_guard, read by the router after it
    route: Literal["agent", "blocked"]
    # appended by every guard node and by every tool step, so they cover the whole run
    guardrail_log: Annotated[list[dict], operator.add]
    tools_called: Annotated[list[str], operator.add]
    retrieved_chunks: Annotated[list[dict], operator.add]
    # written by the agent when it calls final_answer, read by output_guard
    draft: str
    final: dict | None
    final_call_id: str | None
    # written by output_guard or refuse
    withheld: bool
    delivered: str


async def _agent_not_built(state: GraphState) -> dict:
    raise NotImplementedError("the agent node is not built yet: pass agent= to build_graph")


async def _tools_not_built(state: GraphState) -> dict:
    raise NotImplementedError("the tools node is not built yet: pass tools= to build_graph")


def route_after_input(state: GraphState) -> str:
    return "refuse" if state["route"] == "blocked" else "agent"


def route_after_agent(state: GraphState) -> str:
    """Tool calls go to the tools; a final_answer call, or a reply without tool calls, ends the run."""
    calls = getattr(state["messages"][-1], "tool_calls", None) or []
    if calls and all(c["name"] != FINAL_ANSWER_TOOL for c in calls):
        return "tools"
    return "output_guard"


def build_graph(guards: Guards, *, agent: Node = _agent_not_built, tools: Node = _tools_not_built):
    g = StateGraph(GraphState)
    g.add_node("input_guard", guards.input_guard)
    g.add_node("refuse", guards.refuse)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("tool_guard", guards.tool_guard)
    g.add_node("output_guard", guards.output_guard)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_input, ["refuse", "agent"])
    g.add_edge("refuse", END)
    g.add_conditional_edges("agent", route_after_agent, ["tools", "output_guard"])
    g.add_edge("tools", "tool_guard")
    g.add_edge("tool_guard", "agent")
    g.add_edge("output_guard", END)
    return g.compile()
