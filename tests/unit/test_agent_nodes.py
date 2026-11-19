"""The agent's nodes outside a graph: the model turn with a scripted model, the tool step, the approval
step's policy decline, and the helpers they share. No model API, no MCP server."""

from __future__ import annotations

import json
from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rampart.agent.nodes import (
    BUDGET_NOTE,
    STRUCTURE_NOTE,
    TOOL_BUDGET,
    TURN_BUDGET,
    decisions,
    load_prompt,
    make_agent_node,
    make_approval_node,
    make_tools_node,
    render_prompt,
    wants_action,
)
from rampart.agent.state import FINAL_ANSWER_TOOL
from tests.stubs import CveArgs, ScriptedChatModel, TriageTools, ai, call, failing_tool, final

TODAY = date(2026, 10, 16)


def _agent(model, tools=None):
    tools = tools if tools is not None else TriageTools().all()
    return make_agent_node(
        model, tools, prompt="Analyst of {company}, today {today}.", company="Acme", today=lambda: TODAY
    )


def _state(*messages, **extra) -> dict:
    return {"messages": [HumanMessage(content="Triage CVE-2026-85706."), *messages], **extra}


def _envelope(message) -> dict:
    return json.loads(message.content)


# ── helpers ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Triage CVE-2026-72898 and open the remediation ticket(s) our policy requires.", True),
        ("Please file a ticket for the data platform.", True),
        ("Go ahead and create it.", True),
        ("Triage CVE-2026-85706: are we affected, and what does our policy require?", False),
        ("Which ticket priority does a High finding get?", False),
        ("", False),
    ],
)
def test_only_a_request_for_action_lets_a_ticket_reach_the_operator(query, expected):
    assert wants_action(query) is expected


def test_the_prompt_version_is_read_from_its_header_and_not_sent_to_the_model():
    template, version = load_prompt()
    assert version.startswith("agent-") and "<!--" not in template
    rendered = render_prompt(template, company="Larkspur Logistics", today=TODAY)
    assert "Larkspur Logistics" in rendered and "2026-10-16" in rendered and "{company}" not in rendered


@pytest.mark.parametrize(
    ("decision", "flags", "reason"),
    [
        (True, [True, True], ""),
        ({"approve_all": False, "reason": "wrong team"}, [False, False], "wrong team"),
        ({"approved": [True]}, [True, False], ""),  # a missing flag declines
        ({"approved": [True, "yes"]}, [True, False], ""),  # only a literal True approves
        ({"approve_all": "true"}, [False, False], ""),
        ("approve", [False, False], "The decision was not understood, so every ticket was declined."),
    ],
)
def test_operator_decisions_fail_closed(decision, flags, reason):
    assert decisions(decision, 2) == (flags, reason)


# ── the agent turn ───────────────────────────────────────────────────────────
async def test_a_turn_binds_every_tool_and_counts_the_calls():
    model = ScriptedChatModel(replies=[ai(call("get_cve_details", cve_id="CVE-2026-85706"))])
    out = await _agent(model)(_state())
    (seen,) = model.calls
    assert set(seen["bound"]) == {t.name for t in TriageTools().all()} | {FINAL_ANSWER_TOOL}
    assert seen["tool_choice"] is None
    assert seen["messages"][0] == SystemMessage(content="Analyst of Acme, today 2026-10-16.")
    assert out["tool_calls_made"] == 1 and out["agent_turns"] == 1 and out["forced_final"] is False
    assert "draft" not in out  # the run goes on to the tools


async def test_a_final_answer_is_validated_and_rendered():
    model = ScriptedChatModel(
        replies=[final("GitLab appears affected [Source: get_cve_details].", sources=["get_cve_details"])]
    )
    out = await _agent(model)(_state())
    assert out["final_call_id"] == "final-call" and out["final"]["sources"] == ["get_cve_details"]
    assert out["draft"].startswith("GitLab appears affected") and "Sources: get_cve_details" in out["draft"]


async def test_an_invalid_final_answer_is_reported_in_the_draft():
    model = ScriptedChatModel(replies=[ai(call(FINAL_ANSWER_TOOL, confidence=3))])
    out = await _agent(model)(_state())
    assert out["final"] is None and out["final_call_id"] == "final_answer-call"
    assert out["draft"].startswith("(final answer failed validation")


async def test_a_plain_reply_is_converted_with_one_constrained_call():
    plain = AIMessage(content="GitLab appears affected [Source: get_cve_details].")
    model = ScriptedChatModel(replies=[plain, final("GitLab appears affected [Source: get_cve_details].")])
    out = await _agent(model)(_state())
    first, second = model.calls
    assert first["tool_choice"] is None
    assert second["tool_choice"] == FINAL_ANSWER_TOOL and second["bound"] == (FINAL_ANSWER_TOOL,)
    assert second["messages"][-2] is plain and second["messages"][-1] == SystemMessage(content=STRUCTURE_NOTE)
    assert [type(m) for m in out["messages"]] == [AIMessage, AIMessage] and out["final"] is not None


async def test_a_plain_reply_stays_the_draft_when_the_model_will_not_structure_it():
    model = ScriptedChatModel(replies=[AIMessage(content="Patch soon."), AIMessage(content="Still plain.")])
    out = await _agent(model)(_state())
    assert out["draft"] == "Patch soon." and out["final"] is None and out["final_call_id"] is None


@pytest.mark.parametrize(("made", "turns"), [(TOOL_BUDGET, 1), (0, TURN_BUDGET)])
async def test_past_a_budget_the_model_may_only_call_final_answer(made, turns):
    model = ScriptedChatModel(replies=[final("What is known so far [Source: get_cve_details].")])
    out = await _agent(model)(_state(tool_calls_made=made, agent_turns=turns))
    (seen,) = model.calls
    assert seen["bound"] == (FINAL_ANSWER_TOOL,) and seen["tool_choice"] == FINAL_ANSWER_TOOL
    assert seen["messages"][-1] == SystemMessage(content=BUDGET_NOTE.format(made=made, turns=turns))
    assert out["forced_final"] is True and out["final"] is not None


async def test_a_final_answer_batched_with_tool_calls_is_not_taken():
    model = ScriptedChatModel(
        replies=[ai(call("get_cve_details", cve_id="CVE-1"), call(FINAL_ANSWER_TOOL, summary="early"))]
    )
    out = await _agent(model)(_state())
    assert "draft" not in out and out["tool_calls_made"] == 1


# ── the tool step ────────────────────────────────────────────────────────────
async def test_the_tool_step_answers_every_call_in_order_and_never_raises():
    stubs = TriageTools()
    tools = [*stubs.all(), failing_tool("search_advisories_v2", CveArgs, "isError: upstream refused")]
    turn = ai(
        call("get_cve_details", id="c0", cve_id="CVE-2026-85706"),
        call(FINAL_ANSWER_TOOL, id="c1", summary="early"),
        call("open_ticket", id="c2", cve_id="CVE-1", system="s", priority="P1", due_date="2030-01-01", summary="x"),
        call("no_such_tool", id="c3"),
        call("search_advisories_v2", id="c4", cve_id="CVE-1"),
        call("get_exploit_intel", id="c5"),  # missing its required argument
    )
    out = await make_tools_node(tools)(_state(turn))
    msgs = out["messages"]
    assert [m.tool_call_id for m in msgs] == ["c0", "c1", "c2", "c3", "c4", "c5"]
    assert _envelope(msgs[0])["status"] == "ok" and stubs.calls["get_cve_details"] == [{"cve_id": "CVE-2026-85706"}]
    codes = [_envelope(m)["error"]["code"] for m in msgs[1:]]
    assert codes == ["PREMATURE_FINAL", "APPROVAL_REQUIRED", "UNKNOWN_TOOL", "TOOL_ERROR", "INVALID_INPUT"]
    assert all(m.status == "error" for m in msgs[1:])
    assert stubs.calls.get("open_ticket", []) == []  # the tool step never opens a ticket


# ── the approval step ────────────────────────────────────────────────────────
async def test_without_a_request_for_action_a_proposed_ticket_is_declined_by_policy():
    stubs = TriageTools()
    ticket = {
        "cve_id": "CVE-2026-72898",
        "system": "data-platform",
        "priority": "P1",
        "due_date": "2030-01-01",
        "summary": "x",
    }
    turn = ai(call("open_ticket", id="t1", **ticket), call("get_exploit_intel", id="e1", cve_id="CVE-2026-72898"))
    state = _state(turn, user_query="Triage CVE-2026-72898: are we affected?", approvals=[])
    out = await make_approval_node(stubs.all())(state)  # no interrupt: no graph needed
    t1, e1 = out["messages"]
    assert _envelope(t1)["error"]["code"] == "NOT_REQUESTED" and _envelope(e1)["status"] == "ok"
    assert stubs.calls.get("open_ticket", []) == [] and len(stubs.calls["get_exploit_intel"]) == 1
    assert out["approvals"] == [{"args": ticket, "approved": False, "decided_by": "policy"}]
    assert out["guardrail_log"] == [
        {"stage": "approval", "requested": 1, "approved": 0, "auto_declined": True, "systems": ["data-platform"]}
    ]
