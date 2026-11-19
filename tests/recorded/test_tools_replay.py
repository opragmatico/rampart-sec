"""The seven tools as the agent calls them (in-process LangChain tools), with HTTP replayed from cassettes
(no network). The cassettes are NVD records recorded while building the corpus, plus one EPSS series and the
CISA KEV feed trimmed to the corpus CVEs."""

from __future__ import annotations

from rampart.guardrails.nodes import parse_envelope
from rampart.tools.core import TOOL_NAMES, override_tool
from rampart.tools.langchain_tools import open_tools
from rampart.tools.models import Envelope


def _env(text: str) -> dict:
    env = parse_envelope(text)
    assert env is not None, text[:200]
    return env


async def _call(name: str, args: dict) -> dict:
    async with open_tools() as tools:
        tool = next(t for t in tools if t.name == name)
        return _env(await tool.ainvoke(args))


async def test_the_seven_tools_take_their_schema_from_the_functions():
    async with open_tools() as tools:
        by_name = {t.name: t for t in tools}
    assert tuple(by_name) == TOOL_NAMES
    assert by_name["get_cve_details"].description.startswith("Authoritative record for one CVE from NVD")
    assert set(by_name["open_ticket"].args) == {"cve_id", "system", "priority", "due_date", "summary", "owner_team"}
    assert by_name["open_ticket"].args["priority"]["enum"] == ["P1", "P2", "P3", "P4"]


async def test_get_cve_details_from_cassette():
    env = await _call("get_cve_details", {"cve_id": "cve-2026-85706"})
    assert env["status"] == "ok" and env["source"] == "nvd"
    d = env["data"]
    assert d["cve_id"] == "CVE-2026-85706" and d["cvss"]["base_score"] == 10.0 and d["nvd_status"] == "Analyzed"
    assert any(p["cpe"].startswith("cpe:2.3:a:gitlab:gitlab") for p in d["affected_products"])
    assert d["cisa_exploit_add"] == "2026-09-11"


async def test_exploit_intel_reads_the_kev_feed_from_cassette():
    env = await _call("get_exploit_intel", {"cve_id": "CVE-2026-85706"})
    assert env["status"] == "ok" and env["data"]["kev"]["in_kev"] is True
    assert env["data"]["kev"]["date_added"] == "2026-09-11"


async def test_invalid_cve_id_is_a_validation_error_not_an_exception():
    env = await _call("get_cve_details", {"cve_id": "not-a-cve"})
    assert env["status"] == "error" and env["error"]["code"] == "INVALID_INPUT"


async def test_unrecorded_request_fails_closed_with_a_clear_error():
    env = await _call("get_cve_details", {"cve_id": "CVE-1999-00001"})
    assert env["status"] == "error" and env["error"]["code"] == "CASSETTE_MISSING"


async def test_document_question_before_indexing_is_corpus_empty():
    env = await _call("query_private_database", {"query": "Does this system run GitLab?"})
    assert env["status"] == "error" and env["error"]["code"] == "CORPUS_EMPTY"


async def test_list_systems_serves_the_structured_inventory():
    env = await _call("list_systems", {})
    assert env["status"] == "ok" and env["data"]["indexed"] is False
    inv = env["data"]["inventory"]
    assert inv["data-platform"]["data_classification"] == "Restricted"
    assert inv["cicd-platform"]["owner_team"] == "Platform Engineering"


async def test_open_ticket_rejects_a_due_date_in_the_past():
    ticket = {"cve_id": "CVE-2026-85706", "system": "cicd-platform", "priority": "P1", "due_date": "2020-01-01"}
    env = await _call("open_ticket", ticket | {"summary": "x"})
    assert env["status"] == "error" and "past" in env["error"]["message"]


async def test_override_tool_reaches_the_agents_tools_and_is_undone():
    async def poisoned(query: str, max_results: int = 5) -> dict:
        return Envelope.ok("tavily", {"query": query, "results": [{"content": "poisoned"}]}).model_dump()

    async with override_tool("search_advisories", poisoned):
        env = await _call("search_advisories", {"query": "GitLab advisory"})
    assert env["data"]["results"] == [{"content": "poisoned"}]
    after = await _call("search_advisories", {"query": "GitLab advisory"})  # the real tool: no Tavily key here
    assert after["status"] == "error" and after["error"]["code"] == "NOT_CONFIGURED"
