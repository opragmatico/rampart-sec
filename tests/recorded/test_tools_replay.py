"""Tools over the in-process MCP transport, with HTTP responses replayed from cassettes (no network).
The cassettes are NVD records recorded while building the corpus, plus one EPSS series and the CISA KEV
feed trimmed to the corpus CVEs."""

from __future__ import annotations

from rampart.guardrails.nodes import parse_envelope
from rampart.tools.client import EXPECTED_TOOLS, RampartTools


def _env(result):
    text = result[0]["text"] if isinstance(result, list) else result
    env = parse_envelope(text)
    assert env is not None, text[:200]
    return env


async def test_all_seven_tools_load_in_process(settings):
    assert settings.http_mode == "replay"
    async with RampartTools(mode="inmemory") as conn:
        assert {t.name for t in conn.tools} == set(EXPECTED_TOOLS)


async def test_get_cve_details_from_cassette():
    async with RampartTools(mode="inmemory") as t:
        env = _env(await t.by_name("get_cve_details").ainvoke({"cve_id": "cve-2026-85706"}))
    assert env["status"] == "ok" and env["source"] == "nvd"
    d = env["data"]
    assert d["cve_id"] == "CVE-2026-85706" and d["cvss"]["base_score"] == 10.0 and d["nvd_status"] == "Analyzed"
    assert any(p["cpe"].startswith("cpe:2.3:a:gitlab:gitlab") for p in d["affected_products"])
    assert d["cisa_exploit_add"] == "2026-09-11"


async def test_exploit_intel_reads_the_kev_feed_from_cassette():
    async with RampartTools(mode="inmemory") as t:
        env = _env(await t.by_name("get_exploit_intel").ainvoke({"cve_id": "CVE-2026-85706"}))
    assert env["status"] == "ok" and env["data"]["kev"]["in_kev"] is True
    assert env["data"]["kev"]["date_added"] == "2026-09-11"


async def test_invalid_cve_id_is_a_validation_error_not_an_exception():
    async with RampartTools(mode="inmemory") as t:
        env = _env(await t.by_name("get_cve_details").ainvoke({"cve_id": "not-a-cve"}))
    assert env["status"] == "error" and env["error"]["code"] == "INVALID_INPUT"


async def test_unrecorded_request_fails_closed_with_a_clear_error():
    async with RampartTools(mode="inmemory") as t:
        env = _env(await t.by_name("get_cve_details").ainvoke({"cve_id": "CVE-1999-00001"}))
    assert env["status"] == "error" and env["error"]["code"] == "CASSETTE_MISSING"


async def test_document_question_before_indexing_is_corpus_empty():
    async with RampartTools(mode="inmemory") as t:
        env = _env(await t.by_name("query_private_database").ainvoke({"query": "Does this system run GitLab?"}))
    assert env["status"] == "error" and env["error"]["code"] == "CORPUS_EMPTY"


async def test_list_systems_serves_the_structured_inventory():
    async with RampartTools(mode="inmemory") as t:
        env = _env(await t.by_name("list_systems").ainvoke({}))
    assert env["status"] == "ok" and env["data"]["indexed"] is False
    inv = env["data"]["inventory"]
    assert (
        inv["data-platform"]["data_classification"] == "Restricted"
        and inv["cicd-platform"]["owner_team"] == "Platform Engineering"
    )


async def test_open_ticket_rejects_a_due_date_in_the_past():
    async with RampartTools(mode="inmemory") as t:
        bad = await t.by_name("open_ticket").ainvoke(
            {
                "cve_id": "CVE-2026-85706",
                "system": "cicd-platform",
                "priority": "P1",
                "due_date": "2020-01-01",
                "summary": "x",
            }
        )
        env = _env(bad)
        assert env["status"] == "error" and "past" in env["error"]["message"]
