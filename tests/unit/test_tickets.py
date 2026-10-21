import asyncio
import json
from datetime import date, timedelta

import httpx
import pytest

from rampart.http import HttpClient
from rampart.tickets.github import GitHubIssues
from rampart.tickets.store import Ticket, TicketStore


def test_open_ticket_is_idempotent(tmp_path):
    store = TicketStore(tmp_path / "t.sqlite3")
    due = (date.today() + timedelta(days=7)).isoformat()
    t1 = asyncio.run(
        store.open(cve_id="CVE-2026-1", system="Customer Portal", priority="P1", due_date=due, summary="patch")
    )
    t2 = asyncio.run(
        store.open(cve_id="CVE-2026-1", system="Customer Portal", priority="P2", due_date=due, summary="other")
    )
    assert t1.id == t2.id == "RMP-0001" and t1.created_now and not t2.created_now
    assert len(store.list()) == 1


def test_validation_rejects_bad_input(tmp_path):
    store = TicketStore(tmp_path / "t.sqlite3")
    with pytest.raises(ValueError):
        store.validate("CVE-2026-1", "sys", "P9", "2030-01-01", "x")
    with pytest.raises(ValueError):
        store.validate("CVE-2026-1", "sys", "P1", "2020-01-01", "x")
    with pytest.raises(ValueError):
        store.validate("CVE-2026-1", "sys", "P1", "not-a-date", "x")


async def test_github_issue_payload_is_rendered_from_the_ticket():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(201, json={"html_url": "https://github.com/acme/sec/issues/7"})

    http = HttpClient(mode="live", transport=httpx.MockTransport(handler))
    ticket = Ticket(
        id="RMP-0007",
        cve_id="CVE-2026-1",
        system="customer-portal",
        priority="P1",
        due_date="2030-01-01",
        summary="Patch NetScaler.\nSecond line.",  # a multi-line summary must survive the template intact
        owner_team=None,
        status="open",
        external_ref=None,
        created_by="rampart-agent",
        created_at="t",
        updated_at="t",
    )
    ref = await GitHubIssues(http, "acme/sec", "tok").create(ticket)
    await http.aclose()
    assert ref == "https://github.com/acme/sec/issues/7"
    payload = captured["json"]
    assert payload["title"] == "[P1] CVE-2026-1 on customer-portal (due 2030-01-01)"
    assert payload["body"] == "\n".join(
        [
            "**CVE:** CVE-2026-1",
            "**System:** customer-portal",
            "**Priority:** P1",
            "**Due:** 2030-01-01",
            "**Owner team:** n/a",
            "",
            "Patch NetScaler.",
            "Second line.",
            "",
            "_Opened by Rampart (RMP-0007); approved by an operator._",
        ]
    )
    assert payload["labels"] == ["security", "vulnerability", "p1"]
