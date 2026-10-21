import asyncio
from datetime import date, timedelta

import pytest

from rampart.tickets.store import TicketStore


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
