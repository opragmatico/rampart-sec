"""Remediation tickets.

The SQLite store is the system of record for idempotency: one ticket per (cve_id, system). A
backend adapter may additionally create the ticket in an external tracker (GitHub Issues); the
external reference is stored next to the local row.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

VALID_PRIORITIES = ("P1", "P2", "P3", "P4")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
  id           TEXT PRIMARY KEY,
  cve_id       TEXT NOT NULL,
  system       TEXT NOT NULL,
  priority     TEXT NOT NULL,
  due_date     TEXT NOT NULL,
  summary      TEXT NOT NULL,
  owner_team   TEXT,
  status       TEXT NOT NULL DEFAULT 'open',
  external_ref TEXT,
  created_by   TEXT,
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL,
  UNIQUE (cve_id, system)
);
"""


@dataclass
class Ticket:
    id: str
    cve_id: str
    system: str
    priority: str
    due_date: str
    summary: str
    owner_team: str | None
    status: str
    external_ref: str | None
    created_by: str | None
    created_at: str
    updated_at: str
    created_now: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class ExternalTracker(Protocol):
    async def create(self, ticket: Ticket) -> str: ...  # returns an external reference (URL or key)


class TicketStore:
    def __init__(self, path: Path, tracker: ExternalTracker | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def validate(cve_id: str, system: str, priority: str, due_date: str, summary: str) -> None:
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {VALID_PRIORITIES}, got '{priority}'.")
        try:
            d = date.fromisoformat(due_date)
        except ValueError:
            raise ValueError(f"due_date must be ISO (YYYY-MM-DD), got '{due_date}'.") from None
        if d < date.today():
            raise ValueError(f"due_date {due_date} is in the past.")
        if not summary.strip() or len(summary) > 2000:
            raise ValueError("summary must be 1-2000 characters.")
        if not system.strip():
            raise ValueError("system is required.")

    def find(self, cve_id: str, system: str) -> Ticket | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tickets WHERE cve_id=? AND system=?", (cve_id, system)).fetchone()
        return Ticket(**dict(row)) if row else None

    async def open(
        self,
        *,
        cve_id: str,
        system: str,
        priority: str,
        due_date: str,
        summary: str,
        owner_team: str | None = None,
        created_by: str | None = None,
    ) -> Ticket:
        """Idempotent: returns the existing ticket for (cve_id, system) if there is one."""
        self.validate(cve_id, system, priority, due_date, summary)
        existing = self.find(cve_id, system)
        if existing:
            return existing
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
            tid = f"RMP-{n + 1:04d}"
            c.execute(
                "INSERT INTO tickets (id, cve_id, system, priority, due_date, summary, owner_team, status, "
                "external_ref, created_by, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    tid,
                    cve_id,
                    system,
                    priority,
                    due_date,
                    summary.strip(),
                    owner_team,
                    "open",
                    None,
                    created_by,
                    now,
                    now,
                ),
            )
        ticket = self.find(cve_id, system)
        assert ticket is not None
        if self.tracker is not None:
            ref = await self.tracker.create(ticket)
            with self._conn() as c:
                c.execute("UPDATE tickets SET external_ref=?, updated_at=? WHERE id=?", (ref, now, tid))
            ticket.external_ref = ref
        ticket.created_now = True
        return ticket

    def list(self, status: str | None = None) -> list[Ticket]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM tickets" + (" WHERE status=?" if status else "") + " ORDER BY created_at",
                (status,) if status else (),
            ).fetchall()
        return [Ticket(**dict(r)) for r in rows]
