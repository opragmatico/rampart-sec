"""Audit store: one row per agent run, in SQLite, queryable from the API or the CLI.

The guardrail log inside the graph state answers "what did each guard decide". This table
answers the operational questions: how many runs were blocked and by which layer, how often the
judge fires, cost and latency per run, which tickets a run created.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  thread_id     TEXT,
  ts            TEXT NOT NULL,
  query_sha256  TEXT NOT NULL,
  query_preview TEXT,
  route         TEXT,
  tools_called  TEXT,          -- JSON list
  guardrail_log TEXT,          -- JSON list
  prompt_version TEXT,
  agent_model   TEXT,
  judge_model   TEXT,
  input_tokens  INTEGER,
  output_tokens INTEGER,
  cost_usd      REAL,
  latency_ms    INTEGER,
  node_latency  TEXT,          -- JSON {node: ms}
  verdict       TEXT,
  withheld      INTEGER,
  tickets       TEXT,          -- JSON list of ticket ids
  error         TEXT
);
CREATE INDEX IF NOT EXISTS runs_ts ON runs(ts);
"""
JSON_COLUMNS = ("tools_called", "guardrail_log", "node_latency", "tickets")


def _decode(row: dict) -> dict:
    for k in JSON_COLUMNS:
        if isinstance(row.get(k), str):
            try:
                row[k] = json.loads(row[k])
            except ValueError:
                pass
    return row


class AuditStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def record(self, **row: Any) -> None:
        row.setdefault("ts", datetime.now(UTC).isoformat(timespec="seconds"))
        for k in JSON_COLUMNS:
            if k in row and not isinstance(row[k], str):
                row[k] = json.dumps(row[k], default=str)
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({qs})", list(row.values()))

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def recent(self, limit: int = 20, route: str | None = None) -> list[dict]:
        """Most recent runs, JSON columns decoded, without the guardrail log (use `get` for one run)."""
        where = " WHERE route = ?" if route else ""
        rows = self.query(
            "SELECT run_id, thread_id, ts, query_preview, route, tools_called, prompt_version, agent_model, "
            "judge_model, input_tokens, output_tokens, cost_usd, latency_ms, node_latency, verdict, withheld, "
            f"tickets, error FROM runs{where} ORDER BY ts DESC, rowid DESC LIMIT ?",
            ((route, limit) if route else (limit,)),
        )
        return [_decode(r) for r in rows]

    def get(self, run_id: str) -> dict | None:
        rows = self.query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return _decode(rows[0]) if rows else None

    def summary(self) -> dict:
        rows = self.query("""
            SELECT COUNT(*) AS runs,
                   SUM(route = 'blocked') AS blocked,
                   SUM(withheld = 1) AS withheld,
                   SUM(verdict LIKE 'UNSAFE%') AS judge_unsafe,
                   SUM(verdict = 'INTERRUPTED') AS paused,
                   SUM(error IS NOT NULL) AS failed,
                   ROUND(AVG(latency_ms)) AS avg_latency_ms,
                   ROUND(SUM(cost_usd), 4) AS cost_usd,
                   SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens
            FROM runs""")
        out = rows[0] if rows else {}
        layers = self.query(
            "SELECT SUM(json_extract(guardrail_log, '$[0].rule_hit') = 1) AS by_signature, "
            "SUM(json_extract(guardrail_log, '$[0].llm_hit') = 1) AS by_classifier "
            "FROM runs WHERE route = 'blocked'"
        )
        out["blocked_by"] = {
            "signature": (layers[0]["by_signature"] or 0) if layers else 0,
            "classifier": (layers[0]["by_classifier"] or 0) if layers else 0,
        }
        out["node_latency_ms"] = self.node_latency()
        return out

    def node_latency(self) -> dict[str, int]:
        """Mean latency per graph node over the runs that reached the agent."""
        agg: dict[str, list[int]] = {}
        for row in self.query("SELECT node_latency FROM runs WHERE route = 'agent' AND node_latency IS NOT NULL"):
            try:
                for node, ms in json.loads(row["node_latency"]).items():
                    agg.setdefault(node, []).append(int(ms))
            except (ValueError, AttributeError):
                continue
        return {k: round(sum(v) / len(v)) for k, v in agg.items() if v}
