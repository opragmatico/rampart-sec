from rampart.audit import AuditStore


def test_audit_store_records_lists_and_summarises(tmp_path):
    a = AuditStore(tmp_path / "audit.sqlite3")
    a.record(
        run_id="run-1",
        thread_id="t1",
        query_sha256="x",
        query_preview="q1",
        route="blocked",
        tools_called=[],
        guardrail_log=[{"stage": "input_guard", "rule_hit": True, "llm_hit": None, "route": "blocked"}],
        latency_ms=4,
        cost_usd=0.0,
        withheld=1,
        tickets=[],
        node_latency={"input_guard": 4},
    )
    a.record(
        run_id="run-2",
        thread_id="t2",
        query_sha256="y",
        query_preview="q2",
        route="agent",
        tools_called=["get_cve_details"],
        guardrail_log=[
            {"stage": "input_guard", "rule_hit": False, "llm_hit": False, "route": "agent"},
            {"stage": "output_guard", "judge": "SAFE", "withheld": False},
        ],
        latency_ms=1200,
        cost_usd=0.002,
        verdict="SAFE",
        withheld=0,
        tickets=["RMP-0001"],
        node_latency={"input_guard": 100, "agent": 1000, "output_guard": 100},
    )
    recent = a.recent(limit=5)
    assert [r["run_id"] for r in recent] == ["run-2", "run-1"]
    assert recent[0]["tools_called"] == ["get_cve_details"] and recent[0]["tickets"] == ["RMP-0001"]
    assert "guardrail_log" not in recent[0]
    detail = a.get("run-2")
    assert detail["guardrail_log"][1]["judge"] == "SAFE" and detail["node_latency"]["agent"] == 1000
    assert a.get("nope") is None
    s = a.summary()
    assert s["runs"] == 2 and s["blocked"] == 1 and s["withheld"] == 1
    assert s["blocked_by"] == {"signature": 1, "classifier": 0}
    assert s["node_latency_ms"] == {"input_guard": 100, "agent": 1000, "output_guard": 100}
    assert a.recent(route="blocked")[0]["run_id"] == "run-1"
