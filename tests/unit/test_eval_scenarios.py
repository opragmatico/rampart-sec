"""The scenario harness and its DeepEval layer without a model API: a scenario run becomes a test case, the pass
condition and Tool Correctness score it deterministically, and the judged metrics are added only with a judge."""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, GEval, ToolCorrectnessMetric

from rampart.agent.runner import Rampart, RunResult
from rampart.evals.scenarios import (
    BY_KEY,
    ScenarioChecks,
    ScenarioContext,
    as_test_case,
    execute,
    judge_model,
    metrics_for,
    run_scenario,
)
from rampart.guardrails.nodes import Guards
from tests.stubs import ScriptedChatModel, StubClassifier, StubJudge, TriageTools, ai, call, final

BENIGN = BY_KEY["benign_triage"]
ANSWER = "GitLab 19.2.4 on the CI/CD platform appears affected; CVSS 10.0 and KEV-listed [Source: get_cve_details]."


@pytest.fixture
def isolated(settings, tmp_path):
    return settings.model_copy(update={"var_dir": tmp_path})


def _runner(settings, model) -> Rampart:
    guards = Guards(settings, store=None, classifier=StubClassifier(), judge=StubJudge())
    return Rampart(settings, TriageTools().all(), guards=guards, model=model)


def _triage_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        replies=[
            ai(
                call("get_cve_details", id="c0", cve_id="CVE-2026-85706"),
                call("get_exploit_intel", id="c1", cve_id="CVE-2026-85706"),
                call("query_private_database", id="c2", query="Does this system run GitLab?", system="cicd-platform"),
            ),
            final(ANSWER),
        ]
    )


async def test_a_scripted_benign_triage_passes_its_deterministic_metrics(isolated):
    async with _runner(isolated, _triage_model()) as rp:
        verdict = await run_scenario(rp, BENIGN)
    assert verdict.passed and verdict.route == "agent"
    assert set(verdict.metrics) == {"Scenario checks", "Tool Correctness"}
    assert all(m["success"] for m in verdict.metrics.values())
    assert verdict.checks and all(verdict.checks.values())


async def test_the_run_becomes_a_deepeval_test_case_and_assert_test_accepts_it(isolated):
    async with _runner(isolated, _triage_model()) as rp:
        ctx = await execute(rp, BENIGN)
    case = as_test_case(BENIGN, ctx)
    assert case.input == BENIGN.query and case.actual_output == ctx.final.answer
    assert [t.name for t in case.tools_called] == ["get_cve_details", "get_exploit_intel", "query_private_database"]
    assert case.tools_called[2].input_parameters == {"query": "Does this system run GitLab?", "system": "cicd-platform"}
    assert [t.name for t in case.expected_tools] == list(BENIGN.expected_tools)
    assert case.retrieval_context is None  # no corpus store in this run
    assert case.metadata["status"] == "delivered" and case.name == "1-benign_triage"
    assert_test(case, metrics_for(BENIGN, ctx))  # DeepEval's own runner, deterministic metrics only


async def test_a_missing_evidence_tool_fails_both_metrics_with_reasons(isolated):
    model = ScriptedChatModel(replies=[ai(call("get_cve_details", cve_id="CVE-2026-85706")), final(ANSWER)])
    async with _runner(isolated, model) as rp:
        verdict = await run_scenario(rp, BENIGN)
    assert not verdict.passed
    assert verdict.checks["evidence_tools_called"] is False
    assert "evidence_tools_called" in verdict.metrics["Scenario checks"]["reason"]
    assert verdict.metrics["Tool Correctness"]["success"] is False


async def test_a_blocked_injection_passes_without_a_tool_metric(isolated):
    async with _runner(isolated, ScriptedChatModel()) as rp:
        verdict = await run_scenario(rp, BY_KEY["direct_injection"])
    assert verdict.passed and verdict.route == "blocked" and list(verdict.metrics) == ["Scenario checks"]


def _ctx(settings, retrieval_context=()) -> ScenarioContext:
    r = RunResult("run-x", "t", "q", "answer", {}, [], [], [], {}, 1)
    return ScenarioContext(settings=settings, first=r, final=r, tickets=[], retrieval_context=list(retrieval_context))


def test_judged_metrics_join_only_with_a_judge_model(settings):
    plain = metrics_for(BENIGN, _ctx(settings, ["chunk text"]))
    assert [type(m) for m in plain] == [ScenarioChecks, ToolCorrectnessMetric]
    judged = metrics_for(BENIGN, _ctx(settings, ["chunk text"]), judge_model(settings))
    assert [type(m) for m in judged] == [ScenarioChecks, ToolCorrectnessMetric, GEval, FaithfulnessMetric]
    # Faithfulness needs the retrieved texts; without them only GEval is added
    no_context = metrics_for(BENIGN, _ctx(settings), judge_model(settings))
    assert [type(m) for m in no_context] == [ScenarioChecks, ToolCorrectnessMetric, GEval]
    # a scenario with no written criteria and no expected tools stays deterministic even with a judge
    assert [type(m) for m in metrics_for(BY_KEY["direct_injection"], _ctx(settings), judge_model(settings))] == [
        ScenarioChecks
    ]


def test_the_judge_uses_ramparts_judge_model(settings):
    assert judge_model(settings).name == settings.judge_model
