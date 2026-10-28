"""The structured final answer: how it renders for the analyst and what the schema refuses."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rampart.agent.state import FinalAnswer, SystemFinding


def test_final_answer_renders_sections():
    fa = FinalAnswer(
        summary="Appears exposed.",
        affected_systems=[
            SystemFinding(
                system="Customer Portal", affected="yes", evidence="runs it [Source: query_private_database, SRA p.4]"
            )
        ],
        exploit_intelligence="EPSS 0.9 [Source: get_exploit_intel]",
        sources=["get_exploit_intel"],
        confidence=0.8,
    )
    text = fa.render()
    assert "## Affected systems" in text and "Customer Portal" in text and "Confidence: 80%" in text
    assert "## Exploit intelligence" in text and "## Urgency" not in text  # empty sections are left out


def test_a_bare_answer_says_what_it_does_not_state():
    assert FinalAnswer(summary="  Patch soon.  ").render() == "Patch soon.\n\n_Confidence: not stated. Sources: none._"


def test_the_schema_refuses_out_of_range_confidence_and_unknown_verdicts():
    with pytest.raises(ValidationError):
        FinalAnswer(summary="s", confidence=1.5)
    with pytest.raises(ValidationError):
        SystemFinding(system="data-platform", affected="maybe", evidence="e")
    with pytest.raises(ValidationError):
        FinalAnswer.model_validate({"affected_systems": []})  # the summary is required
