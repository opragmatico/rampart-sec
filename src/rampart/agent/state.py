"""The contract between the graph's nodes, and the structured final answer.

`GraphState` declares every key a node reads or writes. `messages` spans the whole conversation on a
thread. The other lists accumulate within one run, and the runner resets them at the start of each run, so
a thread's next question is judged on its own evidence. Scalar keys are overwritten by the node that sets
them.

`FinalAnswer` is the schema of the `final_answer` tool. The model fills it field by field, the output guard
checks the rendered text, and the analyst reads it rendered.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

FINAL_ANSWER_TOOL = "final_answer"


class GraphState(TypedDict, total=False):
    # set by the runner at the start of a run
    user_query: str
    run_id: str
    # add_messages appends, and replaces a message whose id is already present (how the tool guard
    # swaps in a redacted ToolMessage)
    messages: Annotated[list[BaseMessage], add_messages]
    # written by input_guard, read by the router after it
    route: Literal["agent", "blocked"]
    # appended by every guard node and by every tool step within a run
    guardrail_log: Annotated[list[dict], operator.add]
    tools_called: Annotated[list[str], operator.add]
    retrieved_chunks: Annotated[list[dict], operator.add]
    # written by the agent each turn: the budgets, and whether this turn was forced to answer
    tool_calls_made: int
    agent_turns: int
    forced_final: bool
    # written by the approval step: every proposed ticket with its decision
    approvals: list[dict]
    # written by the agent when it calls final_answer, read by output_guard
    draft: str
    final: dict | None
    final_call_id: str | None
    # written by output_guard or refuse
    withheld: bool
    delivered: str


class SystemFinding(BaseModel):
    system: str = Field(description="System name exactly as list_systems returns it (never 'policy')")
    affected: Literal["yes", "no", "unknown"] = Field(
        description="yes = the assessment lists the affected product at a vulnerable version; no = the assessment lists no such product or a fixed version; unknown = the documents do not say"
    )
    evidence: str = Field(
        description="One or two full sentences naming the component and version from the assessment, ending with a citation such as [Source: query_private_database, System Risk Assessment — CI/CD and Artifact Platform p.3]"
    )


class FinalAnswer(BaseModel):
    """The structured final answer. The output guard checks its fields; the user sees it rendered."""

    summary: str = Field(
        description="Two or three hedged sentences: what the CVE is, whether we appear exposed, what to do. Every factual sentence ends with a [Source: ...] citation"
    )
    affected_systems: list[SystemFinding] = Field(
        default_factory=list, description="One entry per system you checked with query_private_database"
    )
    exposure: str = Field(
        default="",
        description="Two to four sentences: for each affected system, internet-facing or internal, data classification, and the compensating controls or findings that matter for this CVE, each with a citation such as [Source: query_private_database, System Risk Assessment — Customer Portal and Booking API p.5]",
    )
    exploit_intelligence: str = Field(
        default="",
        description="Two to four sentences: CVSS score and vector source, NVD status, EPSS probability and trend, KEV status and CISA due date, each ending with [Source: get_cve_details] or [Source: get_exploit_intel]",
    )
    urgency: str = Field(
        default="",
        description="The urgency label, score and the scorer's reasoning in one to three sentences, ending with [Source: assess_urgency]",
    )
    policy_deadline: str = Field(
        default="",
        description="The severity under the policy, the deadline rule that applies, the computed due date from today, and the KEV rule if relevant, ending with a citation such as [Source: query_private_database, Vulnerability Management Policy p.2]",
    )
    recommended_action: str = Field(
        default="",
        description="Patch, mitigate, ticket or risk acceptance per affected system, the fixed version if known, the owner team, hedged, with citations",
    )
    sources: list[str] = Field(
        default_factory=list, description="Tool names used, e.g. get_cve_details, query_private_database"
    )
    confidence: float | None = Field(default=None, ge=0, le=1, description="How complete the evidence was, 0 to 1")

    def render(self) -> str:
        lines = [self.summary.strip(), ""]
        if self.affected_systems:
            lines.append("## Affected systems")
            for f in self.affected_systems:
                lines.append(f"- **{f.system}** — {f.affected}: {f.evidence.strip()}")
            lines.append("")
        for title, body in (
            ("Exposure and controls", self.exposure),
            ("Exploit intelligence", self.exploit_intelligence),
            ("Urgency", self.urgency),
            ("Policy deadline", self.policy_deadline),
            ("Recommended action", self.recommended_action),
        ):
            if body and body.strip():
                lines += [f"## {title}", body.strip(), ""]
        conf = f"{self.confidence:.0%}" if self.confidence is not None else "not stated"
        lines.append(f"_Confidence: {conf}. Sources: {', '.join(self.sources) or 'none'}._")
        return "\n".join(lines)
