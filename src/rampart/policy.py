"""The Vulnerability Management Policy's decision table, as code.

The policy document (data/corpus/POLICY-vulnerability-management.pdf, sections 4 to 6) is the
human-readable source; this module is the executable mirror of its severity rules, deadlines and
ticket priorities, so that a ticket's priority and due date never depend on how the model read the
document. Both are versioned together (POLICY_VERSION matches the document's version).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

POLICY_VERSION = "SEC-POL-007 v3.2"
LEVELS = ["low", "medium", "high", "critical"]
DEADLINE_DAYS = {
    ("critical", "internet-facing"): 7,
    ("critical", "internal"): 14,
    ("critical", "isolated"): 14,
    ("high", "internet-facing"): 14,
    ("high", "internal"): 30,
    ("high", "isolated"): 30,
    ("medium", "internet-facing"): 60,
    ("medium", "internal"): 60,
    ("medium", "isolated"): 60,
    ("low", "internet-facing"): 180,
    ("low", "internal"): 180,
    ("low", "isolated"): 180,
}
PRIORITY = {"critical": "P1", "high": "P2", "medium": "P3", "low": "P4"}


@dataclass
class PolicyOutcome:
    severity: str
    base_severity: str
    priority: str
    deadline_days: int
    due_date: str
    triage_date: str
    kev_due_date: str | None
    kev_due_applies: bool
    ticket_required: bool
    rationale: list[str]
    policy_version: str = POLICY_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


def _bump(level: str, n: int) -> str:
    return LEVELS[max(0, min(len(LEVELS) - 1, LEVELS.index(level) + n))]


def base_severity(cvss: float | None, epss: float | None, in_kev: bool, exposure: str) -> tuple[str, str]:
    """Section 4 of the policy, rules in order; returns (severity, rule that set it)."""
    external = exposure == "internet-facing"
    cvss = cvss or 0.0
    epss = epss or 0.0
    if in_kev and external:
        return "critical", "KEV-listed and internet-facing"
    if cvss >= 9.0 and external:
        return "critical", "CVSS >= 9.0 on an internet-facing system"
    if in_kev:
        return "high", "KEV-listed (any exposure)"
    if cvss >= 9.0:
        return "high", "CVSS >= 9.0 on an internal system"
    if 7.0 <= cvss < 9.0 and external:
        return "high", "CVSS 7.0-8.9 on an internet-facing system"
    if epss >= 0.5 and external:
        return "high", "EPSS >= 0.50 on an internet-facing system"
    if 7.0 <= cvss < 9.0:
        return "medium", "CVSS 7.0-8.9 on an internal system"
    if 4.0 <= cvss < 7.0 and external:
        return "medium", "CVSS 4.0-6.9 on an internet-facing system"
    if 0.10 <= epss < 0.5:
        return "medium", "EPSS 0.10-0.49"
    return "low", "none of the Critical/High/Medium rules applies"


def apply(
    *,
    cvss: float | None,
    epss: float | None,
    in_kev: bool,
    kev_due_date: str | None,
    exposure: str,
    data_classification: str | None,
    today: date | None = None,
) -> PolicyOutcome:
    today = today or date.today()
    base, rule = base_severity(cvss, epss, in_kev, exposure)
    rationale = [f"Section 4: {rule} -> {base}."]
    severity = base
    if (data_classification or "").strip().lower() == "restricted" and severity != "critical":
        severity = _bump(severity, 1)
        rationale.append(f"Section 4: the system processes Restricted data -> raised to {severity}.")
    exp = exposure if exposure in ("internet-facing", "internal", "isolated") else "internal"
    days = DEADLINE_DAYS[(severity, exp)]
    due = today + timedelta(days=days)
    rationale.append(
        f"Section 5: {severity}, {exp} -> {days} days from the triage date {today.isoformat()} = {due.isoformat()}."
    )
    kev_applies = False
    if in_kev and kev_due_date:
        try:
            kd = date.fromisoformat(kev_due_date)
            if today <= kd < due:
                due, kev_applies = kd, True
                rationale.append(f"Section 5 KEV rule: the CISA due date {kd.isoformat()} is earlier and applies.")
            elif kd < today:
                rationale.append(
                    f"Section 5 KEV rule: the CISA due date {kd.isoformat()} has passed; the policy deadline applies "
                    "and the finding is already overdue by CISA's standard."
                )
        except ValueError:
            pass
    if in_kev:
        rationale.append("Section 5 KEV rule: a remediation ticket must be opened within one business day of triage.")
    ticket_required = severity in ("medium", "high", "critical")
    rationale.append(
        f"Section 6: {severity} -> ticket priority {PRIORITY[severity]}; "
        f"{'a ticket is required' if ticket_required else 'no ticket is required for Low'}."
    )
    return PolicyOutcome(
        severity=severity,
        base_severity=base,
        priority=PRIORITY[severity],
        deadline_days=days,
        due_date=due.isoformat(),
        triage_date=today.isoformat(),
        kev_due_date=kev_due_date,
        kev_due_applies=kev_applies,
        ticket_required=ticket_required,
        rationale=rationale,
    )
