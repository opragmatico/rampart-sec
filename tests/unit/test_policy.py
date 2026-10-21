from datetime import date

from rampart.policy import apply, base_severity


def test_severity_rules_in_order():
    assert base_severity(10.0, 0.9, True, "internet-facing")[0] == "critical"
    assert base_severity(9.5, 0.1, False, "internet-facing")[0] == "critical"
    assert base_severity(10.0, 0.9, True, "internal")[0] == "high"
    assert base_severity(9.8, 0.01, False, "internal")[0] == "high"
    assert base_severity(7.5, 0.01, False, "internet-facing")[0] == "high"
    assert base_severity(5.0, 0.6, False, "internet-facing")[0] == "high"
    assert base_severity(8.8, 0.01, False, "internal")[0] == "medium"
    assert base_severity(5.0, 0.2, False, "internal")[0] == "medium"
    assert base_severity(3.0, 0.01, False, "isolated")[0] == "low"


def test_restricted_data_raises_one_level_and_deadlines_follow():
    o = apply(
        cvss=10.0,
        epss=0.97,
        in_kev=True,
        kev_due_date="2026-08-14",
        exposure="internal",
        data_classification="Restricted",
        today=date(2026, 10, 16),
    )
    assert o.base_severity == "high" and o.severity == "critical" and o.priority == "P1"
    assert o.deadline_days == 14 and o.due_date == "2026-10-30" and not o.kev_due_applies and o.ticket_required


def test_kev_due_date_applies_when_earlier():
    o = apply(
        cvss=9.5,
        epss=0.5,
        in_kev=True,
        kev_due_date="2026-10-20",
        exposure="internet-facing",
        data_classification="Restricted",
        today=date(2026, 10, 16),
    )
    assert o.severity == "critical" and o.deadline_days == 7 and o.due_date == "2026-10-20" and o.kev_due_applies


def test_low_needs_no_ticket():
    o = apply(
        cvss=3.1,
        epss=0.001,
        in_kev=False,
        kev_due_date=None,
        exposure="isolated",
        data_classification="Internal",
        today=date(2026, 10, 16),
    )
    assert o.severity == "low" and o.priority == "P4" and not o.ticket_required and o.deadline_days == 180
