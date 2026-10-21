import pytest
from pydantic import ValidationError

from rampart.tools.models import Envelope, TicketRequest, UrgencyEvidence


def test_envelope_shapes():
    ok = Envelope.ok("nvd", {"a": 1}).model_dump()
    assert ok["status"] == "ok" and ok["data"] == {"a": 1} and ok["error"] is None
    bad = Envelope.fail("nvd", "NOT_FOUND", "nope").model_dump()
    assert bad["status"] == "error" and bad["error"]["code"] == "NOT_FOUND"


def test_inputs_are_normalised_and_validated():
    assert UrgencyEvidence(cve_id="cve-2026-1234").cve_id == "CVE-2026-1234"
    with pytest.raises(ValidationError):
        UrgencyEvidence(cve_id="nope")
    with pytest.raises(ValidationError):
        TicketRequest(cve_id="CVE-2026-1", system="x", priority="P9", due_date="2030-01-01", summary="s")
