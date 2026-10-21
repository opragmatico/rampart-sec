from rampart.guardrails.overlap import overlap_ratio
from rampart.guardrails.redaction import Redactor
from rampart.guardrails.softening import soften


def test_redaction_replaces_identifiers_with_stable_pseudonyms():
    r = Redactor(["corp.example", "example.com"])
    text = "pay-api-01.corp.example (192.0.2.10) owner mara.lindqvist@example.com; also 192.0.2.10 again; version 3.0.14 stays"
    out = r.redact(text)
    assert out.counts == {"ipv4": 2, "hostname": 1, "email": 1}
    assert "192.0.2.10" not in out.text and "corp.example" not in out.text and "@" not in out.text
    assert "3.0.14" in out.text
    first = out.text.split("(")[1].split(")")[0]
    assert out.text.count(first) == 2  # same ip -> same pseudonym


def test_softening_hedges_assertions_and_keeps_negated_meaning():
    out, changes = soften("The gateway is 100% safe and there is no need to patch; it is not affected.")
    assert "likely safe" in out and "less urgency to patch" in out and "does not appear to be affected" in out
    out2, _ = soften("Nothing is 100% safe, and no system has zero risk.")
    assert "absolutely safe" in out2 and "zero risk" in out2
    out3, _ = soften("It is not affected and will definitely stay secure.")
    assert "may stay secure" in out3


def test_overlap_detects_paste_and_passes_paraphrase():
    src = (
        "The payments gateway terminates TLS at the edge using OpenSSL 3.5.1 on two nginx hosts behind the WAF "
        "and forwards requests to the Spring Boot services on the internal segment."
    )
    paste = src + " [Source: query_private_database, Customer Portal p.3]"
    para = "Per the assessment, the gateway uses OpenSSL 3.5.1 behind a WAF and talks to internal Spring services."
    assert overlap_ratio(paste, [src]) > 0.6
    assert overlap_ratio(para, [src]) < 0.1
