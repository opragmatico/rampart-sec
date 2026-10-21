"""Deterministic redaction of internal identifiers in outbound text.

The answer channel is less protected than the inventory, so hostnames, IP addresses and e-mail
addresses are replaced by stable pseudonyms. Regex is the right tool here because the job is
substitution, which a similarity score cannot do.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"(?<![\w.])(?:{_OCTET}\.){{3}}{_OCTET}(?:/\d{{1,2}})?(?![\w.])")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _tag(prefix: str, raw: str) -> str:
    return f"{prefix}-{hashlib.sha1(raw.lower().encode()).hexdigest()[:4]}"


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class Redactor:
    def __init__(self, internal_domains: list[str]):
        doms = "|".join(re.escape(d) for d in internal_domains)
        # host names under an internal domain, e.g. pay-api-01.corp.example
        self.host_re = re.compile(rf"\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:{doms})\b", re.IGNORECASE)
        self.internal_domains = internal_domains

    def redact(self, text: str) -> RedactionResult:
        counts = {"ipv4": 0, "hostname": 0, "email": 0}

        def _ip(m: re.Match) -> str:
            counts["ipv4"] += 1
            return _tag("ip", m.group())

        def _host(m: re.Match) -> str:
            counts["hostname"] += 1
            return _tag("host", m.group())

        def _email(m: re.Match) -> str:
            counts["email"] += 1
            return "[email]"

        out = EMAIL_RE.sub(_email, text)  # e-mails first: their domains would otherwise match as hosts
        out = self.host_re.sub(_host, out)
        out = IPV4_RE.sub(_ip, out)
        return RedactionResult(out, {k: v for k, v in counts.items() if v})

    def find(self, text: str) -> dict[str, list[str]]:
        """Identifiers present, without changing the text (used by tests and the judge log)."""
        return {"ipv4": IPV4_RE.findall(text), "hostname": self.host_re.findall(text), "email": EMAIL_RE.findall(text)}
