"""Rule-based softening of absolute assurance language.

Two passes, as in the original project. A phrase inside a sentence that already negates it
("nothing is 100% safe") is reworded with the same meaning rather than inverted. An asserted
phrase ("the system is 100% safe") is hedged.
"""

from __future__ import annotations

import re

NEGATION = re.compile(
    r"\b(?:no|not|never|nothing|none|nor|neither|cannot|can't|isn't|aren't|won't|wouldn't|"
    r"doesn't|don't|without)\b",
    re.IGNORECASE,
)
# A negation only reaches a phrase inside its own clause: punctuation or a conjunction ends the scope.
_CLAUSE_BREAK = re.compile(r"[.!?;:,\n]|\b(?:and|but|or|so|then|while|whereas|although)\b", re.IGNORECASE)

# (asserted pattern, hedged replacement, same-meaning replacement when the sentence negates it)
RULES: list[tuple[str, str, str]] = [
    (
        r"\b(?:100\s*%|completely|totally|entirely|fully|absolutely)\s+(safe|secure|protected|patched|mitigated|immune)\b",
        r"likely \1",
        r"absolutely \1",
    ),
    (r"\bno\s+risk\b", "low risk", "zero risk"),
    (r"\bzero\s+risk\b", "low risk", "zero risk"),
    (r"\bno\s+need\s+to\s+patch\b", "less urgency to patch", "no reason to patch"),
    (r"\bimpossible\s+to\s+exploit\b", "difficult to exploit", "unexploitable"),
    (r"\bcannot\s+be\s+exploited\b", "is unlikely to be exploited", "cannot be exploited"),
    (r"\bwill\s+definitely\b", "may", "will certainly"),
    (r"\bguaranteed\s+(?:to\s+be\s+)?(safe|secure|protected|unaffected)\b", r"likely \1", r"certain to be \1"),
    (r"\bis\s+not\s+affected\b", "does not appear to be affected", "is not affected"),
    (r"\bare\s+not\s+affected\b", "do not appear to be affected", "are not affected"),
    (r"\bsafe\s+to\s+ignore\b", "lower priority", "safe to ignore"),
    (r"\bcan\s+be\s+ignored\b", "may be lower priority", "can be ignored"),
    (r"\bno\s+action\s+(?:is\s+)?(?:needed|required)\b", "limited action may be needed", "no action needed"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), hedged, same) for p, hedged, same in RULES]


def _clause_prefix(text: str, pos: int, max_words: int = 6) -> str:
    """The words of the current clause that precede `pos` (at most `max_words`)."""
    starts = [m.end() for m in _CLAUSE_BREAK.finditer(text, 0, pos)]
    prefix = text[(starts[-1] if starts else 0) : pos]
    return " ".join(prefix.split()[-max_words:])


def soften(text: str) -> tuple[str, list[dict]]:
    """Return the softened text and a list of the substitutions made."""
    changes: list[dict] = []
    out = text
    for rx, hedged, same in _COMPILED:
        out = rx.sub(_substitution(out, hedged, same, changes), out)
    return out, changes


def _substitution(text: str, hedged: str, same: str, changes: list[dict]):
    """The replacement callback for one rule over one version of the text."""

    def _sub(m: re.Match) -> str:
        negated = bool(NEGATION.search(_clause_prefix(text, m.start())))
        template = same if negated else hedged
        repl = m.expand(template)
        if repl.lower() != m.group().lower():
            changes.append({"from": m.group(), "to": repl, "negated": negated})
        return repl

    return _sub
