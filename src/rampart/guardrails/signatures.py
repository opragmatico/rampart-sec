"""BM25 signature matching: layer 1 of the input guard and the tool guard.

A signature is a canonical attack phrase. Text is tokenised into non-stopword unigrams plus
adjacent bigrams, so "ignore all of the previous instructions, please" still matches
"ignore previous instructions" (the bigram survives the filler words), while a sentence that
merely contains "ignore" and "instructions" far apart earns only unigram credit.

Scores are normalised by each signature's self-score, so an exact phrase scores about 1.0 and
unrelated text about 0. The threshold is calibrated on the evaluation sets (`rampart eval guardrail`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

STOPWORDS = frozenset(
    """
a an the and or but if then else of to in on at by for from with without into onto over under
is are was were be been being am do does did done have has had having can could may might shall
should will would must this that these those it its it's i me my mine we us our you your yours he
him his she her hers they them their there here what which who whom whose when where why how
as so than too very just also not no nor only own same such any all some each every both few more
most other another about above below up down out off again further once please kindly
""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def _stem(w: str) -> str:
    """Plural-only stemming, applied identically to signatures and queries (rules -> rule, boxes -> box)."""
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        if w.endswith(("ses", "xes", "zes", "ches", "shes")):
            return w[:-2]
        return w[:-1]
    return w


def tokenize_spans(text: str) -> list[tuple[str, int, int]]:
    """Non-stopword tokens with their character spans."""
    out = []
    for m in _WORD.finditer(text.lower()):
        w = m.group()
        if w in STOPWORDS or len(w) < 2:
            continue
        out.append((_stem(w), m.start(), m.end()))
    return out


def features(tokens: list[str]) -> list[str]:
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]


@dataclass
class SignatureMatch:
    score: float
    signature: str | None
    family: str | None
    window: str
    start: int  # char offsets of the scored window in the input text
    end: int

    def as_log(self) -> dict:
        return {"score": round(self.score, 3), "signature": self.signature, "family": self.family}


class SignatureIndex:
    WINDOW, STRIDE = 24, 12  # tokens

    def __init__(self, signatures: list[tuple[str, str]]):
        self.families = [f for f, _ in signatures]
        self.texts = [t for _, t in signatures]
        self.docs = [features([tok for tok, _, _ in tokenize_spans(t)]) for t in self.texts]
        self.bigrams = [frozenset(f for f in d if "_" in f) for d in self.docs]
        self.bm25 = BM25Okapi(self.docs)
        self.self_scores = [max(self.bm25.get_scores(d)[i], 1e-9) for i, d in enumerate(self.docs)]

    @classmethod
    def load(cls, path: Path) -> SignatureIndex:
        sigs, family = [], "unspecified"
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("## "):
                family = line[3:].strip()
            elif not line.startswith("#"):
                sigs.append((family, line))
        if not sigs:
            raise ValueError(f"No signatures in {path}")
        weak = [t for _, t in sigs if len(tokenize_spans(t)) < 2]
        if weak:
            raise ValueError(
                f"Signatures need at least two non-stopword tokens (one word would match any text "
                f"containing it): {weak}"
            )
        return cls(sigs)

    def __len__(self) -> int:
        return len(self.texts)

    BIGRAM_GATE = 0.5  # a signature with bigrams needs at least one matching bigram to score above this

    def _best(self, feats: list[str]) -> tuple[float, int]:
        """Best normalised score and signature index. Without a shared bigram the score is capped, so a
        signature cannot act as a wildcard for one common word ("rules") spread across a benign sentence."""
        if not feats:
            return 0.0, -1
        qset = set(feats)
        scores = self.bm25.get_scores(feats)
        best_s, best_i = 0.0, -1
        for i, raw in enumerate(scores):
            s = float(raw / self.self_scores[i])
            if self.bigrams[i] and not (qset & self.bigrams[i]):
                s = min(s, self.BIGRAM_GATE)
            if s > best_s:
                best_s, best_i = s, i
        return best_s, best_i

    def score(self, text: str) -> SignatureMatch:
        """Best normalised score over the whole text (short) or over sliding windows (long)."""
        matches = self.scan(text, threshold=-1.0)
        return max(matches, key=lambda m: m.score) if matches else SignatureMatch(0.0, None, None, text, 0, len(text))

    def scan(self, text: str, threshold: float) -> list[SignatureMatch]:
        """Every window scoring above the threshold, with character offsets for redaction."""
        spans = tokenize_spans(text)
        if not spans:
            return []
        toks = [t for t, _, _ in spans]
        windows = (
            [(0, len(toks))]
            if len(toks) <= self.WINDOW
            else [
                (i, min(i + self.WINDOW, len(toks)))
                for i in range(0, len(toks) - self.WINDOW + self.STRIDE, self.STRIDE)
            ]
        )
        out = []
        for a, b in windows:
            s, idx = self._best(features(toks[a:b]))
            if s <= threshold:
                continue
            start, end = spans[a][1], spans[b - 1][2]
            # Refine to the sentence(s) that carry the match, so redaction removes the injected
            # sentence and not the whole window of legitimate text around it.
            refined = self._sentences_above(text, start, end, threshold)
            if refined:
                out.extend(refined)
            else:
                out.append(
                    SignatureMatch(
                        s,
                        self.texts[idx] if idx >= 0 else None,
                        self.families[idx] if idx >= 0 else None,
                        text[start:end],
                        start,
                        end,
                    )
                )
        # de-duplicate identical spans from overlapping windows
        seen, uniq = set(), []
        for m in sorted(out, key=lambda m: (m.start, -m.score)):
            if (m.start, m.end) not in seen:
                seen.add((m.start, m.end))
                uniq.append(m)
        return uniq

    _SENT = re.compile(r"[^.!?\n]+[.!?\n]*")

    def _sentences_above(self, text: str, start: int, end: int, threshold: float) -> list[SignatureMatch]:
        hits = []
        for m in self._SENT.finditer(text, start, end):
            sent = m.group()
            s, idx = self._best(features([t for t, _, _ in tokenize_spans(sent)]))
            if s > threshold and idx >= 0:
                hits.append(SignatureMatch(s, self.texts[idx], self.families[idx], sent.strip(), m.start(), m.end()))
        return hits

    @staticmethod
    def redact(
        text: str,
        matches: list[SignatureMatch],
        marker: str = "[content removed by tool guard: suspected prompt injection]",
    ) -> str:
        """Replace matched windows (merged when overlapping) with a marker."""
        if not matches:
            return text
        spans = sorted((m.start, m.end) for m in matches)
        merged = [list(spans[0])]
        for s, e in spans[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out, pos = [], 0
        for s, e in merged:
            out.append(text[pos:s])
            out.append(marker)
            pos = e
        out.append(text[pos:])
        return "".join(out)
