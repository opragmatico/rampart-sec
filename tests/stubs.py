"""Stand-ins for the model-backed guards, shared by the guard-node and graph tests."""

from __future__ import annotations


class StubClassifier:
    """Stands in for LlmClassifier: answers as told and records what it was asked."""

    model = "stub"

    def __init__(self, hit: bool = False):
        self.hit, self.calls = hit, []

    async def classify(self, text: str, config=None) -> dict:
        self.calls.append(text)
        verdict = "INJECTION" if self.hit else "SAFE"
        return {"hit": self.hit, "verdict": verdict, "reason": "stub", "error": None}


class StubJudge:
    """Stands in for LlmJudge: answers as told and records what it was asked."""

    model = "stub"

    def __init__(self, verdict: str = "SAFE"):
        self.verdict, self.calls = verdict, []

    async def judge_answer(self, text: str, config=None) -> dict:
        self.calls.append(text)
        return {"verdict": self.verdict, "reasons": ["stub"], "error": None}
