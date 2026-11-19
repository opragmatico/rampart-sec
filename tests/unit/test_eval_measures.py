"""The two measurement evals without a model API: the signature guard's precision/recall sweep and the combined
input guard over the attack and benign sets, and the retriever's hit rate over a gold set."""

from __future__ import annotations

import json

from rampart.evals.guardrail import combined_report, load_sets, signature_report
from rampart.evals.retrieval import retrieval_eval
from rampart.guardrails.nodes import Guards
from rampart.rag.store import Hit
from tests.stubs import StubClassifier, StubJudge


def test_the_signature_layer_is_precise_at_the_configured_threshold(settings):
    report = signature_report(settings)
    attacks, benign = load_sets(settings)
    assert report["attacks"] == len(attacks) == 38 and report["benign"] == len(benign) == 38
    at = next(r for r in report["sweep"] if r["threshold"] == settings.injection_threshold)
    assert at["precision"] == 1.0 and at["fp"] == 0 and at["tp"] + at["fn"] == 38
    assert report["at_configured_threshold"]["false_alarms"] == []
    assert len(report["at_configured_threshold"]["missed_attacks"]) == at["fn"]


async def test_the_combined_guard_adds_the_classifier_on_what_signatures_miss(settings):
    layer1 = next(r for r in signature_report(settings)["sweep"] if r["threshold"] == settings.injection_threshold)

    quiet = Guards(settings, store=None, classifier=StubClassifier(hit=False), judge=StubJudge())
    combined = (await combined_report(settings, quiet))["combined"]["combined"]
    assert combined["tp"] == layer1["tp"] and combined["fp"] == 0  # the classifier adds nothing

    eager = Guards(settings, store=None, classifier=StubClassifier(hit=True), judge=StubJudge())
    combined = (await combined_report(settings, eager))["combined"]
    assert combined["combined"]["recall"] == 1.0 and combined["combined"]["fp"] == 38  # it flags every benign question
    assert combined["llm_only_on_items_layer1_passed"]["attacks_seen"] == layer1["fn"]


class _Store:
    """Answers each question with a fixed ranking of documents."""

    def __init__(self, rankings: dict[str, list[str]]):
        self.rankings = rankings

    def search(self, query: str, k: int | None = None, system: str | None = None) -> list[Hit]:
        docs = self.rankings[query] if system is None else [d for d in self.rankings[query] if system in d]
        return [Hit(f"{d}-{i}", d, d, system or "", 1, 1.0 - i / 10, "text") for i, d in enumerate(docs[: k or 10])]


def test_retrieval_eval_reports_rank_share_and_the_filtered_hit(tmp_path):
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        "\n".join(
            json.dumps(g)
            for g in [
                {"question": "q1", "doc": "SRA-cicd.pdf", "system": "cicd"},
                {"question": "q2", "doc": "SRA-data.pdf", "system": "data"},
            ]
        )
    )
    store = _Store({"q1": ["SRA-cicd.pdf", "SRA-cicd.pdf", "SRA-data.pdf"], "q2": ["SRA-cicd.pdf", "POLICY.pdf"]})
    rows, summary = retrieval_eval(store, gold, k=3)
    assert [(r["hit"], r["rank"]) for r in rows] == [(True, 1), (False, None)]
    assert rows[0]["right_doc_share"] == 2 / 3 and rows[0]["filtered_hit"] is True and rows[1]["filtered_hit"] is False
    assert summary == {
        "questions": 2,
        "k": 3,
        "hit_rate": 0.5,
        "top1_rate": 0.5,
        "mean_right_doc_share": 0.333,
        "filtered_hit_rate": 0.5,
    }
