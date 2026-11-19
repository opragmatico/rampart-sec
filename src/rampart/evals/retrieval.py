"""Retrieval evaluation on the gold set: did the expected document come back, at which rank, and how
much of the top-k came from it. Also checks the system filter, which is how the agent is told to query."""

from __future__ import annotations

import json
from pathlib import Path

from ..rag.store import CorpusStore
from ..settings import Settings


def load_gold(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieval_eval(store: CorpusStore, gold_path: Path, k: int = 10) -> tuple[list[dict], dict]:
    gold = load_gold(gold_path)
    rows = []
    for g in gold:
        hits = store.search(g["question"], k=k)
        docs = [h.doc for h in hits]
        rank = next((i + 1 for i, d in enumerate(docs) if d == g["doc"]), None)
        filtered = store.search(g["question"], k=3, system=g["system"]) if g.get("system") else []
        rows.append(
            {
                "question": g["question"],
                "doc": g["doc"],
                "hit": rank is not None,
                "rank": rank,
                "right_doc_share": docs.count(g["doc"]) / len(docs) if docs else 0.0,
                "filtered_hit": bool(filtered) and all(h.doc == g["doc"] for h in filtered),
            }
        )
    n = len(rows) or 1
    summary = {
        "questions": len(rows),
        "k": k,
        "hit_rate": round(sum(r["hit"] for r in rows) / n, 3),
        "top1_rate": round(sum(r["rank"] == 1 for r in rows) / n, 3),
        "mean_right_doc_share": round(sum(r["right_doc_share"] for r in rows) / n, 3),
        "filtered_hit_rate": round(sum(r["filtered_hit"] for r in rows) / n, 3),
    }
    return rows, summary


def retrieval_report(settings: Settings, store: CorpusStore, k: int | None = None) -> dict:
    rows, summary = retrieval_eval(store, settings.evals_dir / "retrieval_gold.jsonl", k=k or settings.retrieval_k)
    return {"summary": summary, "rows": rows}
