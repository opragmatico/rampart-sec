"""Precision / recall of the signature guard on the evaluation sets, per threshold, and of the combined
input guard (signatures OR the LLM classifier) when model calls are allowed."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ..guardrails.signatures import SignatureIndex
from ..settings import Settings

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sets(settings: Settings) -> tuple[list[dict], list[dict]]:
    """(attacks, benign analyst questions) from data/evals."""
    return load_jsonl(settings.evals_dir / "injections.jsonl"), load_jsonl(settings.evals_dir / "benign.jsonl")


def score_sets(index: SignatureIndex, attacks: list[dict], benign: list[dict]) -> tuple[list[float], list[float]]:
    return ([index.score(a["text"]).score for a in attacks], [index.score(b["text"]).score for b in benign])


def sweep(
    index: SignatureIndex, attacks: list[dict], benign: list[dict], thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
) -> list[dict]:
    a_scores, b_scores = score_sets(index, attacks, benign)
    rows = []
    for t in thresholds:
        tp = sum(s > t for s in a_scores)
        fp = sum(s > t for s in b_scores)
        fn = len(a_scores) - tp
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / len(a_scores) if a_scores else 0.0
        rows.append(
            {
                "threshold": t,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "benign_pass_rate": round(1 - fp / len(b_scores), 3) if b_scores else 1.0,
            }
        )
    return rows


def misses(index: SignatureIndex, attacks: list[dict], benign: list[dict], threshold: float) -> dict:
    missed = [
        {
            "text": a["text"],
            "family": a.get("family"),
            "variant": a.get("variant"),
            "score": round(index.score(a["text"]).score, 3),
        }
        for a in attacks
        if index.score(a["text"]).score <= threshold
    ]
    false_alarms = []
    for b in benign:
        m = index.score(b["text"])
        if m.score > threshold:
            false_alarms.append({"text": b["text"], "score": round(m.score, 3), "signature": m.signature})
    return {"missed_attacks": missed, "false_alarms": false_alarms}


def signature_report(settings: Settings, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    """Layer 1 alone: no model calls, runs in milliseconds."""
    index = SignatureIndex.load(settings.signatures_path)
    attacks, benign = load_sets(settings)
    ts = tuple(sorted(set(thresholds) | {settings.injection_threshold}))
    return {
        "signatures": len(index),
        "attacks": len(attacks),
        "benign": len(benign),
        "configured_threshold": settings.injection_threshold,
        "sweep": sweep(index, attacks, benign, ts),
        "at_configured_threshold": misses(index, attacks, benign, settings.injection_threshold),
    }


async def llm_sweep(guards, attacks: list[dict], benign: list[dict], threshold: float) -> dict:
    """Layer 2 (the LLM classifier) and the combined input guard (layer 1 OR layer 2) on the same sets.
    Costs one model call per item that layer 1 does not already block."""

    async def _combined(text: str) -> tuple[bool, bool | None, str | None]:
        rule = guards.index.score(text).score > threshold
        if rule:
            return True, None, None
        v = await guards.classifier.classify(text)
        return v["hit"], v["hit"], v["reason"]

    a = await asyncio.gather(*(_combined(x["text"]) for x in attacks))
    b = await asyncio.gather(*(_combined(x["text"]) for x in benign))
    tp = sum(hit for hit, _, _ in a)
    fp = sum(hit for hit, _, _ in b)
    llm_a = [(x, r) for x, r in zip(attacks, a, strict=True) if r[1] is not None]
    llm_b = [(x, r) for x, r in zip(benign, b, strict=True) if r[1] is not None]
    return {
        "threshold": threshold,
        "combined": {
            "tp": tp,
            "fn": len(attacks) - tp,
            "fp": fp,
            "precision": round(tp / (tp + fp), 3) if tp + fp else 1.0,
            "recall": round(tp / len(attacks), 3),
        },
        "llm_only_on_items_layer1_passed": {
            "attacks_seen": len(llm_a),
            "caught": sum(r[1] for _, r in llm_a),
            "benign_seen": len(llm_b),
            "false_alarms": sum(r[1] for _, r in llm_b),
        },
        "missed_attacks": [
            {"text": x["text"][:90], "family": x.get("family"), "variant": x.get("variant"), "llm_reason": r[2]}
            for x, r in zip(attacks, a, strict=True)
            if not r[0]
        ],
        "false_alarms": [{"text": x["text"][:90], "llm_reason": r[2]} for x, r in zip(benign, b, strict=True) if r[0]],
    }


async def combined_report(settings: Settings, guards) -> dict[str, Any]:
    """Layer 1 and the combined guard. `guards` is a `rampart.guardrails.nodes.Guards`; its classifier makes one
    model call per item layer 1 lets through."""
    attacks, benign = load_sets(settings)
    report = signature_report(settings)
    report["combined"] = await llm_sweep(guards, attacks, benign, settings.injection_threshold)
    return report
