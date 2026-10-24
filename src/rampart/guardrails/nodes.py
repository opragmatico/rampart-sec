"""The guardrail nodes of the graph: input guard, tool guard, output guard.

They are plain functions over the graph state so that each can be unit-tested without a model,
a server or a graph (the deterministic parts), and so that every decision lands in
`guardrail_log` with the scores that produced it.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from ..rag.store import CorpusStore
from ..settings import Settings
from .llm_classify import CLASSIFIER_PROMPT_VERSION, LlmClassifier
from .llm_judge import JUDGE_PROMPT_VERSION, LlmJudge
from .overlap import overlap_ratio
from .redaction import Redactor
from .signatures import SignatureIndex
from .softening import soften

DISCLAIMER = (
    "\n\n⚠️ *This triage is decision support generated from NVD, EPSS, CISA KEV, web advisories and the "
    "company's internal documents. It is not a substitute for your security team's assessment and "
    "change process. Verify the cited evidence before acting.*"
)
CITATION_NOTE = (
    "\n\n📌 *No source citations were detected in this answer; verify every claim against the "
    "tools and documents before acting on it.*"
)
WITHHELD = (
    "🚫 This answer was withheld by the output safety checks ({why}). Ask for a summary with "
    "citations, or for a hedged assessment, and the triage will be re-run."
)
REFUSAL = (
    "🚫 I can't help with that request. It was blocked by the input safety checks ({why}). "
    "Please rephrase it as a vulnerability-triage question."
)

_CITE_TOOL = re.compile(r"\[Source:\s*([A-Za-z_][A-Za-z_ ]*?)(?=[,\]\s])", re.IGNORECASE)
_CITE_ANY = re.compile(r"\[source", re.IGNORECASE)
KNOWN_TOOLS = {
    "get_cve_details",
    "get_exploit_intel",
    "search_advisories",
    "assess_urgency",
    "query_private_database",
    "list_systems",
    "open_ticket",
}


def tool_text(msg: BaseMessage) -> str:
    """Text of a ToolMessage whose content may be a string or a list of content blocks."""
    c = msg.content
    if isinstance(c, str):
        return c
    parts = []
    for block in c:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def parse_envelope(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) and "status" in obj else None
    except (ValueError, TypeError):
        return None


class Guards:
    """Holds the shared guard components; the graph binds its nodes to one instance."""

    def __init__(
        self,
        settings: Settings,
        store: CorpusStore | None = None,
        classifier: LlmClassifier | None = None,
        judge: LlmJudge | None = None,
    ):
        self.settings = settings
        self.index = SignatureIndex.load(settings.signatures_path)
        self.redactor = Redactor(settings.internal_domains)
        self.store = store
        self.classifier = classifier or LlmClassifier(settings)
        self.judge = judge or LlmJudge(settings)

    # ── input guard ──────────────────────────────────────────────────────────
    async def input_guard(self, state: dict, config=None) -> dict:
        t0 = time.monotonic()
        query = state["user_query"]
        m = self.index.score(query)
        rule_hit = m.score > self.settings.injection_threshold
        log: dict[str, Any] = {
            "stage": "input_guard",
            "rule_score": round(m.score, 3),
            "rule_signature": m.signature,
            "rule_family": m.family,
            "rule_hit": rule_hit,
            "rule_threshold": self.settings.injection_threshold,
            "llm_hit": None,
            "llm_reason": None,
            "llm_error": None,
            "prompt_version": CLASSIFIER_PROMPT_VERSION,
        }
        if rule_hit:
            log["llm_reason"] = "skipped: layer 1 fired"
            blocked = True
        else:
            v = await self.classifier.classify(query, config)
            log.update(
                {
                    "llm_hit": v["hit"],
                    "llm_reason": v["reason"],
                    "llm_error": v["error"],
                    "llm_model": self.classifier.model,
                }
            )
            blocked = v["hit"]
        log["route"] = "blocked" if blocked else "agent"
        log["latency_ms"] = int((time.monotonic() - t0) * 1000)
        return {"route": log["route"], "guardrail_log": [log]}

    # ── tool guard ───────────────────────────────────────────────────────────
    def tool_guard(self, state: dict) -> dict:
        """Scan the ToolMessages produced by the last tool step; redact suspected injected passages;
        collect which tools answered and which chunks the RAG tool retrieved."""
        msgs: list[BaseMessage] = state["messages"]
        trailing: list[ToolMessage] = []
        for msg in reversed(msgs):
            if isinstance(msg, ToolMessage):
                trailing.append(msg)
            else:
                break
        updates: list[BaseMessage] = []
        logs, tools_called, chunks = [], [], []
        for msg in reversed(trailing):
            text = tool_text(msg)
            env = parse_envelope(text)
            entry: dict[str, Any] = {
                "stage": "tool_guard",
                "tool": msg.name,
                "status": env.get("status") if env else "unparsed",
                "hits": 0,
                "max_score": 0.0,
                "removed_chars": 0,
            }
            if env and env.get("status") == "ok":
                tools_called.append(msg.name)
                if msg.name == "query_private_database":
                    for ref in (env.get("data") or {}).get("retrieved", []):
                        chunks.append({k: ref.get(k) for k in ("chunk_id", "doc", "title", "page", "score")})
            hits = self.index.scan(text, self.settings.tool_guard_threshold)
            if hits:
                redacted = self.index.redact(text, hits)
                entry.update(
                    {
                        "hits": len(hits),
                        "max_score": round(max(h.score for h in hits), 3),
                        "families": sorted({h.family for h in hits if h.family}),
                        "removed_chars": len(text) - len(redacted) + len(hits) * 0,
                        "examples": [h.window[:120] for h in hits[:2]],
                    }
                )
                updates.append(ToolMessage(content=redacted, tool_call_id=msg.tool_call_id, name=msg.name, id=msg.id))
            logs.append(entry)
        out: dict[str, Any] = {"guardrail_log": logs, "tools_called": tools_called, "retrieved_chunks": chunks}
        if updates:
            out["messages"] = updates  # same ids: the add_messages reducer replaces the originals
        return out

    # ── output guard ─────────────────────────────────────────────────────────
    async def output_guard(self, state: dict, config=None) -> dict:
        t0 = time.monotonic()
        draft: str = state.get("draft") or ""
        log: dict[str, Any] = {
            "stage": "output_guard",
            "structured": bool(state.get("final")),
            "prompt_version": JUDGE_PROMPT_VERSION,
        }
        withheld_why: list[str] = []

        # 1. redaction of internal identifiers
        red = self.redactor.redact(draft)
        log["redacted"] = red.counts
        text = red.text

        # 2. softening of absolute assurance
        text, changes = soften(text)
        log["absolute_softened"] = bool(changes)
        log["softening_changes"] = changes[:8]

        # 3. citation presence
        log["has_citation"] = bool(_CITE_ANY.search(text))
        if not log["has_citation"]:
            text += CITATION_NOTE

        # 4. citation-evidence check: every cited tool must have answered in this run; a citation that
        #    names a document (anything that is not a tool name) needs the RAG tool to have run.
        captured = [c.strip().lower() for c in _CITE_TOOL.findall(text)]
        cited = {c for c in captured if c in KNOWN_TOOLS}
        doc_cited = any(c not in KNOWN_TOOLS for c in captured)
        called = set(state.get("tools_called") or [])
        uncalled = sorted(cited - called)
        if doc_cited and "query_private_database" not in called:
            uncalled.append("query_private_database (documents cited without querying them)")
        log["cited_tools"] = sorted(cited) + (["<documents>"] if doc_cited else [])
        log["uncalled_citations"] = uncalled
        if uncalled:
            withheld_why.append(f"cites tools that did not run: {', '.join(uncalled)}")

        # 5. verbatim overlap with the retrieved chunks
        ratio = 0.0
        ids = [c["chunk_id"] for c in state.get("retrieved_chunks") or [] if c.get("chunk_id")]
        if ids and self.store is not None:
            try:
                sources = list(self.store.fetch_chunks(list(dict.fromkeys(ids))).values())
                ratio = overlap_ratio(text, sources)
            except Exception as e:  # noqa: BLE001
                log["overlap_error"] = f"{type(e).__name__}: {e}"
        log["overlap_ratio"] = round(ratio, 3)
        if ratio > self.settings.overlap_threshold:
            withheld_why.append(f"verbatim overlap {ratio:.0%} with internal documents")

        # 6. disclaimer
        log["disclaimer_added"] = "not a substitute for your security team" not in text.lower()
        if log["disclaimer_added"]:
            text += DISCLAIMER

        # 7. LLM judge (last, on the text the user would see)
        j = await self.judge.judge_answer(text, config)
        log.update(
            {
                "judge": j["verdict"],
                "judge_reasons": j["reasons"],
                "judge_error": j["error"],
                "judge_model": self.judge.model,
            }
        )
        if j["verdict"] != "SAFE":
            withheld_why.append("judge: " + "; ".join(j["reasons"][:3]))

        withheld = bool(withheld_why)
        log["withheld"] = withheld
        log["withheld_why"] = withheld_why
        log["latency_ms"] = int((time.monotonic() - t0) * 1000)
        final = WITHHELD.format(why="; ".join(withheld_why)) + DISCLAIMER if withheld else text

        messages: list[BaseMessage] = []
        if state.get("final_call_id"):
            messages.append(
                ToolMessage(content="final answer recorded", tool_call_id=state["final_call_id"], name="final_answer")
            )
        messages.append(AIMessage(content=final))
        return {"messages": messages, "guardrail_log": [log], "withheld": withheld, "delivered": final}

    @staticmethod
    def refuse(state: dict) -> dict:
        entry = state["guardrail_log"][-1] if state.get("guardrail_log") else {}
        why = "signature match" if entry.get("rule_hit") else "classifier"
        return {
            "messages": [AIMessage(content=REFUSAL.format(why=why))],
            "withheld": True,
            "delivered": REFUSAL.format(why=why),
        }
