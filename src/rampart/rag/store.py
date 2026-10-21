"""The corpus in Qdrant: one collection, dense vectors, payload filters by system and document.

Two modes from one code path: local mode (on-disk, no server) when QDRANT_URL is unset, and a
Qdrant server otherwise. Point ids are deterministic (UUID5 of `doc:chunk_index`), so ingesting
the same document twice overwrites instead of duplicating.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from ..llm import retry
from ..settings import Settings

NAMESPACE = uuid.UUID("3f1c6c1e-9d0f-4f0a-9f4b-0b6b7e3b2a11")
EMBED_DIM = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}


def point_id(doc: str, chunk_index: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{doc}:{chunk_index}"))


@dataclass
class Hit:
    chunk_id: str
    doc: str
    title: str
    system: str
    page: int
    score: float
    text: str

    def ref(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc": self.doc,
            "title": self.title,
            "system": self.system,
            "page": self.page,
            "score": round(self.score, 4),
        }


class CorpusStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
                timeout=30,
            )
            self.mode = "server"
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))
            self.mode = "local"
        self.collection = settings.qdrant_collection
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_s,
        )
        self.dim = EMBED_DIM.get(settings.embedding_model, 1536)
        self._vs: QdrantVectorStore | None = None

    # ── collection lifecycle ────────────────────────────────────────────────
    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection, vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            )
        if self.mode == "server":  # local mode filters without indexes and warns if asked to create them
            for field in ("metadata.system", "metadata.doc", "metadata.sha256"):
                try:
                    self.client.create_payload_index(
                        self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception:  # already exists
                    pass

    def vectorstore(self) -> QdrantVectorStore:
        """Built once: the constructor validates the collection by embedding a probe text, which would
        otherwise cost one embeddings call per search."""
        if self._vs is None:
            self.ensure_collection()
            self._vs = retry(
                lambda: QdrantVectorStore(
                    client=self.client, collection_name=self.collection, embedding=self.embeddings
                ),
                what="vectorstore.init",
            )
        return self._vs

    def ready(self) -> bool:
        try:
            return self.client.collection_exists(self.collection) and self.count() > 0
        except Exception:
            return False

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    # ── ingestion helpers ───────────────────────────────────────────────────
    def existing_hash(self, doc: str) -> str | None:
        """sha256 stored with the document's points, or None when it was never ingested."""
        if not self.client.collection_exists(self.collection):
            return None
        pts, _ = self.client.scroll(
            self.collection, limit=1, with_payload=["metadata.sha256"], scroll_filter=self._filter(doc=doc)
        )
        if not pts:
            return None
        return (pts[0].payload or {}).get("metadata", {}).get("sha256")

    def delete_doc(self, doc: str) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=self._filter(doc=doc)))

    def upsert_chunks(self, chunks: list[Document]) -> int:
        vs = self.vectorstore()
        ids = [point_id(c.metadata["doc"], c.metadata["chunk_index"]) for c in chunks]
        retry(lambda: vs.add_documents(chunks, ids=ids, batch_size=64), what="add_documents")
        return len(ids)

    NON_SYSTEM_SUBJECTS = {"policy"}

    def systems(self) -> list[str]:
        """Systems covered by the assessments (the policy document is a subject, not a system)."""
        return [s for s in self.subjects() if s not in self.NON_SYSTEM_SUBJECTS]

    def chunk_counts(self) -> dict[str, int]:
        """Chunks per subject, so a caller can tell when a filtered search returned a whole document."""
        return {d["system"]: d["chunks"] for d in self.documents()}

    def subjects(self) -> list[str]:
        """Distinct subject values in the collection (small corpus: a scroll is fine)."""
        if not self.client.collection_exists(self.collection):
            return []
        seen: set[str] = set()
        offset = None
        while True:
            pts, offset = self.client.scroll(
                self.collection, limit=256, with_payload=["metadata.system"], offset=offset
            )
            for p in pts:
                seen.add((p.payload or {}).get("metadata", {}).get("system", ""))
            if offset is None:
                break
        return sorted(s for s in seen if s)

    def documents(self) -> list[dict]:
        """One row per ingested document: doc, title, system, chunks, pages."""
        if not self.client.collection_exists(self.collection):
            return []
        rows: dict[str, dict] = {}
        offset = None
        while True:
            pts, offset = self.client.scroll(self.collection, limit=256, with_payload=["metadata"], offset=offset)
            for p in pts:
                m = (p.payload or {}).get("metadata", {})
                r = rows.setdefault(
                    m.get("doc", "?"),
                    {"doc": m.get("doc"), "title": m.get("title"), "system": m.get("system"), "chunks": 0, "pages": 0},
                )
                r["chunks"] += 1
                r["pages"] = max(r["pages"], int(m.get("page", 0)))
            if offset is None:
                break
        return sorted(rows.values(), key=lambda r: r["doc"])

    # ── retrieval ───────────────────────────────────────────────────────────
    @staticmethod
    def _filter(system: str | None = None, doc: str | None = None) -> models.Filter | None:
        must = []
        if system:
            must.append(models.FieldCondition(key="metadata.system", match=models.MatchValue(value=system)))
        if doc:
            must.append(models.FieldCondition(key="metadata.doc", match=models.MatchValue(value=doc)))
        return models.Filter(must=must) if must else None

    def search(self, query: str, k: int | None = None, system: str | None = None) -> list[Hit]:
        vs = self.vectorstore()
        pairs = retry(
            lambda: vs.similarity_search_with_score(
                query, k=k or self.settings.retrieval_k, filter=self._filter(system=system)
            ),
            what="search",
        )
        return [
            Hit(
                chunk_id=point_id(d.metadata["doc"], d.metadata["chunk_index"]),
                doc=d.metadata["doc"],
                title=d.metadata.get("title", d.metadata["doc"]),
                system=d.metadata.get("system", ""),
                page=int(d.metadata.get("page", 0)),
                score=float(s),
                text=d.page_content,
            )
            for d, s in pairs
        ]

    def fetch_chunks(self, chunk_ids: list[str]) -> dict[str, str]:
        """Chunk texts by id, for the output guard's overlap check."""
        if not chunk_ids:
            return {}
        pts = self.client.retrieve(self.collection, ids=chunk_ids, with_payload=["page_content"])
        return {str(p.id): (p.payload or {}).get("page_content", "") for p in pts}

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
