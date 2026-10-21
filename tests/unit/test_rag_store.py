"""The Qdrant store in local mode and the ingestion job's error path, without embeddings or a model."""

from __future__ import annotations

import uuid

import pytest

from rampart.rag.ingest import IngestReport, ingest
from rampart.rag.store import CorpusStore, point_id


def test_point_ids_are_deterministic_uuids():
    assert point_id("a.pdf", 1) == point_id("a.pdf", 1)
    assert len({point_id("a.pdf", 1), point_id("a.pdf", 2), point_id("b.pdf", 1)}) == 3
    assert uuid.UUID(point_id("a.pdf", 0)).version == 5


def test_empty_local_store_lifecycle(settings, tmp_path):
    s = settings.model_copy(update={"qdrant_path": tmp_path / "qdrant"})
    store = CorpusStore(s)
    try:
        assert store.mode == "local" and store.collection == s.qdrant_collection
        assert store.ready() is False and store.existing_hash("SRA-cicd-platform.pdf") is None
        store.ensure_collection()
        store.ensure_collection()  # idempotent
        assert store.count() == 0 and store.ready() is False  # a collection without points is not ready
        assert store.documents() == [] and store.subjects() == [] and store.systems() == []
        assert store.chunk_counts() == {} and store.fetch_chunks([]) == {}
        f = CorpusStore._filter(system="cicd-platform", doc="x.pdf")
        assert [c.key for c in f.must] == ["metadata.system", "metadata.doc"]
        assert CorpusStore._filter() is None
    finally:
        store.close()


def test_ingest_without_pdfs_fails_before_any_embedding(settings, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    s = settings.model_copy(update={"qdrant_path": tmp_path / "qdrant", "corpus_dir": empty})
    store = CorpusStore(s)
    try:
        with pytest.raises(FileNotFoundError, match="No PDFs"):
            ingest(s, store=store)
        assert store.count() == 0
    finally:
        store.close()
    assert IngestReport().as_dict() == {"scanned": 0, "skipped": [], "ingested": {}, "total_points": 0}
