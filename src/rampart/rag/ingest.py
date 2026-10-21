"""Ingestion job: PDFs -> pages -> chunks -> embeddings -> Qdrant. Idempotent and incremental.

A document whose sha256 already sits in the collection is skipped; a changed document is
deleted and re-embedded; nothing else is touched. Run it with `rampart index`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from ..logging import get_logger
from ..settings import Settings
from .store import CorpusStore

log = get_logger(__name__)


@dataclass
class IngestReport:
    scanned: int = 0
    skipped: list[str] = field(default_factory=list)
    ingested: dict[str, int] = field(default_factory=dict)  # doc -> chunks
    total_points: int = 0

    def as_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "skipped": self.skipped,
            "ingested": self.ingested,
            "total_points": self.total_points,
        }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pdf(path: Path) -> list[Document]:
    """One Document per page with doc / title / system / page metadata (system comes from the PDF's Subject)."""
    reader = PdfReader(str(path))
    meta = reader.metadata or {}
    title = (meta.title or path.stem).strip()
    system = (meta.subject or "").strip() or "unspecified"
    digest = sha256_of(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        pages.append(
            Document(
                page_content=text,
                metadata={"doc": path.name, "title": title, "system": system, "page": i, "sha256": digest},
            )
        )
    return pages


def chunk_pages(pages: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks


def ingest(settings: Settings, store: CorpusStore | None = None, force: bool = False) -> IngestReport:
    store = store or CorpusStore(settings)
    store.ensure_collection()
    report = IngestReport()
    pdfs = sorted(settings.corpus_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {settings.corpus_dir}. Run scripts/build_corpus.py first.")
    for pdf in pdfs:
        report.scanned += 1
        digest = sha256_of(pdf)
        if not force and store.existing_hash(pdf.name) == digest:
            report.skipped.append(pdf.name)
            log.info("ingest.skip_unchanged", doc=pdf.name)
            continue
        if store.existing_hash(pdf.name):
            store.delete_doc(pdf.name)
            log.info("ingest.replace", doc=pdf.name)
        chunks = chunk_pages(load_pdf(pdf), settings)
        n = store.upsert_chunks(chunks)
        report.ingested[pdf.name] = n
        log.info("ingest.done", doc=pdf.name, pages=len({c.metadata["page"] for c in chunks}), chunks=n)
    report.total_points = store.count()
    return report
