"""The corpus builder without the network: version-range logic, and the PDFs plus inventory it produces from
the YAML specs when NVD verification is skipped."""

from __future__ import annotations

import json
import shutil

from rampart.corpus.build import build_corpus, load_specs, version_in_range
from rampart.rag.ingest import load_pdf, sha256_of

SYSTEM_PDFS = {
    "SRA-customer-portal.pdf",
    "SRA-cicd-platform.pdf",
    "SRA-collaboration-suite.pdf",
    "SRA-data-platform.pdf",
    "SRA-ai-assistant-platform.pdf",
}


def test_version_ranges_compare_numeric_and_alphabetic_runs():
    assert version_in_range("14.1-72.18", {"start_incl": "14.1", "end_excl": "14.1-73.37"}) is True
    assert version_in_range("14.1-73.37", {"end_excl": "14.1-73.37"}) is False
    assert version_in_range("R35", {"end_excl": "R36"}) is True
    assert version_in_range("10.1.55", {"start_incl": "10.1.0", "end_incl": "10.1.50"}) is False
    assert version_in_range("2.568.1", {"start_incl": "2.570"}) is False
    assert version_in_range("1.0", {}) is None  # a CPE without a range says nothing


def test_specs_load_as_company_policy_and_systems(settings):
    company, policy, systems = load_specs(settings.corpus_dir / "specs")
    assert company["name"].startswith("Larkspur") and policy["id"] == "policy"
    assert [s["id"] for s in systems] == [
        "customer-portal",
        "cicd-platform",
        "collaboration-suite",
        "data-platform",
        "ai-assistant-platform",
    ]
    # every system runs at least one component with CVEs in the window; the rest are context (e.g. PostgreSQL)
    assert all(any(comp.get("cves") for comp in s["components"]) for s in systems)


def test_specs_build_the_corpus_and_inventory_offline(settings, tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(settings.corpus_dir / "specs", corpus / "specs")
    s = settings.model_copy(update={"corpus_dir": corpus})

    report = build_corpus(s, skip_verify=True)

    assert report.verified is False and report.problems == []
    assert set(report.built) == SYSTEM_PDFS | {"POLICY-vulnerability-management.pdf"}
    assert all(pages >= 2 for pages in report.built.values()), report.built
    assert not (corpus / "SOURCES.md").exists()  # the evidence table needs the NVD verification

    inventory = json.loads((corpus / "inventory.json").read_text())
    assert report.inventory_systems == len(inventory) == 5
    assert inventory["customer-portal"]["exposure"] == "internet-facing"
    assert inventory["data-platform"]["data_classification"] == "Restricted"
    assert all(
        {"name", "doc_id", "exposure", "data_classification", "owner_team", "assessment_date", "components"} <= set(v)
        for v in inventory.values()
    )
    # the committed inventory is what the specs produce: the tools read it as data, so it must not drift
    assert inventory == json.loads((settings.corpus_dir / "inventory.json").read_text())

    # the PDF metadata is what ingestion reads: the Subject is the system id used as the retrieval filter
    pages = load_pdf(corpus / "SRA-cicd-platform.pdf")
    assert pages and pages[0].metadata["doc"] == "SRA-cicd-platform.pdf"
    assert pages[0].metadata["system"] == "cicd-platform"
    assert pages[0].metadata["title"].startswith("System Risk Assessment")
    assert pages[0].metadata["sha256"] == sha256_of(corpus / "SRA-cicd-platform.pdf")
    assert [p.metadata["page"] for p in pages] == list(range(1, len(pages) + 1))
    assert load_pdf(corpus / "POLICY-vulnerability-management.pdf")[0].metadata["system"] == "policy"


def test_only_rebuilds_one_document(settings, tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(settings.corpus_dir / "specs", corpus / "specs")
    report = build_corpus(settings.model_copy(update={"corpus_dir": corpus}), skip_verify=True, only="policy")
    assert set(report.built) == {"POLICY-vulnerability-management.pdf"}
    assert (corpus / "inventory.json").exists() and not any(corpus.glob("SRA-*.pdf"))
