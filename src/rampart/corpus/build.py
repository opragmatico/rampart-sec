"""Build the internal corpus (five System Risk Assessments + the policy) as PDFs from YAML specs.

Every CVE a spec refers to is verified against NVD at build time: it must exist, be published inside
the research window, and (when NVD has CPE data) the spec's component version must fall inside a
vulnerable range. The evidence is written to data/corpus/SOURCES.md and the structured inventory to
data/corpus/inventory.json. The company is fictitious; the software, versions and CVEs are real.

Usage: rampart build-corpus [--skip-verify] [--only policy|<system-id>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..http import HttpClient
from ..intel import kev, nvd
from ..settings import Settings, get_settings

WINDOW = ("2026-06-16", "2026-10-16")  # CVEs in the corpus were published inside this window

# ── styles ────────────────────────────────────────────────────────────────────
_ss = getSampleStyleSheet()
BODY = ParagraphStyle(
    "body", parent=_ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6
)
H1 = ParagraphStyle("h1", parent=_ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, spaceBefore=10, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
TITLE = ParagraphStyle("title", parent=_ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=10)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=11, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=11, alignment=0, spaceAfter=0)


@dataclass
class BuildReport:
    built: dict[str, int] = field(default_factory=dict)  # pdf name -> pages
    inventory_systems: int = 0
    problems: list[str] = field(default_factory=list)
    verified: bool = False

    def as_dict(self) -> dict:
        return {
            "built": self.built,
            "inventory_systems": self.inventory_systems,
            "verified": self.verified,
            "problems": self.problems,
        }


def P(text: str, style=BODY) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), style)


def table(rows: list[list[str]], widths: list[float], header: bool = True) -> Table:
    data = [[P(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde4ee"))]
    t.setStyle(TableStyle(style))
    return t


def banner_factory(company: dict, doc_id: str, title: str):
    def _draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(colors.HexColor("#8b0000"))
        canvas.drawCentredString(A4[0] / 2, A4[1] - 1.2 * cm, company["classification_banner"])
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(2 * cm, 1.2 * cm, f"{doc_id} · {title}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    return _draw


def make_doc(path: Path, title: str, subject: str, company: dict, keywords: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2 * cm,
        title=title,
        subject=subject,
        author=company["name"],
        keywords=keywords,
        creator="Rampart corpus builder",
    )


# ── CVE verification ──────────────────────────────────────────────────────────
def _vt(v: str) -> tuple:
    """Comparable version tuple for forms like 14.1-73.37, R35, 2.568.1, 1402.v94c9ce464861:
    numeric runs compare as numbers, alphabetic runs as lower-case strings."""
    return tuple((1, int(t)) if t.isdigit() else (0, t.lower()) for t in re.findall(r"\d+|[A-Za-z]+", v))


def version_in_range(version: str, rng: dict) -> bool | None:
    """True/False when a comparable range exists, None when the CPE carries no range."""
    v = _vt(version)
    lo, hi_excl, hi_incl = rng.get("start_incl"), rng.get("end_excl"), rng.get("end_incl")
    if not (lo or hi_excl or hi_incl):
        return None
    if lo and v < _vt(lo):
        return False
    if hi_excl and not v < _vt(hi_excl):
        return False
    if hi_incl and v > _vt(hi_incl):
        return False
    return True


def _product_tokens(product: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", product.lower()) if len(t) > 2}


async def verify(specs: list[dict], settings: Settings) -> tuple[dict, list[str]]:
    http = HttpClient(mode=settings.http_mode, cassette_dir=settings.cassette_dir, cache_dir=settings.cache_dir)
    nvd.configure(http, settings.nvd_api_key.get_secret_value() if settings.nvd_api_key else None)
    records, problems = {}, []
    refs: list[tuple[str, str, str, str]] = []  # (cve, system, product, version)
    for spec in specs:
        for comp in spec.get("components", []):
            for cve_id in comp.get("cves", []):
                refs.append((cve_id, spec["name"], comp["product"], str(comp["version"])))
    for cve_id in sorted({r[0] for r in refs}):
        try:
            rec = await nvd.fetch_cve(http, cve_id)
            k = await kev.lookup(http, cve_id)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{cve_id}: {e}")
            continue
        rec["kev"] = k
        pub = (rec.get("published") or "")[:10]
        if not (WINDOW[0] <= pub <= WINDOW[1]):
            problems.append(f"{cve_id}: published {pub}, outside the window {WINDOW}")
        records[cve_id] = rec
    for cve_id, system, product, version in refs:
        rec = records.get(cve_id)
        if not rec:
            continue
        toks = _product_tokens(product)
        matching = [
            a
            for a in rec.get("affected_products", [])
            if toks & _product_tokens(a["cpe"].split(":")[3] + " " + a["cpe"].split(":")[4])
        ]
        verdicts = [version_in_range(version, a) for a in matching]
        if matching and not any(v for v in verdicts if v) and any(v is False for v in verdicts):
            problems.append(
                f"{cve_id}: {product} {version} on {system} is not inside any NVD vulnerable range "
                f"({[(a['cpe'].split(':')[4], a.get('start_incl'), a.get('end_excl') or a.get('end_incl')) for a in matching][:4]})"
            )
        desc = (rec.get("description") or "").lower()
        if not matching and not (toks & set(re.split(r"[^a-z0-9]+", desc))):
            problems.append(f"{cve_id}: neither CPE data nor the description names {product}")
        rec.setdefault("evidence", {})[(system, product, version)] = (
            "CPE range"
            if matching and any(verdicts)
            else ("CPE listed, no range" if matching else "description names the product")
        )
    await http.aclose()
    return records, problems


# ── PDF builders ──────────────────────────────────────────────────────────────
def build_system_pdf(out: Path, company: dict, spec: dict) -> Path:
    path = out / f"SRA-{spec['id']}.pdf"
    doc = make_doc(path, spec["title"], spec["id"], company, f"system risk assessment, {spec['name']}, SBOM, exposure")
    s = []
    s += [
        P(spec["title"], TITLE),
        P(f"{company['name']} · System Risk Assessment · {spec['doc_id']} v{spec['version']}", SUB),
        Spacer(1, 10),
    ]
    s += [
        table(
            [
                ["Field", "Value"],
                ["Document id", spec["doc_id"]],
                ["Version", str(spec["version"])],
                ["Classification", "INTERNAL — CONFIDENTIAL"],
                ["System", spec["name"]],
                ["Owner team", spec["owner_team"]],
                ["Assessment date", str(spec["assessment_date"])],
                ["Next review", str(spec["next_review"])],
                ["Assessor", spec["assessor"]],
                ["Exposure", spec["exposure"]],
                ["Data classification", spec["data_classification"]],
            ],
            [4.5, 12.5],
        )
    ]
    s += [P("1. Purpose and scope", H1), P(spec["purpose"])]
    s += [P("2. System overview", H1)]
    for para in spec["overview"]:
        s.append(P(para))
    s += [P("3. Architecture and hosts", H1)]
    for para in spec["architecture"]:
        s.append(P(para))
    s += [
        P("3.1 Host inventory", H2),
        table(
            [["Hostname", "IP address", "Zone", "Role", "Operating system"]]
            + [[h["hostname"], h["ip"], h["zone"], h["role"], h["os"]] for h in spec["hosts"]],
            [5.2, 3.0, 2.3, 3.7, 2.8],
        ),
    ]
    s += [
        P("4. Software bill of materials", H1),
        P(
            spec.get(
                "sbom_intro",
                "The components below were confirmed from package "
                "manifests and running processes during the assessment. Versions are as deployed on the assessment date.",
            )
        ),
    ]
    s += [
        table(
            [["Component", "Vendor / project", "Version", "Role in the system", "Deployed on"]]
            + [
                [c["product"], c["vendor"], str(c["version"]), c["role"], c.get("deployed_on", "")]
                for c in spec["components"]
            ],
            [3.6, 3.2, 2.0, 5.2, 3.0],
        )
    ]
    s += [P("4.1 Component notes", H2)]
    for c in spec["components"]:
        s.append(KeepTogether([P(f"{c['product']} {c['version']}", H2), P(c["notes"])]))
    s += [P("5. Network exposure", H1)]
    for para in spec["exposure_notes"]:
        s.append(P(para))
    s += [
        P("5.1 Exposed endpoints", H2),
        table(
            [["Port / protocol", "Service", "Reachable from", "Authentication"]]
            + [[e["port"], e["service"], e["reachable_from"], e["auth"]] for e in spec["endpoints"]],
            [3.2, 4.6, 4.6, 4.6],
        ),
    ]
    s += [P("6. Security controls", H1)]
    s += [
        table(
            [["Control", "Status", "Last verified", "Notes"]]
            + [[c["control"], c["status"], str(c["verified"]), c["notes"]] for c in spec["controls"]],
            [4.6, 2.2, 2.4, 7.8],
        )
    ]
    s += [P("7. Open findings", H1)]
    s += [
        table(
            [["Id", "Severity", "Finding", "Owner", "Due"]]
            + [[f["id"], f["severity"], f["text"], f["owner"], str(f["due"])] for f in spec["findings"]],
            [2.0, 1.8, 8.4, 2.6, 2.2],
        )
    ]
    s += [P("8. Ownership and contacts", H1)]
    s += [
        table(
            [["Role", "Name", "Contact"]] + [[c["role"], c["name"], c["contact"]] for c in spec["contacts"]],
            [4.5, 5.5, 7.0],
        )
    ]
    for para in spec.get("ownership_notes", []):
        s.append(P(para))
    s += [
        P("9. Assessment history", H1),
        table(
            [["Date", "Version", "Summary"]]
            + [[str(h["date"]), str(h["version"]), h["summary"]] for h in spec["history"]],
            [2.6, 1.8, 12.6],
        ),
    ]
    s += [
        PageBreak(),
        P("Appendix A. Service and port matrix", H1),
        table(
            [["Host", "Port", "Service", "Listens on", "Purpose"]]
            + [[r["host"], r["port"], r["service"], r["listens"], r["purpose"]] for r in spec["service_matrix"]],
            [4.6, 1.8, 3.4, 3.0, 4.2],
        ),
    ]
    s += [P("Appendix B. Backup, recovery and change management", H1)]
    for para in spec["operations"]:
        s.append(P(para))
    s += [
        P("Appendix C. Change log", H1),
        table(
            [["Date", "Change", "Approved by"]]
            + [[str(c["date"]), c["change"], c["approved_by"]] for c in spec["changelog"]],
            [2.6, 10.4, 4.0],
        ),
    ]
    draw = banner_factory(company, spec["doc_id"], spec["title"])
    doc.build(s, onFirstPage=draw, onLaterPages=draw)
    return path


def build_policy_pdf(out: Path, company: dict, spec: dict) -> Path:
    path = out / "POLICY-vulnerability-management.pdf"
    doc = make_doc(
        path, spec["title"], "policy", company, "vulnerability management policy, SLA, severity, KEV, risk acceptance"
    )
    s = [
        P(spec["title"], TITLE),
        P(f"{company['name']} · {spec['doc_id']} v{spec['version']} · effective {spec['effective']}", SUB),
        Spacer(1, 10),
    ]
    s += [
        table(
            [
                ["Field", "Value"],
                ["Document id", spec["doc_id"]],
                ["Version", spec["version"]],
                ["Effective", str(spec["effective"])],
                ["Owner", spec["owner"]],
                ["Approved by", spec["approved_by"]],
                ["Classification", "INTERNAL"],
            ],
            [4.5, 12.5],
        )
    ]
    for sec in spec["sections"]:
        s.append(P(sec["heading"], H1))
        for para in sec["text"].strip().split("\n\n"):
            s.append(P(" ".join(para.split())))
    s += [
        PageBreak(),
        P("Appendix A. Remediation deadline matrix", H1),
        table(
            [
                ["Severity", "Internet-facing", "Internal", "Isolated"],
                ["Critical", "7 days", "14 days", "14 days (next window with written isolation confirmation)"],
                ["High", "14 days", "30 days", "30 days (same condition)"],
                ["Medium", "60 days", "60 days", "60 days"],
                ["Low", "180 days or next window", "180 days or next window", "180 days or next window"],
                [
                    "Any severity, KEV-listed",
                    "CISA due date if earlier; ticket within 1 business day",
                    "CISA due date if earlier; ticket within 1 business day",
                    "CISA due date if earlier",
                ],
            ],
            [3.6, 4.4, 4.4, 4.6],
        ),
    ]
    s += [
        P("Appendix B. Ticket priority mapping", H1),
        table(
            [
                ["Severity", "Ticket priority", "Typical first action"],
                ["Critical", "P1", "Emergency change; Owner Team paged; status update daily"],
                ["High", "P2", "Scheduled within the deadline; CAB notified"],
                ["Medium", "P3", "Next maintenance cycle inside 60 days"],
                ["Low", "P4", "Backlog; bundled with routine upgrades"],
            ],
            [3.6, 3.4, 10.0],
        ),
    ]
    s += [
        P("Appendix C. Required ticket fields", H1),
        P(
            "CVE identifier; system name as in the SRA; affected component and deployed version; evidence used at triage "
            "(CVSS base score and source, EPSS probability and date, KEV status and CISA due date, exposure and data "
            "classification from the SRA, compensating controls); severity and resulting deadline under section 5; fixed "
            "version or vendor mitigation when known; Owner Team; approving analyst."
        ),
    ]
    draw = banner_factory(company, spec["doc_id"], spec["title"])
    doc.build(s, onFirstPage=draw, onLaterPages=draw)
    return path


def write_sources(out: Path, company: dict, specs: list[dict], records: dict, problems: list[str]) -> None:
    lines = [
        "# Corpus sources\n",
        f"Built {date.today().isoformat()}. Company **{company['name']}** and every host, owner and contact are fictitious. "
        f"Software, versions and CVEs are real; each CVE below was fetched from NVD at build time and checked against the "
        f"window {WINDOW[0]} to {WINDOW[1]}.\n",
        "| System | Component | Version | CVE | Published | CVSS (source) | NVD status | KEV | Evidence |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for spec in specs:
        for comp in spec.get("components", []):
            for cve_id in comp.get("cves", []):
                r = records.get(cve_id)
                if not r:
                    lines.append(
                        f"| {spec['name']} | {comp['product']} | {comp['version']} | {cve_id} | ? | ? | not fetched | ? | ? |"
                    )
                    continue
                c = r.get("cvss") or {}
                ev = r.get("evidence", {}).get((spec["name"], comp["product"], str(comp["version"])), "")
                kev_s = r["kev"].get("date_added", "yes") if r["kev"].get("in_kev") else "no"
                lines.append(
                    f"| {spec['name']} | {comp['product']} | {comp['version']} | {cve_id} | {(r.get('published') or '')[:10]} | "
                    f"{c.get('base_score')} {c.get('severity') or ''} ({(c.get('source') or '?').split('@')[-1]}) | {r.get('nvd_status')} | {kev_s} | {ev} |"
                )
    lines += ["", "## Descriptions (from NVD)", ""]
    for cve_id, r in sorted(records.items()):
        lines.append(f"- **{cve_id}** — {(r.get('description') or '').strip()[:400]}")
    if problems:
        lines += ["", "## Verification problems", ""] + [f"- {p}" for p in problems]
    (out / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inventory(out: Path, systems: list[dict]) -> int:
    inventory = {
        sp["id"]: {
            "name": sp["name"],
            "doc_id": sp["doc_id"],
            "exposure": sp["exposure"],
            "data_classification": sp["data_classification"],
            "owner_team": sp["owner_team"],
            "assessment_date": str(sp["assessment_date"]),
            "components": [{"product": c["product"], "version": str(c["version"])} for c in sp["components"]],
        }
        for sp in systems
    }
    (out / "inventory.json").write_text(json.dumps(inventory, indent=1), encoding="utf-8")
    return len(inventory)


def load_specs(specs_dir: Path) -> tuple[dict, dict, list[dict]]:
    company = yaml.safe_load((specs_dir / "company.yaml").read_text())
    policy = yaml.safe_load((specs_dir / "policy.yaml").read_text())
    systems = [yaml.safe_load(p.read_text()) for p in sorted((specs_dir / "systems").glob("*.yaml"))]
    return company, policy, systems


def build_corpus(settings: Settings, *, skip_verify: bool = False, only: str | None = None) -> BuildReport:
    """Build the PDFs (and, unless `only` is set, SOURCES.md and inventory.json) under settings.corpus_dir."""
    out = settings.corpus_dir
    company, policy, systems = load_specs(out / "specs")
    out.mkdir(parents=True, exist_ok=True)
    report = BuildReport()
    records: dict = {}
    if not skip_verify and systems:
        records, report.problems = asyncio.run(verify(systems, settings))
        report.verified = True
    built: list[Path] = []
    if only in (None, "policy"):
        built.append(build_policy_pdf(out, company, policy))
    for spec in systems:
        if only in (None, spec["id"]):
            built.append(build_system_pdf(out, company, spec))
    if systems and report.verified:
        write_sources(out, company, systems, records, report.problems)
    if systems:
        report.inventory_systems = write_inventory(out, systems)
    for p in built:
        report.built[p.name] = len(PdfReader(str(p)).pages)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-verify", action="store_true", help="do not fetch the CVEs from NVD")
    ap.add_argument("--only", default=None, help="policy or a system id, to rebuild one document")
    args = ap.parse_args(argv)
    report = build_corpus(get_settings(), skip_verify=args.skip_verify, only=args.only)
    for name, pages in report.built.items():
        print(f"built {name}: {pages} pages")
    if report.inventory_systems:
        print(f"wrote inventory.json for {report.inventory_systems} systems")
    if report.problems:
        print("\nVERIFICATION PROBLEMS:")
        for p in report.problems:
            print(" -", p)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
