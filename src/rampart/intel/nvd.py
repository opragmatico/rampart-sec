"""NVD CVE API 2.0 client. Keyless: 5 requests per rolling 30 s. With a key: 50 per 30 s."""

from __future__ import annotations

import re
from typing import Any

from ..http import HttpClient, HttpError

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$")
CACHE_TTL_S = 6 * 3600
MAX_CPES, MAX_REFS = 40, 15


def normalise_cve_id(cve_id: str) -> str:
    cid = cve_id.strip().upper()
    if not CVE_RE.match(cid):
        raise ValueError(f"'{cve_id}' is not a CVE id (expected CVE-YYYY-NNNN).")
    return cid


def configure(http: HttpClient, api_key: str | None) -> None:
    http.limiter.configure("nvd", 50 if api_key else 5, 30.0)


async def fetch_cve(http: HttpClient, cve_id: str, api_key: str | None = None) -> dict[str, Any]:
    cid = normalise_cve_id(cve_id)
    headers = {"apiKey": api_key} if api_key else None
    res = await http.get_json(
        NVD_URL, params={"cveId": cid}, headers=headers, limiter_key="nvd", cache_ttl=CACHE_TTL_S, timeout=40, retries=4
    )
    items = (res.body or {}).get("vulnerabilities", []) if isinstance(res.body, dict) else []
    if not items:
        raise HttpError(f"{cid} is not in NVD (no record returned).", status=404, code="NOT_FOUND")
    return _normalise(items[0]["cve"]) | {"fetched_at": res.fetched_at, "cached": res.cached or res.replayed}


def _normalise(c: dict) -> dict[str, Any]:
    metrics = c.get("metrics", {}) or {}
    cvss: dict | None = None
    # Prefer NVD's own score; fall back to the CNA's. Both are labelled so the agent can tell.
    for key, version in (("cvssMetricV40", "4.0"), ("cvssMetricV31", "3.1"), ("cvssMetricV30", "3.0")):
        entries = metrics.get(key) or []
        if not entries:
            continue
        primary = next((m for m in entries if m.get("type") == "Primary"), entries[0])
        d = primary["cvssData"]
        cvss = {
            "version": version,
            "base_score": d.get("baseScore"),
            "severity": d.get("baseSeverity"),
            "vector": d.get("vectorString"),
            "source": primary.get("source"),
            "type": primary.get("type"),
        }
        break
    cwes = sorted(
        {
            desc["value"]
            for w in c.get("weaknesses", [])
            for desc in w.get("description", [])
            if desc.get("lang") == "en" and desc["value"].startswith("CWE-")
        }
    )
    cpes: list[dict] = []
    for conf in c.get("configurations", []) or []:
        for node in conf.get("nodes", []):
            for m in node.get("cpeMatch", []):
                if m.get("vulnerable"):
                    cpes.append(
                        {
                            "cpe": m["criteria"],
                            "start_incl": m.get("versionStartIncluding"),
                            "end_excl": m.get("versionEndExcluding"),
                            "end_incl": m.get("versionEndIncluding"),
                        }
                    )
    refs = [{"url": r["url"], "tags": r.get("tags", [])} for r in c.get("references", [])][:MAX_REFS]
    desc = next((d["value"] for d in c.get("descriptions", []) if d.get("lang") == "en"), "")
    return {
        "cve_id": c["id"],
        "published": c.get("published"),
        "last_modified": c.get("lastModified"),
        "nvd_status": c.get("vulnStatus"),  # e.g. "Analyzed", "Awaiting Analysis" (the backlog is real)
        "description": desc,
        "cvss": cvss,
        "cwes": cwes,
        "affected_products": cpes[:MAX_CPES],
        "affected_products_truncated": len(cpes) > MAX_CPES,
        "references": refs,
        "cisa_exploit_add": c.get("cisaExploitAdd"),
        "cisa_action_due": c.get("cisaActionDue"),
    }
