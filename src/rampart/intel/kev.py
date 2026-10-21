"""CISA Known Exploited Vulnerabilities catalog: one JSON feed, cached on disk for 12 hours."""

from __future__ import annotations

from typing import Any

from ..http import HttpClient
from .nvd import normalise_cve_id

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
CACHE_TTL_S = 12 * 3600


async def load_catalog(http: HttpClient) -> dict[str, Any]:
    res = await http.get_json(KEV_URL, cache_ttl=CACHE_TTL_S, cache_on_disk=True, timeout=60)
    body = res.body
    return {
        "catalog_version": body.get("catalogVersion"),
        "date_released": body.get("dateReleased"),
        "count": body.get("count"),
        "fetched_at": res.fetched_at,
        "cached": res.cached or res.replayed,
        "entries": {v["cveID"]: v for v in body.get("vulnerabilities", [])},
    }


async def lookup(http: HttpClient, cve_id: str) -> dict[str, Any]:
    cid = normalise_cve_id(cve_id)
    cat = await load_catalog(http)
    v = cat["entries"].get(cid)
    out = {
        "cve_id": cid,
        "in_kev": v is not None,
        "catalog_version": cat["catalog_version"],
        "catalog_date": cat["date_released"],
        "fetched_at": cat["fetched_at"],
        "cached": cat["cached"],
    }
    if v:
        out |= {
            "vendor": v.get("vendorProject"),
            "product": v.get("product"),
            "name": v.get("vulnerabilityName"),
            "date_added": v.get("dateAdded"),
            "due_date": v.get("dueDate"),
            "required_action": v.get("requiredAction"),
            "known_ransomware_use": v.get("knownRansomwareCampaignUse"),
            "notes": v.get("notes"),
        }
    return out
