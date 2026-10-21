"""FIRST EPSS API: probability that a CVE is exploited in the next 30 days, with a 30-day series."""

from __future__ import annotations

from typing import Any

from ..http import HttpClient, HttpError
from .nvd import normalise_cve_id

EPSS_URL = "https://api.first.org/data/v1/epss"
CACHE_TTL_S = 6 * 3600


def configure(http: HttpClient) -> None:
    http.limiter.configure("epss", 20, 10.0)


async def fetch_epss(http: HttpClient, cve_id: str) -> dict[str, Any]:
    cid = normalise_cve_id(cve_id)
    res = await http.get_json(
        EPSS_URL, params={"cve": cid, "scope": "time-series"}, limiter_key="epss", cache_ttl=CACHE_TTL_S, timeout=30
    )
    data = (res.body or {}).get("data", []) if isinstance(res.body, dict) else []
    if not data:
        raise HttpError(f"EPSS has no score for {cid}.", status=404, code="NOT_FOUND")
    row = data[0]
    series = sorted(
        (
            {"date": p["date"], "epss": float(p["epss"]), "percentile": float(p["percentile"])}
            for p in row.get("time-series", [])
        ),
        key=lambda p: p["date"],
    )
    today = {"date": row["date"], "epss": float(row["epss"]), "percentile": float(row["percentile"])}
    if not series or series[-1]["date"] != today["date"]:
        series.append(today)
    first = series[0]
    return {
        "cve_id": cid,
        "epss": today["epss"],
        "percentile": today["percentile"],
        "as_of": today["date"],
        "series_days": len(series),
        "epss_30d_ago": first["epss"],
        "trend": round(today["epss"] - first["epss"], 5),
        "series": series,
        "fetched_at": res.fetched_at,
        "cached": res.cached or res.replayed,
    }
