"""Tavily search over its REST API (through the shared HTTP client, so it is recorded and rate-limited)."""

from __future__ import annotations

from typing import Any

from ..http import HttpClient, HttpError

TAVILY_URL = "https://api.tavily.com/search"
MAX_CONTENT_CHARS = 1500


def configure(http: HttpClient) -> None:
    http.limiter.configure("tavily", 10, 10.0)


async def search(
    http: HttpClient, api_key: str | None, query: str, *, max_results: int = 5, include_domains: list[str] | None = None
) -> dict[str, Any]:
    if not api_key:
        raise HttpError("TAVILY_API_KEY is not configured.", code="NOT_CONFIGURED")
    body: dict[str, Any] = {
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
    }
    if include_domains:
        body["include_domains"] = include_domains
    res = await http.post_json(
        TAVILY_URL,
        body,
        headers={"Authorization": f"Bearer {api_key}"},
        limiter_key="tavily",
        cache_ttl=1800,
        timeout=45,
        retries=3,
    )
    results = []
    for r in (res.body or {}).get("results", []):
        content = r.get("content") or ""
        results.append(
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "score": r.get("score"),
                "published_date": r.get("published_date"),
                "content": content[:MAX_CONTENT_CHARS],
                "truncated": len(content) > MAX_CONTENT_CHARS,
            }
        )
    return {"query": query, "results": results, "fetched_at": res.fetched_at, "cached": res.cached or res.replayed}
