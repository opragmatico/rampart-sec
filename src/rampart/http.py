"""One HTTP client for every external API, with the behaviour a production tool needs:

- retries with exponential backoff and jitter on 429, 5xx and transport errors;
- a per-service rate limiter (NVD allows 5 requests per rolling 30 s without a key);
- a TTL cache, in memory and optionally on disk for large feeds;
- record / replay of responses as "cassettes", so the test-suite runs without network or keys.

Secrets never enter a cassette or a cache key: headers are excluded from both.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .logging import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class HttpError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False, code: str = "HTTP_ERROR"):
        super().__init__(message)
        self.status, self.retryable, self.code = status, retryable, code


class CassetteMissing(HttpError):
    def __init__(self, key: str, descriptor: dict):
        super().__init__(
            f"No cassette {key} for {descriptor['method']} {descriptor['url']} "
            f"(params={descriptor.get('params')}). Run once with RAMPART_HTTP_MODE=record.",
            code="CASSETTE_MISSING",
        )


@dataclass
class HttpResult:
    status: int
    url: str
    body: Any  # parsed JSON, or text when the body is not JSON
    fetched_at: str
    cached: bool = False
    replayed: bool = False
    attempts: int = 1
    elapsed_ms: int = 0


class RateLimiter:
    """Sliding-window limiter: at most `max_calls` per `period_s`, per key."""

    def __init__(self) -> None:
        self._windows: dict[str, deque[float]] = {}
        self._limits: dict[str, tuple[int, float]] = {}
        self._lock = asyncio.Lock()

    def configure(self, key: str, max_calls: int, period_s: float) -> None:
        self._limits[key] = (max_calls, period_s)

    async def acquire(self, key: str) -> float:
        if key not in self._limits:
            return 0.0
        max_calls, period = self._limits[key]
        waited = 0.0
        while True:
            async with self._lock:
                q = self._windows.setdefault(key, deque())
                now = time.monotonic()
                while q and now - q[0] > period:
                    q.popleft()
                if len(q) < max_calls:
                    q.append(now)
                    return waited
                sleep_for = period - (now - q[0]) + 0.05
            await asyncio.sleep(sleep_for)
            waited += sleep_for


class TtlCache:
    def __init__(self, disk_dir: Path | None = None) -> None:
        self._mem: dict[str, tuple[float, Any]] = {}
        self.disk_dir = disk_dir

    def get(self, key: str) -> Any | None:
        hit = self._mem.get(key)
        if hit and hit[0] > time.time():
            return hit[1]
        if self.disk_dir is not None:
            p = self.disk_dir / f"{key}.json"
            if p.exists():
                try:
                    blob = json.loads(p.read_text())
                    if blob["expires"] > time.time():
                        self._mem[key] = (blob["expires"], blob["value"])
                        return blob["value"]
                except (ValueError, KeyError):
                    pass
        return None

    def set(self, key: str, value: Any, ttl_s: float, on_disk: bool = False) -> None:
        expires = time.time() + ttl_s
        self._mem[key] = (expires, value)
        if on_disk and self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            (self.disk_dir / f"{key}.json").write_text(json.dumps({"expires": expires, "value": value}))


def _request_key(method: str, url: str, params: dict | None, body: Any) -> tuple[str, dict]:
    descriptor = {"method": method.upper(), "url": url, "params": dict(sorted((params or {}).items())), "body": body}
    raw = json.dumps(descriptor, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:20], descriptor


class HttpClient:
    """Shared by all intel clients. `mode` is live | record | replay."""

    def __init__(
        self,
        *,
        mode: str = "live",
        cassette_dir: Path | None = None,
        cache_dir: Path | None = None,
        user_agent: str = "rampart/0.1 (+vulnerability triage agent)",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.mode = mode
        self.cassette_dir = cassette_dir
        self.limiter = RateLimiter()
        self.cache = TtlCache(cache_dir)
        self._client = httpx.AsyncClient(headers={"User-Agent": user_agent}, follow_redirects=True, transport=transport)
        self.stats = {"requests": 0, "cache_hits": 0, "replayed": 0, "retries": 0, "rate_limit_wait_s": 0.0}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json_body: Any = None,
        headers: dict | None = None,
        limiter_key: str | None = None,
        cache_ttl: float = 0,
        cache_on_disk: bool = False,
        timeout: float = 30.0,
        retries: int = 4,
        retry_on_404: bool = False,
    ) -> HttpResult:
        key, descriptor = _request_key(method, url, params, json_body)
        now_iso = datetime.now(UTC).isoformat(timespec="seconds")

        if cache_ttl:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats["cache_hits"] += 1
                return HttpResult(status=200, url=url, body=hit, fetched_at=now_iso, cached=True)

        if self.mode == "replay":
            tape = self._load_cassette(key)
            if tape is None:
                raise CassetteMissing(key, descriptor)
            self.stats["replayed"] += 1
            res = HttpResult(
                status=tape["status"], url=url, body=tape["body"], fetched_at=tape["fetched_at"], replayed=True
            )
            self._raise_for_status(res)
            if cache_ttl:
                self.cache.set(key, res.body, cache_ttl, on_disk=cache_on_disk)
            return res

        if limiter_key:
            self.stats["rate_limit_wait_s"] += await self.limiter.acquire(limiter_key)

        t0 = time.monotonic()
        last_exc: Exception | None = None
        resp: httpx.Response | None = None
        for attempt in range(1, retries + 1):
            try:
                self.stats["requests"] += 1
                resp = await self._client.request(
                    method, url, params=params, json=json_body, headers=headers, timeout=timeout
                )
                if resp.status_code in RETRYABLE_STATUS or (retry_on_404 and resp.status_code == 404):
                    last_exc = HttpError(f"{resp.status_code} from {url}", status=resp.status_code, retryable=True)
                else:
                    last_exc = None
                    break
            except httpx.TransportError as e:
                last_exc = HttpError(f"transport error for {url}: {e}", retryable=True, code="TRANSPORT")
            if attempt < retries:
                delay = min(20.0, (1.5**attempt) + random.uniform(0, 0.5))
                retry_after = resp.headers.get("Retry-After") if resp is not None else None
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
                self.stats["retries"] += 1
                log.warning("http.retry", url=url, attempt=attempt, delay_s=round(delay, 1), reason=str(last_exc))
                await asyncio.sleep(delay)
        if last_exc is not None and (resp is None or resp.status_code in RETRYABLE_STATUS):
            raise last_exc
        assert resp is not None

        body: Any
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        res = HttpResult(
            status=resp.status_code,
            url=str(resp.url),
            body=body,
            fetched_at=now_iso,
            attempts=attempt,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
        )
        if self.mode == "record":
            self._save_cassette(key, descriptor, res)
        self._raise_for_status(res, resp)
        if cache_ttl:
            self.cache.set(key, res.body, cache_ttl, on_disk=cache_on_disk)
        return res

    async def get_json(self, url: str, **kw) -> HttpResult:
        return await self.request("GET", url, **kw)

    async def post_json(self, url: str, json_body: Any, **kw) -> HttpResult:
        return await self.request("POST", url, json_body=json_body, **kw)

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _raise_for_status(res: HttpResult, resp: httpx.Response | None = None) -> None:
        if res.status >= 400:
            detail = resp.headers.get("message", "") if resp is not None else ""
            code = {401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT_FOUND", 429: "RATE_LIMITED"}.get(
                res.status, "HTTP_ERROR"
            )
            raise HttpError(
                f"HTTP {res.status} from {res.url}. {detail}".strip(),
                status=res.status,
                retryable=res.status in RETRYABLE_STATUS,
                code=code,
            )

    def _load_cassette(self, key: str) -> dict | None:
        if self.cassette_dir is None:
            return None
        p = self.cassette_dir / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _save_cassette(self, key: str, descriptor: dict, res: HttpResult) -> None:
        if self.cassette_dir is None:
            return
        self.cassette_dir.mkdir(parents=True, exist_ok=True)
        (self.cassette_dir / f"{key}.json").write_text(
            json.dumps(
                {"request": descriptor, "status": res.status, "fetched_at": res.fetched_at, "body": res.body},
                indent=1,
                default=str,
            )
        )
