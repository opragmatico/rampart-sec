"""Bounded retry for transient model-API failures.

The OpenAI client retries connection errors and 429s on its own, but not a 401. The course proxy in
front of the API is known to return a spurious 401 ("You didn't provide an API key") now and then,
so authentication errors are retried a few times with backoff before they are treated as real.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

import openai

from .logging import get_logger

log = get_logger(__name__)
TRANSIENT = (
    openai.AuthenticationError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APITimeoutError,
)


async def aretry[T](fn: Callable[[], Awaitable[T]], *, what: str, attempts: int = 4) -> T:
    for i in range(1, attempts + 1):
        try:
            return await fn()
        except TRANSIENT as e:
            if i == attempts:
                raise
            delay = min(10.0, 0.8 * (2 ** (i - 1)) + random.uniform(0, 0.4))
            log.warning(
                "llm.retry", what=what, attempt=i, delay_s=round(delay, 1), error=f"{type(e).__name__}: {str(e)[:80]}"
            )
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")


def retry[T](fn: Callable[[], T], *, what: str, attempts: int = 4) -> T:
    for i in range(1, attempts + 1):
        try:
            return fn()
        except TRANSIENT as e:
            if i == attempts:
                raise
            delay = min(10.0, 0.8 * (2 ** (i - 1)) + random.uniform(0, 0.4))
            log.warning(
                "llm.retry", what=what, attempt=i, delay_s=round(delay, 1), error=f"{type(e).__name__}: {str(e)[:80]}"
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")
