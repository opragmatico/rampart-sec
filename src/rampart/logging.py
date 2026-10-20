"""Structured logging: console for humans, JSON lines on disk for machines.

Every event carries whatever is bound in the context (run_id, thread_id, node), so a single
run can be reconstructed from the log file with `grep run_id`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

_CONFIGURED = False


def configure_logging(fmt: str = "console", level: str = "INFO", log_file: Path | None = None) -> None:
    global _CONFIGURED
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level.upper())

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer() if fmt == "json" else structlog.dev.ConsoleRenderer(colors=False),
            ],
            foreign_pre_chain=shared,
        )
    )
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
                foreign_pre_chain=shared,
            )
        )
        root.addHandler(fh)

    # third-party noise
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "qdrant_client", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


def bind(**kwargs) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind(*keys: str) -> None:
    structlog.contextvars.unbind_contextvars(*keys)
