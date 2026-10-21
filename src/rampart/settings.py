"""Configuration: environment variables validated once at startup (fail fast).

Secrets come from the environment or a `.env` file in the project home. Nothing is read from a
JSON file in the working directory, and no setting is read lazily inside a tool.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_home(start: Path | None = None) -> Path:
    """Project home: $RAMPART_HOME, else the nearest ancestor holding pyproject.toml + src/rampart."""
    env = os.environ.get("RAMPART_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "pyproject.toml").exists() and (cand / "src" / "rampart").exists():
            return cand
    return here


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAMPART_", extra="ignore", case_sensitive=False)

    # ── location ────────────────────────────────────────────────────────────────
    home: Path = Field(default_factory=find_home)
    company: str = "Larkspur Logistics"  # the name the agent answers for

    # ── model providers ─────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    tavily_api_key: SecretStr | None = Field(default=None, validation_alias="TAVILY_API_KEY")
    nvd_api_key: SecretStr | None = Field(default=None, validation_alias="NVD_API_KEY")

    agent_model: str = "gpt-4o-mini"
    scorer_model: str = "gpt-4o-mini"  # assess_urgency and the RAG answerer
    classifier_model: str = "gpt-4o-mini"  # input guard layer 2
    judge_model: str = "gpt-4o-mini"  # output guard layer 2 (upgrade in production)
    embedding_model: str = "text-embedding-3-small"
    llm_timeout_s: float = 60.0

    # ── retrieval (Qdrant) ──────────────────────────────────────────────────────
    qdrant_url: str | None = Field(default=None, validation_alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_path: Path = Path("var/qdrant")  # local mode when qdrant_url is unset
    qdrant_collection: str = "rampart_corpus"
    corpus_dir: Path = Path("data/corpus")
    evals_dir: Path = Path("data/evals")
    retrieval_k: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── guardrails ──────────────────────────────────────────────────────────────
    signatures_path: Path = Path("data/signatures/injection_signatures.txt")
    injection_threshold: float = 0.70  # BM25 normalised score above which layer 1 fires (precision first)
    tool_guard_threshold: float = 0.70
    overlap_threshold: float = 0.35  # share of answer 10-grams found in retrieved chunks
    internal_domains: list[str] = ["corp.example", "example.com", "example.net", "example.org"]

    # ── the service (REST API + MCP endpoint) ───────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8765
    # One bearer token protects /v1 and /mcp. RAMPART_MCP_AUTH_TOKEN is accepted for compatibility.
    auth_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("RAMPART_AUTH_TOKEN", "RAMPART_MCP_AUTH_TOKEN")
    )
    mcp_url: str = "http://127.0.0.1:8765/mcp"  # where `--mode http` clients find the tools
    max_concurrent_runs: int = 4  # agent runs served at once by the API

    # ── tickets ─────────────────────────────────────────────────────────────────
    ticket_backend: Literal["sqlite", "github"] = "sqlite"
    github_repo: str | None = None  # "owner/name"
    github_token: SecretStr | None = Field(default=None, validation_alias="GITHUB_TOKEN")

    # ── HTTP, state, logs ───────────────────────────────────────────────────────
    http_mode: Literal["live", "record", "replay"] = "live"
    cassette_dir: Path = Path("tests/cassettes")
    var_dir: Path = Path("var")
    log_format: Literal["console", "json"] = "console"
    log_level: str = "INFO"
    langsmith_tracing: bool = False

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        for name in ("qdrant_path", "corpus_dir", "evals_dir", "signatures_path", "cassette_dir", "var_dir"):
            p: Path = getattr(self, name)
            if not p.is_absolute():
                object.__setattr__(self, name, (self.home / p).resolve())
        return self

    # convenience paths
    @property
    def audit_db(self) -> Path:
        return self.var_dir / "audit.sqlite3"

    @property
    def tickets_db(self) -> Path:
        return self.var_dir / "tickets.sqlite3"

    @property
    def checkpoints_db(self) -> Path:
        return self.var_dir / "checkpoints.sqlite3"

    @property
    def cache_dir(self) -> Path:
        return self.var_dir / "cache"

    @property
    def log_file(self) -> Path:
        return self.var_dir / "logs" / "rampart.jsonl"

    @property
    def inventory_path(self) -> Path:
        return self.corpus_dir / "inventory.json"

    def ensure_dirs(self) -> None:
        for d in (self.var_dir, self.cache_dir, self.log_file.parent, self.qdrant_path.parent):
            d.mkdir(parents=True, exist_ok=True)


class ConfigurationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load `.env` from the project home (without overriding real environment variables), then validate."""
    home = find_home()
    load_dotenv(home / ".env", override=False)
    try:
        s = Settings()
    except ValidationError as e:
        missing = [".".join(str(x) for x in err["loc"]) for err in e.errors() if err["type"] == "missing"]
        raise ConfigurationError(
            "Rampart is not configured. Missing: "
            + ", ".join(missing or ["(see details)"])
            + f". Set them in the environment or in {home / '.env'}.\n{e}"
        ) from None
    s.ensure_dirs()
    return s


def reset_settings() -> None:
    """For tests: forget the cached settings so a changed environment is re-read."""
    get_settings.cache_clear()
