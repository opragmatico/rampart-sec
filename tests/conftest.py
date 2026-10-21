"""Shared fixtures. Tests never touch the network or a model: HTTP is replayed from cassettes,
state goes to a temporary directory, and a dummy key satisfies the settings validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="rampart-tests-"))
TEST_TOKEN = "test-token-not-a-secret"

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
os.environ["RAMPART_HTTP_MODE"] = "replay"
os.environ["RAMPART_HOME"] = str(ROOT)
os.environ["RAMPART_VAR_DIR"] = str(_TMP / "var")
os.environ["RAMPART_QDRANT_PATH"] = str(_TMP / "qdrant")
os.environ["RAMPART_AUTH_TOKEN"] = TEST_TOKEN
os.environ.setdefault("RAMPART_LOG_LEVEL", "WARNING")
os.environ.pop("QDRANT_URL", None)  # the suite always uses Qdrant local mode under the temp dir


@pytest.fixture(scope="session")
def settings():
    from rampart.settings import get_settings

    return get_settings()


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TEST_TOKEN}"}
