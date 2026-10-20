from rampart.settings import Settings


def test_paths_resolve_against_home(tmp_path, monkeypatch):
    monkeypatch.delenv("RAMPART_VAR_DIR", raising=False)
    monkeypatch.delenv("RAMPART_QDRANT_PATH", raising=False)
    s = Settings(home=tmp_path, openai_api_key="sk-x")
    assert s.var_dir == tmp_path / "var" and s.corpus_dir == tmp_path / "data" / "corpus"
    assert s.audit_db == tmp_path / "var" / "audit.sqlite3" and s.inventory_path.name == "inventory.json"


def test_auth_token_accepts_the_legacy_name(tmp_path, monkeypatch):
    monkeypatch.delenv("RAMPART_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("RAMPART_MCP_AUTH_TOKEN", "legacy")
    assert Settings(home=tmp_path, openai_api_key="sk-x").auth_token.get_secret_value() == "legacy"
    monkeypatch.setenv("RAMPART_AUTH_TOKEN", "current")
    assert Settings(home=tmp_path, openai_api_key="sk-x").auth_token.get_secret_value() == "current"
