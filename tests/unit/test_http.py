import json

import httpx
import pytest

from rampart.http import CassetteMissing, HttpClient, HttpError


def _transport(responses):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        status, body = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler), calls


@pytest.mark.asyncio
async def test_retries_then_succeeds_and_caches(tmp_path):
    transport, calls = _transport([(503, {}), (200, {"ok": 1})])
    http = HttpClient(mode="live", cache_dir=tmp_path, transport=transport)
    http.limiter.configure("x", 10, 1)
    r = await http.get_json("https://example.test/a", limiter_key="x", cache_ttl=60, retries=3)
    assert r.body == {"ok": 1} and r.attempts == 2 and calls["n"] == 2
    r2 = await http.get_json("https://example.test/a", cache_ttl=60)
    assert r2.cached and calls["n"] == 2
    await http.aclose()


@pytest.mark.asyncio
async def test_record_then_replay_without_network(tmp_path):
    transport, calls = _transport([(200, {"hello": "world"})])
    rec = HttpClient(mode="record", cassette_dir=tmp_path / "c", transport=transport)
    r = await rec.get_json("https://example.test/b", params={"q": "1"})
    assert r.body == {"hello": "world"}
    assert len(list((tmp_path / "c").glob("*.json"))) == 1
    tape = json.loads(next((tmp_path / "c").glob("*.json")).read_text())
    assert "Authorization" not in json.dumps(tape)
    replay = HttpClient(
        mode="replay", cassette_dir=tmp_path / "c", transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    r2 = await replay.get_json("https://example.test/b", params={"q": "1"})
    assert r2.replayed and r2.body == {"hello": "world"}
    with pytest.raises(CassetteMissing):
        await replay.get_json("https://example.test/never-recorded")
    await rec.aclose()
    await replay.aclose()


@pytest.mark.asyncio
async def test_404_is_not_retried(tmp_path):
    transport, calls = _transport([(404, {})])
    http = HttpClient(mode="live", transport=transport)
    with pytest.raises(HttpError) as e:
        await http.get_json("https://example.test/missing", retries=3)
    assert e.value.code == "NOT_FOUND" and calls["n"] == 1
    await http.aclose()
