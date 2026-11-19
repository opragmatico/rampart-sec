"""The validation suite against the real APIs and models, scored by DeepEval (marker `live`, deselected by
default).

`pytest -m live` runs the six scenarios of `rampart.evals.scenarios` through the agent with its real tools,
turns each run into a DeepEval test case and calls `assert_test` with the scenario's metrics: the
deterministic pass condition, Tool Correctness, and the judged GEval and Faithfulness metrics on
RAMPART_JUDGE_MODEL. The corpus is ingested into the temporary Qdrant path of the test session, so the suite
never touches the developer's own index. Needs OPENAI_API_KEY in `.env`; about five minutes and a few cents.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from deepeval import assert_test

from rampart.evals.scenarios import SCENARIOS, as_test_case, execute, judge_model, metrics_for

pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def live_settings():
    os.environ["RAMPART_HTTP_MODE"] = "live"
    from dotenv import load_dotenv

    from rampart.settings import find_home, get_settings, reset_settings

    load_dotenv(find_home() / ".env", override=True)  # conftest seeded a dummy key; the live suite needs the real one
    reset_settings()
    s = get_settings()
    if s.openai_api_key.get_secret_value().startswith("sk-test"):
        pytest.skip("no real OPENAI_API_KEY")
    from rampart.rag.ingest import ingest
    from rampart.tools.core import close_deps, deps

    ingest(s, store=deps().store)
    asyncio.run(close_deps())  # each scenario opens the tools afresh
    return s


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.key for s in SCENARIOS])
def test_scenario(live_settings, scenario):
    """Each scenario opens and closes the agent inside one event loop; DeepEval then measures the metrics."""
    from rampart.triage import open_triage

    async def _run():
        async with open_triage(live_settings) as rp:
            return await execute(rp, scenario)

    ctx = asyncio.run(_run())
    assert_test(as_test_case(scenario, ctx), metrics_for(scenario, ctx, judge_model(live_settings)))
