# Rampart — guardrail-protected vulnerability triage

A FastAPI service around an LLM agent that triages newly disclosed vulnerabilities for a company:
**does it affect us** (which systems run the affected software, from the internal risk assessments),
**how exposed are we** (internet-facing or internal, data classification, compensating controls),
**how likely is exploitation** (CVSS, EPSS, CISA KEV), **what does policy require** (deadline by
severity and exposure) and **who acts, by when** (a ticket to the owner team, which an operator approves).

A triage agent has three failure modes that make it worse than no agent: it reads attacker-influenceable
text (CVE descriptions, advisories — indirect prompt injection), it can leak the inventory (hostnames, IPs,
owners), and it can give false assurance ("not affected" stated as certain). Rampart is the agent plus the
guardrails for those failure modes, with the audit trail a security team needs to trust it.

| Layer | What it is |
|---|---|
| **Tools** | 7 tools as plain async functions: NVD, EPSS + KEV, advisories (Tavily), an urgency scorer, the internal documents (RAG), the system inventory, tickets. The agent calls them in-process as LangChain tools; the same functions are served over REST at `/v1/tools` and over MCP at `/mcp` for other agents. One response envelope, validated inputs, retries, rate limits, caching, record/replay cassettes. |
| **Private knowledge** | Qdrant (local mode or server) + `text-embedding-3-small`; idempotent incremental ingestion; five system risk assessments and one policy, filterable by system. |
| **Agent** | LangGraph: input guard → agent ⇄ tools → tool guard → output guard, with a structured `final_answer` tool, budgets, SQLite checkpoints and an **approval interrupt** before the only side-effecting tool (`open_ticket`). |
| **Guardrails** | Input: BM25 signature guard + LLM classifier. Tool results: BM25 scan that removes injected sentences. Output: redaction, softening, citation-evidence and verbatim-overlap checks, disclaimer, LLM judge. Every guard fails closed and logs its scores. |
| **Policy as code** | The policy document's severity, deadline and priority tables are a pure function (`rampart.policy`), so a ticket's priority never depends on how the model read the document. |
| **Operations** | REST API with bearer auth, `/healthz` `/readyz`, SQLite audit log, structlog JSON logs, CLI, Docker Compose with Qdrant, CI. |

The company in the documents, **Larkspur Logistics**, is fictitious (RFC 5737 addresses, reserved example
domains). The software, versions, CVEs and the NVD, EPSS, KEV and advisory data are real; every CVE in the
corpus was verified against NVD when the corpus was built ([data/corpus/SOURCES.md](data/corpus/SOURCES.md)).
Design: [docs/DESIGN.md](docs/DESIGN.md). Guardrail flows: [docs/GUARDRAIL_FLOWS.md](docs/GUARDRAIL_FLOWS.md). Measured results: [docs/EVALUATION.md](docs/EVALUATION.md).

## Quick start

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env            # fill in OPENAI_API_KEY and RAMPART_AUTH_TOKEN (TAVILY_API_KEY for advisory search)
.venv/bin/rampart index         # embeds the six PDFs into Qdrant local mode under var/qdrant (~5 s of API time)
.venv/bin/rampart serve         # http://127.0.0.1:8765 — interactive docs at /docs
```

Then, with `TOKEN` set to your `RAMPART_AUTH_TOKEN`:

```bash
curl -s localhost:8765/readyz
curl -s -X POST localhost:8765/v1/triage -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"query": "Triage CVE-2026-85706 for our environment: are we affected, how urgent is it, what does policy require?"}' | jq .
```

A triage takes 20–40 seconds and under a cent. The response carries the delivered answer (or why it was
blocked or withheld), the tools that ran, the guardrail log with every score, the execution path and the cost.

When the question asks for action ("…and open the tickets our policy requires"), the run **pauses** with the
proposed tickets, and the operator decides:

```bash
curl -s -X POST localhost:8765/v1/triage/ops-42/resume -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"approve_all": true}' | jq .answer
curl -s localhost:8765/v1/tickets -H "Authorization: Bearer $TOKEN" | jq .
```

The same flow from the terminal: `rampart triage "..."` prompts for the approval.

## The API

Everything under `/v1` and the MCP endpoint need `Authorization: Bearer $RAMPART_AUTH_TOKEN`; the health
endpoints and `/docs` are open.

| Method and path | What it does |
|---|---|
| `GET /healthz`, `GET /readyz` | Liveness; readiness is 503 until the corpus is indexed and the agent runner is open |
| `POST /v1/triage` | Run a question through the guarded agent (`{"query", "thread_id"?}`) |
| `GET /v1/triage/{thread}/approval` | The tickets a paused run is waiting on |
| `POST /v1/triage/{thread}/resume` | Approve or decline them (`{"approve_all": bool}` or `{"approved": [bool, …]}`) |
| `GET /v1/runs`, `GET /v1/runs/{run_id}`, `GET /v1/runs/summary` | The audit log: per run, with the guardrail log; programme-level numbers (blocked by which layer, withheld, cost, latency per node) |
| `GET /v1/tickets`, `GET /v1/tickets/{id}` | Remediation tickets |
| `GET /v1/tools`, `POST /v1/tools/{name}` | The seven tools, callable directly through the same functions the agent calls (`open_ticket` is refused: tickets only come from an approved run) |
| `GET /v1/corpus`, `POST /v1/corpus/index`, `POST /v1/corpus/search` | What is indexed, incremental ingestion, raw retrieval with the system filter |
| `POST /v1/guardrails/score` · `/scan` · `/output-check` | Probe the deterministic guard layers: signature score of a question, tool-guard scan and redaction of a text, the output guard's redaction, softening and citation parse |
| `POST /v1/policy/evaluate` | Severity, priority, deadline and due date under the policy for given evidence |
| `GET /v1/evals/guardrail[?llm=true]`, `GET /v1/evals/retrieval` | Precision/recall of the input guard per threshold; retriever hit rate on the gold set |
| `GET /v1/evals/scenarios`, `POST /v1/evals/scenarios` | The six validation scenarios; run them (real model calls) for a pass rate |
| `POST /mcp` | The tools over MCP (streamable HTTP), for agents outside Rampart such as Claude Desktop or an IDE |

The OpenAPI document is at `/openapi.json`.

## The command line

```
rampart serve                     run the service
rampart index [--force]           ingest the corpus PDFs into Qdrant (incremental, idempotent)
rampart triage "question"         one run, with the approval prompt
rampart health                    check the tools and the index
rampart eval guardrail [--llm]    precision/recall of the signature guard (and the combined guard)
rampart eval retrieval            hit rate of the retriever on the gold set
rampart eval scenarios [--runs N] the validation suite against the real APIs; exit code 1 on a failure
rampart audit [--last N]          audit summary and recent runs
rampart tickets                   remediation tickets
rampart build-corpus              regenerate the PDFs, SOURCES.md and inventory.json from data/corpus/specs
```

## Docker

```bash
docker compose up --build         # Qdrant + Rampart; the container indexes the corpus, then serves on :8765
curl -s localhost:8765/readyz
```

Agents outside Rampart can use the tools at `http://localhost:8765/mcp` with the bearer token.

## Tests and evaluation

```bash
make test        # unit + recorded tests: no network, no tokens (HTTP replayed from tests/cassettes)
make lint        # ruff
make test-live   # the six scenarios end to end (needs keys, ~3 min, a few cents)
rampart eval guardrail && rampart eval retrieval
```

- `tests/unit` — the guards without a model (BM25 matcher, redaction, softening, overlap), the policy table,
  tool contracts, tickets, the audit store, the scenario pass conditions, and the REST API over a scripted runner.
- `tests/recorded` — the seven tools as the agent calls them and through an MCP client session, and the service
  with the real runner, with HTTP responses replayed from cassettes.
- `tests/live` — `rampart.evals.scenarios` run end to end (marker `live`).

The validation scenarios are defined once ([src/rampart/evals/scenarios.py](src/rampart/evals/scenarios.py))
and run three ways: `pytest -m live`, `rampart eval scenarios --runs N`, `POST /v1/evals/scenarios`.

## Configuration

Settings come from the environment or `.env` in the project root and are validated once at start
(`rampart.settings`). The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | required | model and embeddings provider (`OPENAI_BASE_URL` for a compatible proxy) |
| `RAMPART_AUTH_TOKEN` | required by the service | bearer token for `/v1` and `/mcp` |
| `TAVILY_API_KEY`, `NVD_API_KEY` | unset | advisory search; a higher NVD rate limit |
| `QDRANT_URL` | unset (local mode under `var/qdrant`) | a Qdrant server |
| `RAMPART_AGENT_MODEL` … `RAMPART_JUDGE_MODEL` | `gpt-4o-mini` | one model per role; the judge can be stronger |
| `RAMPART_INJECTION_THRESHOLD`, `RAMPART_TOOL_GUARD_THRESHOLD`, `RAMPART_OVERLAP_THRESHOLD` | 0.70, 0.70, 0.35 | guard thresholds, calibrated with the evaluations |
| `RAMPART_MAX_CONCURRENT_RUNS` | 4 | agent runs the API serves at once |
| `RAMPART_TICKET_BACKEND` | `sqlite` | `github` with `RAMPART_GITHUB_REPO` and `GITHUB_TOKEN` opens GitHub issues too |
| `RAMPART_HTTP_MODE` | `live` | `record` writes cassettes, `replay` serves them (the test suite) |

See [.env.example](.env.example) for the full list.

## Layout

```
src/rampart/
  api/          FastAPI: app factory, auth, schemas, routers (triage, runs, tickets, tools, corpus, guardrails, policy, evals)
  agent/        LangGraph graph, state (the structured final answer), runner, the versioned system prompt
  guardrails/   BM25 signatures, LLM classifier and judge, redaction, softening, overlap, the three guard nodes
  tools/        the seven tool functions (core), the agent's LangChain adapter, the MCP adapter, contracts, server-side prompts
  intel/        NVD, FIRST EPSS, CISA KEV, Tavily clients
  rag/          Qdrant store and the ingestion job
  evals/        guardrail precision/recall, retrieval hit rate, the validation scenarios
  tickets/      SQLite ticket store, GitHub Issues adapter
  corpus/       the corpus builder (PDFs from YAML specs, verified against NVD)
  policy.py     the policy's decision table as code
  audit.py      the audit store
  http.py       retries, rate limits, TTL cache, record/replay
  settings.py   pydantic-settings, validated once
  cli.py
data/corpus     the six PDFs, inventory.json, SOURCES.md, specs/
data/evals      injections.jsonl, benign.jsonl, retrieval_gold.jsonl
data/signatures injection_signatures.txt
tests/          unit, recorded (+ cassettes), live
docker/         Dockerfile, entrypoint; docker-compose.yml at the root
docs/           DESIGN.md, EVALUATION.md, GUARDRAIL_FLOWS.md
```

## Provenance

Rampart started as a variation of a course project (JHU, agentic AI: a guarded research agent with MCP
tools, RAG and a validation suite) rebuilt around a security problem and delivered as a notebook over a
package. This repository is the production shape of that work: the notebook's demonstrations became API
endpoints, CLI commands and tests, and the scenarios it ran by hand are code.

## License

MIT — see [LICENSE](LICENSE).
