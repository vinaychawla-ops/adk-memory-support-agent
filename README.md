# Customer Support Agent with Layered Memory

A Google ADK rebuild of the e-commerce customer support agent from
*"The AI Agent Memory Architecture Explained — From State to Long-Term Memory"*
(Applied with AI – Sanjay Kumar). Phases 0–2 of the roadmap in
[`ROADMAP.md`](ROADMAP.md) are done: one agent, four memory layers,
durable backends in local Docker. Mock data; runs locally.

## The four layers, and where each lives

| Video layer | What it holds | Phase 0–1 (zero infra) | Phase 2 (Docker) |
|---|---|---|---|
| Conversation history & session state | Structured execution state: `active_order_id`, `return_in_progress`, `memory_context` — never raw transcript | ADK `InMemorySessionService` (`Session.state`); tools write it | `RedisSessionService` — JSON sessions with sliding TTL; any replica serves any session |
| Episodic memory | What happened in past interactions (session events) | ADK `InMemoryMemoryService` via `EpisodicBank` | `PgVectorMemoryService` — one embedded row per event, vector-ranked recall (replaces the keyword search) |
| Semantic memory | Durable customer facts/preferences with upsert + supersede | `SemanticBank` (in-memory) | `FirestoreSemanticBank` — one document per user, `{fact, confidence, supersedes, expires}` |
| Procedural memory | How the agent must behave: tone, refund thresholds, escalation | System instruction + deterministic `rules.py` | `rules/policy.yaml` (validated thresholds) + `rules/procedural.md`, versioned in git — the commit history is the audit trail |

Backend selection is env-driven (`SESSION_BACKEND`, `EPISODIC_BACKEND`,
`SEMANTIC_BACKEND`, `EMBEDDER`); defaults keep everything in-memory, so
zero-infra runs still work. The agent code never changes to accommodate a
backend — ADK's service interfaces are the seam.

The recall flow: `before_agent_callback` calls `search_memory` once per
session, builds a recall block (facts + relevant episodes), and stores it
in `Session.state["memory_context"]`; the instruction provider renders it
into the system prompt. After a session, `add_session_to_memory` stores
the episode. Tools: a mock Order Management System
(`lookup_order`, `track_order`, `start_return`, `get_return_policy`,
`request_refund`, `escalate_to_human`) plus `remember_fact` /
`recall_memories`.

Policy, enforced deterministically in `rules.py` (thresholds from
`rules/policy.yaml`): refunds/credits at or below **$50** may be issued;
anything above is **drafted and escalated to a human — never
auto-executed**. Returns only within 30 days of delivery.

## Audit log (Phase 1)

`support_agent/audit.py` — every turn, tool call, memory write
(episodic + semantic), procedural rule firing, and memory recall is
logged to an `AuditLog`: in-memory events plus an optional JSONL file.
This is the durable, on-disk episodic record.

- Prompts are stored as a truncated preview + SHA-256 hash, never the
  full text. Card-number-shaped strings are redacted to `[REDACTED]`
  *before* anything is stored — they can't reach any memory layer,
  including the backends and this log.
- Wired through ADK callbacks (`before_agent_callback`,
  `before/after_tool_callback`) plus the tool call sites; with no log
  registered the callbacks are no-ops (zero overhead).
- `AuditLog.replay()` re-reads the JSONL file, proving the on-disk
  trail replays the story the agent saw.

## Eval set (Phase 1)

`tests/test_eval.py` — 12 scripted support scenarios, each asserting
expected memory behavior, all offline: preference recall across
sessions, conflicting preferences superseding (not appending),
card-number PII never entering any layer, episodic record/recall
round-trip, session state across turns, the $50.00/$50.01 refund
boundary, escalation triggers, the 30-day return window, the
late-delivery goodwill credit, recall injection into the prompt, PII
scrubbing in audit previews, and a full multi-session story whose
16-event audit trail replays in exact order from the JSONL file.

## Durable backends (Phase 2)

`docker-compose.yml` brings up Redis, PostgreSQL + pgvector, and the
Firestore emulator:

```bash
docker compose up -d
```

Then point the agent at them (any subset; unset means in-memory):

```bash
export SESSION_BACKEND=redis          # 2a: sessions in Redis
export EPISODIC_BACKEND=pgvector      # 2b: episodes in pgvector
export SEMANTIC_BACKEND=firestore     # 2c: facts in Firestore
export FIRESTORE_EMULATOR_HOST=localhost:8080   # the emulator, not cloud
# export EMBEDDER=gemini              # real Gemini embeddings (needs GOOGLE_API_KEY);
                                      # default "fake" is deterministic and offline
```

Real Firestore (later) is a pure connection swap: unset
`FIRESTORE_EMULATOR_HOST` and provide real credentials — no code
changes. Same for managed Redis/Postgres: just change the URLs.

Kill/restart drill (the roadmap's acceptance): `docker compose restart
redis pgvector`, then re-run the integration tests — long-term memory
must survive, because it lives server-side.

## Serving (Phase 3)

`docker compose up -d --build` also starts three identical agent replicas
(Agent 1-a / 1-b / 1-c from the diagram) behind an nginx load balancer.
Any replica serves any user — session state lives in Redis and memory in
pgvector/Firestore, so there are no sticky sessions.

Ports (host): **8000** = load balancer (the only public entry), 6379 =
Redis, 5432 = pgvector, 8080 = Firestore emulator. Replicas listen on
:8000 inside the compose network only.

```bash
docker compose up -d --build
curl localhost:8000/health
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"user_id":"maya","session_id":"s1","message":"track order ORD-1003"}'
```

Live turns need a key: `GOOGLE_API_KEY=... docker compose up -d --build`
(without it, `/health` works and `/chat` returns a 500 naming the missing
key — never a traceback). The response carries `replica_id` so you can
watch turns fan out across replicas while the conversation stays coherent:

```bash
for i in 1 2 3 4 5 6; do
  curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' \
    -d '{"user_id":"maya","session_id":"s1","message":"track ORD-1003"}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["replica_id"], d["events_recorded"])'
done
```

p95 turn-latency budget: documented as an acceptance criterion in
ROADMAP.md, but not measurable here — it needs the live model and real
load, so it stays a manual check on your machine.

## Run it

```bash
# one-time setup
/home/hatch/workspace/.venv-adk/bin/pip install -r requirements.txt

# full offline test suite -- no API key, no Docker needed
# (125 unit tests; 8 integration tests self-skip without the compose stack)
/home/hatch/workspace/.venv-adk/bin/python -m pytest tests/ -q

# integration tests only (needs: docker compose up -d --build)
/home/hatch/workspace/.venv-adk/bin/python -m pytest tests/ -q -m integration

# eval set against the real backends (needs: docker compose up -d)
SESSION_BACKEND=redis EPISODIC_BACKEND=pgvector SEMANTIC_BACKEND=firestore \
  /home/hatch/workspace/.venv-adk/bin/python -m pytest tests/test_eval.py -q

# offline smoke test of the non-LLM parts
/home/hatch/workspace/.venv-adk/bin/python demo.py

# same smoke, with the audit log written to JSONL
/home/hatch/workspace/.venv-adk/bin/python demo.py --audit /tmp/audit.jsonl

# live two-session demo -- needs GOOGLE_API_KEY in the environment
export GOOGLE_API_KEY=...
/home/hatch/workspace/.venv-adk/bin/python demo.py --live
# live demo with audit trail
/home/hatch/workspace/.venv-adk/bin/python demo.py --live --audit /tmp/audit.jsonl
```

The live demo scripts the roadmap's acceptance story: session 1, Maya
reports order ORD-1001 arrived 5 days late (gets a $10 goodwill credit)
and asks for email updates (preference remembered); session 2, a fresh
session, opens with the preference recalled and the past episode
referenced; her $75 refund request on ORD-1002 is drafted and escalated,
not executed.

## Layout

```
support_agent/
  agent.py         LlmAgent, instruction provider, recall + audit callbacks, make_runner()
  audit.py         AuditLog (in-memory + JSONL), PII scrubbing, per-user registry
  tools.py         mock OMS tools + remember_fact/recall_memories (FunctionTools)
  rules.py         deterministic procedural rules (pure functions; thresholds from policy.yaml)
  memory_banks.py  EpisodicBank + SemanticBank (+ shared scrub_event/score_facts helpers)
  config.py        model name, app name, policy (from rules/policy.yaml), backend env config
  api.py           Phase 3: FastAPI app -- POST /chat + GET /health over the ADK Runner
  backends/
    __init__.py           env-selected factories: make_session_service/_memory_service/_semantic_store/_embedder
    redis_sessions.py     2a: RedisSessionService (BaseSessionService, sliding TTL, PII-scrubbed)
    pgvector_memory.py    2b: PgVectorMemoryService + Fake/Gemini embedders + Fake/PgVector stores
    firestore_semantics.py 2c: FirestoreSemanticBank (SemanticBank-compatible) + FakeFirestore
    git_rules.py          2d: loads + validates rules/policy.yaml, renders rules/procedural.md
  fixtures.py      mock customers/orders/policies -- all fictional, no real PII
rules/
  policy.yaml      procedural policy thresholds (git-versioned source of truth)
  procedural.md    system-prompt prose (git-versioned; {refund_threshold}/{return_window_days} slots)
tests/             125 unit tests (offline) + 12-scenario eval + 8 integration tests (marked, self-skip)
demo.py            offline smoke (default) / live two-session demo (--live); --audit PATH
docker-compose.yml Phase 2 backends + Phase 3 replicas (agent-1a/b/c) + nginx LB (:8000)
Dockerfile         agent replica image (uvicorn support_agent.api:app on :8000)
nginx/nginx.conf   least-conn LB over the three replicas
docs/              target-architecture.svg -- the video's Step-5 diagram, redrawn
```

All data is mock and fictional. See [`AGENTS.md`](AGENTS.md) for working
conventions and [`ROADMAP.md`](ROADMAP.md) for Phase 4.
