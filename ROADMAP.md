# Customer Support Agent with Layered Memory — Roadmap

**Source:** "The AI Agent Memory Architecture Explained — From State to Long-Term Memory"
(Applied with AI – Sanjay Kumar), rebuilt as a Google ADK agent.
**Target architecture:** the video's *Step 5 – Procedural memory* diagram — see
[`docs/target-architecture.svg`](docs/target-architecture.svg), which recreates it as the end state.
**Decisions (2026-10-11):** Gemini model · final working agent only (no staged failure demos) ·
ADK-native services for now · this roadmap kept for future reference.

## The target, in one paragraph

A chat UI talks to an API layer; a load balancer fans out to N identical agent replicas
(Agent 1-a / 1-b / 1-c). The LLM receives the user prompt plus previous conversations. The agents
call the org's internal APIs as tools (the Order Management System). Behind them sits a
**Memory Handling** panel with four stores, each holding a different kind of memory:

| Store (video) | Holds | Lifetime |
|---|---|---|
| PostgreSQL + pgvector | Episodic memory — what happened in past interactions | Long term |
| Redis (distributed cache) | Conversation history & session state | Session / TTL |
| Firestore | Semantic memory — facts and preferences | Long term |
| Git repo / CMS / S3 | Procedural memory — rules for how to behave | Long term, versioned |

The video's closing line is the real spec: the goal isn't just an agent that remembers, but
knowing **what to remember, where to remember it, and when to forget**. Phase 4 is that last clause.

---

## Phase 0 — ADK-native build (now, in progress)

**What changes vs nothing:** this is the baseline. One agent, zero infrastructure.

- Single `LlmAgent` on Gemini. E-commerce customer support: order lookup, tracking,
  returns, escalation.
- **Session state + conversation history** → ADK `InMemorySessionService`.
  `Session.state` holds *structured* execution state (active `order_id`, return in progress,
  what info is still missing) — not a raw message dump. History comes from session events.
- **Episodic + semantic** → ADK `InMemoryMemoryService`. After each session,
  `add_session_to_memory` extracts what happened (episodic) and durable facts/preferences
  (semantic); a recall step calls `search_memory` so a new session opens already informed.
- **Procedural** → system instruction (tone, escalation thresholds) plus a deterministic
  rules module (the checks.py pattern): e.g. refunds above $X need human approval, the agent
  never auto-executes them.
- **Tools** → mock Order Management System in code (`lookup_order`, `track_order`,
  `start_return`, `escalate_to_human`). No real order data, no real PII.
- Runs locally with `GOOGLE_API_KEY`; everything else is pytest-verified offline.

**Acceptance:** a scripted two-session demo — session 1 establishes a preference (email updates)
and an episode (late delivery, resolved with a credit); session 2, fresh session, recalls the
preference, references the past episode, and enforces the refund-approval rule. Full pytest
suite green without a key.

## Phase 1 — Harden the core

**What changes:** the agent stops being a demo and starts being testable.

- `audit.py` JSONL event log (the established pattern) — a durable, on-disk episodic record
  from day one; prompt stored as truncated preview + hash, never credentials.
- Eval set: ~12 scripted support scenarios, each asserting expected memory behavior
  (recall happened / rule fired / fact superseded). Runs as pytest.
- Conflict handling: a preference stated in session 3 supersedes session 1's — semantic
  upsert, not append. PII guardrail: card numbers and the like never enter any memory layer.

**Acceptance:** eval set green; audit log replays a full multi-session story.

## Phase 2 — Durable memory backends (the Memory Handling panel, box by box)

**Status: done 2026-10-11** — all four boxes implemented behind the ADK
seams, running in local Docker per the user's decision (real cloud is a
later connection swap). Agent code has no backend branches.

**What changed:** each ADK-native service is swapped for the video's store. The agent code
doesn't change — ADK's service interfaces are the seam.

- **2a — Redis:** `backends/redis_sessions.py` (`RedisSessionService`).
  Sessions as JSON under `msa:*` keys with a sliding 30-day TTL; app/user
  state in shared hashes, merged on read exactly like the in-memory
  service. Any replica serves any session (unblocks Phase 3). Events are
  PII-scrubbed before persistence.
- **2b — PostgreSQL + pgvector:** `backends/pgvector_memory.py`
  (`PgVectorMemoryService`). One embedded row per event; `search_memory`
  orders by cosine distance with a keyword-overlap boost (hybrid). This
  **replaces the Phase-0 keyword-search hack** — with
  `EPISODIC_BACKEND=pgvector` the recall query is the raw user text, no
  appended domain terms. Embeddings via `EMBEDDER=fake` (deterministic,
  offline) or `gemini` (key from env at call time).
- **2c — Firestore:** `backends/firestore_semantics.py`
  (`FirestoreSemanticBank`, same contract as `SemanticBank`). One document
  per user (`semantic_memory/{app}__{user}`), facts as
  `{slot, text, confidence, superseded_by, expires}`. Local = Firestore
  emulator in Docker; the client honors `FIRESTORE_EMULATOR_HOST`, so
  real Firestore is a pure connection swap.
- **2d — Git repo:** `rules/policy.yaml` (validated thresholds — the
  source of truth `config.py` loads at import) + `rules/procedural.md`
  (system-prompt prose with `{refund_threshold}`/`{return_window_days}`
  slots). Rule changes go through git — the changelog *is* the audit trail.

**Acceptance per sub-phase:** the Phase-1 eval set passes with the new
backends (`SESSION_BACKEND=redis EPISODIC_BACKEND=pgvector
SEMANTIC_BACKEND=firestore pytest tests/test_eval.py`); the kill/restart
drill (`docker compose restart redis pgvector`) loses no long-term
memory — data lives server-side. 6 integration tests
(`tests/test_backends_integration.py`, marked, self-skipping) pin this;
115 unit tests stay green offline with fakes.

## Phase 3 — Serve and scale (the left half of the diagram)

**Status: done 2026-10-11** — the agent is an HTTP service: three
identical replicas behind nginx, all in the compose file. No sticky
sessions, no auth (localhost only), everything env-driven.

**What changed:**
- `support_agent/api.py`: FastAPI app. `POST /chat`
  (`{user_id, session_id, message}` → `{reply, model, replica_id,
  backends, memory_recall, events_recorded}`) runs one turn through the
  ADK Runner built by the existing `make_runner()` factories; `GET
  /health` reports liveness + per-backend reachability and never needs a
  key. After each turn only the *new* session events are written to
  episodic memory, deduplicated via `episodic_recorded_upto` in session
  state; each turn's delta is stored under `<session_id>#e<offset>` so
  the in-memory backend (which keys episodes by session id) can't
  overwrite the previous turn's episode. Errors are 500s with
  `{"error": ...}`, never tracebacks. PII scrubbing and audit logging
  carry over unchanged (episodes scrubbed before storage, audit previews
  scrubbed + hashed).
- `Dockerfile`: replica image (`uvicorn support_agent.api:app` on :8000).
- `nginx/nginx.conf`: `least_conn` over `agent-1a/1-b/1-c:8000`.
- `docker-compose.yml`: backends + 3 replicas + nginx. Host ports:
  **8000** = nginx (the only public entry), 6379 = redis, 5432 =
  pgvector, 8080 = firestore emulator (no clash). Replicas listen on
  :8000 inside the compose network only; `REPLICA_ID` env identifies
  each replica in responses.
- LLM config centralized via env (`MODEL`, `GOOGLE_API_KEY`), same
  pattern as the rest — no per-replica config.

**Acceptance:** two replicas sharing one backend (simulated offline with
a shared in-memory session service) handle interleaved turns for one
user with no loss — state, history, and episodic memory all visible
across replicas (`tests/test_api.py::
test_interleaved_replicas_share_everything`). The live version is a
manual drill on his machine: `curl` turns at `localhost:8000/chat` and
watch `replica_id` cycle across agent-1a/1-b/1-c while the conversation
stays coherent (commands in README). p95 turn latency is documented as
an acceptance criterion but stays unverifiable here — it needs the live
model and real load.

## Phase 4 — Operate: the forgetting layer

**What changes:** memory gets a lifecycle. This is the part the video names and almost no
tutorial implements.

- **Session state:** Redis TTL expiry; abandoned carts/sessions compacted, not kept forever.
- **Episodic:** old episodes summarized into semantic facts ("customer had 3 late deliveries
  in Q4" replaces three full transcripts), then the raw episodes drop.
- **Semantic:** staleness review — facts carry `expires` / `last_confirmed`; low-confidence
  or contradicted facts are re-confirmed or purged.
- **Procedural:** rules versioned in git; deprecated rules stay readable but stop firing.
- **Metrics:** memory hit rate per layer, retrieval precision on the eval set, context tokens
  consumed per turn per layer, cost per resolved ticket.

**Acceptance:** a "forgetting drill" — aged-out data is provably gone from every store, and
the eval set still passes.

---

## Appendix A — ADK seams (what to subclass, and when)

| Diagram box | Phase 0–1 (in-memory) | Phase 2 (done 2026-10-11) | ADK seam |
|---|---|---|---|
| Conversation history & session state | `InMemorySessionService` | `RedisSessionService` (JSON, sliding TTL) | `BaseSessionService` |
| Episodic, long term | `InMemoryMemoryService` | `PgVectorMemoryService` (embedded rows, hybrid search) | `BaseMemoryService` |
| Semantic, long term | `SemanticBank` (in-memory) | `FirestoreSemanticBank` (one doc/user, emulator-ready) | `SemanticBank`-compatible store |
| Procedural, long term | system instruction + rules module | `rules/policy.yaml` + `rules/procedural.md`, git-versioned | loaded config, no subclass needed |
| API | `demo.py` CLI | FastAPI + Runner (done 2026-10-11) | `Runner` |
| Load balancer / replicas | single process | nginx → 3 replicas (done 2026-10-11) | infra, not ADK |

## Appendix B — Out of scope

Real order data, real customer PII, auto-executing refunds or charges (verdicts and drafts
only — a human approves), multi-tenant isolation. Those are separate decisions, not separate
phases.

---
*Effort notes are deliberately absent per phase — sized when each phase is approved, not before.*
