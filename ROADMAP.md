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

**What changes:** each ADK-native service is swapped for the video's store. The agent code
doesn't change — ADK's service interfaces are the seam.

- **2a — Redis:** custom `SessionService` on Redis. Conversation history + session state,
  with TTL per session. Any replica can serve any session (this unblocks Phase 3).
- **2b — PostgreSQL + pgvector:** custom `MemoryService` episodic store. Past interactions
  embedded (Gemini embeddings) and retrieved by hybrid search.
- **2c — Firestore:** semantic facts as documents `{fact, confidence, supersedes, expires}`.
  Upsert + staleness logic lives here.
- **2d — Git repo:** procedural rules as versioned YAML/Markdown, loaded at startup.
  Rule changes go through PRs — the changelog *is* the audit trail.

**Acceptance per sub-phase:** Phase-0 eval set still green with the new backend; a backend
can be killed and restarted without losing long-term memory.

## Phase 3 — Serve and scale (the left half of the diagram)

**What changes:** one local agent becomes a service.

- FastAPI `/chat` endpoint with the ADK `Runner` behind it (the green API box).
- Load balancer → N agent replicas (1-a / 1-b / 1-c). Because state lives in Redis and memory
  in pgvector/Firestore, any replica serves any user — no sticky sessions required.
- LLM config centralized (model, thinking budget, fallback model).

**Acceptance:** 3 replicas behind the LB handle interleaved sessions for the same user with
no memory loss; p95 turn latency budgeted.

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

| Diagram box | Phase 0 (now) | Target | ADK seam | Phase |
|---|---|---|---|---|
| Conversation history & session state | `InMemorySessionService` | Redis | `BaseSessionService` | 2a |
| Episodic, long term | `InMemoryMemoryService` | pgvector | `BaseMemoryService` | 2b |
| Semantic, long term | `InMemoryMemoryService` | Firestore | `BaseMemoryService` (second bank) or doc store | 2c |
| Procedural, long term | system instruction + rules module | Git-versioned rules | loaded config, no subclass needed | 2d |
| API | `demo.py` CLI | FastAPI + Runner | `Runner` | 3 |
| Load balancer / replicas | single process | LB → N replicas | infra, not ADK | 3 |

## Appendix B — Out of scope

Real order data, real customer PII, auto-executing refunds or charges (verdicts and drafts
only — a human approves), multi-tenant isolation. Those are separate decisions, not separate
phases.

---
*Effort notes are deliberately absent per phase — sized when each phase is approved, not before.*
