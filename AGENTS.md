# AGENTS.md — agent-memory-support-agent

## What this repo is
Phases 0–2 of ROADMAP.md: the video's e-commerce customer support agent
rebuilt as a single Google ADK `LlmAgent` on Gemini, with the four memory
layers (session state, episodic, semantic, procedural) wired through
ADK-native in-memory services by default and durable Docker backends
(Redis / pgvector / Firestore / git-versioned rules) via env selection.
Mock Order Management System, mock customers. Runs locally; live turns
need `GOOGLE_API_KEY`.

## Build / test commands
```bash
VENV=/home/hatch/workspace/.venv-adk/bin
$VENV/python -m pytest tests/ -q        # full suite, offline, must be green
                                       # (integration tests self-skip w/o backends)
$VENV/python -m pytest tests/ -q -m integration   # needs: docker compose up -d
$VENV/python demo.py                    # offline smoke of the non-LLM parts
export GOOGLE_API_KEY=...               # real key, never committed
$VENV/python demo.py --live             # scripted two-session live demo
$VENV/python -m py_compile $(find support_agent tests demo.py -name '*.py')
```
Docker backends: `docker compose up -d` (redis :6379, pgvector :5432,
firestore emulator :8080). Select via env:
`SESSION_BACKEND=redis EPISODIC_BACKEND=pgvector SEMANTIC_BACKEND=firestore`
(+ `FIRESTORE_EMULATOR_HOST=localhost:8080`, `EMBEDDER=fake|gemini`).

## Conventions
- `config.py`: model name, app name, backend env config. Policy thresholds
  come from `rules/policy.yaml` (loaded + validated at import via
  `backends/git_rules.py`) -- never hardcode a threshold in rules.py,
  tools.py, or the instruction; import it from here.
- `rules/`: git-versioned procedural memory. `policy.yaml` is the validated
  source of truth for thresholds (flat `key: value`, stdlib-parsed; bad
  policy fails fast at startup). `procedural.md` is the system-prompt prose
  with `{refund_threshold}` / `{return_window_days}` slots, rendered by
  `git_rules.render_system_prompt()` -- prompt and deterministic rules can
  never drift apart. Rule changes go through git: the history is the audit
  trail. Do not add other `{braces}` to the markdown.
- `backends/`: Phase 2, one module per box. `__init__.py` holds only the
  env-selected factories (`make_session_service`, `make_memory_service`,
  `make_semantic_store`, `make_embedder`) with lazy imports -- importing
  this package must never require redis/psycopg/firestore libs or a key.
  Agent code never branches on backends; the ADK seams are the interface.
- `backends/redis_sessions.py`: `RedisSessionService` mirrors
  InMemorySessionService semantics (state scopes, merge on read, dedupe,
  list ordering) with JSON sessions under `msa:*` keys and a sliding TTL
  (`SESSION_TTL_SECONDS`). Events are PII-scrubbed before persistence;
  the caller's session object is never mutated by the scrub.
- `backends/pgvector_memory.py`: `PgVectorMemoryService` stores one
  embedded row per event; `search_memory` orders by cosine distance with
  a keyword-overlap boost (hybrid). `FakeEmbedder` is deterministic and
  offline (tests plumbing, not semantic quality); `GeminiEmbedder` reads
  the key from the env at call time. Blocking psycopg calls run in
  `asyncio.to_thread`. Keep `EMBEDDING_DIM` stable once the table exists.
- `backends/firestore_semantics.py`: `FirestoreSemanticBank` implements
  the `SemanticBank` contract (upsert/supersede/history/search) on one
  document per user (`semantic_memory/{app}__{user}`) -- no composite
  indexes needed. `Fact.expires` is an optional ISO date; expired facts
  stop surfacing, `purge_expired()` drops them (Phase 4 schedules it).
  The real client honors `FIRESTORE_EMULATOR_HOST`, so emulator vs. cloud
  is a pure connection swap.
- Probes must fail fast: backend `ping()`s use short timeouts (the sandbox
  blackholes ports, so bare connects hang). Integration tests are marked
  `@pytest.mark.integration` and skip when a backend is unreachable;
  `tests/conftest.py::_ensure_reachable` lets the whole eval set run
  against real backends via env vars (or skip cleanly).
- Recall query: `_recall_before_agent` appends stable domain terms
  ("customer preferences past orders") ONLY when `EPISODIC_BACKEND=memory`
  (keyword search needs word overlap). With `pgvector` the raw user text
  goes to vector search -- the Phase-0 hack is gone there.
- `rules.py`: pure functions only. No I/O, no network, no LLM. Each returns
  `{"rule", ...}`; malformed input returns `{"rule", "error"}`. This module
  plus the system instruction IS the procedural memory.
- `tools.py`: module-level, annotated, docstringed functions returning
  JSON-serializable dicts -- ADK builds `FunctionTool` schemas from them.
  Keep signatures stable; `agent.py` and `demo.py` depend on them. Every
  failure returns `{"error": ...}`, never a traceback.
- `fixtures.py`: mock data only, all fictional. `TODAY` is fixed for
  determinism; `reset_counters()` exists for tests. No real PII, ever.
- `memory_banks.py`: episodic = ADK `BaseMemoryService` (raw session events,
  keyword search); semantic = `SemanticBank` fact store with upsert/supersede
  (same slot -> old fact marked superseded, kept in `history()` for audit).
  Card-number-shaped strings are refused at the semantic boundary
  (`CARD_RE`, imported from `audit.py` -- single source of truth) and
  scrubbed from episodes in `EpisodicBank.record_session` before storage;
  the original session object is never mutated. Banks are keyed
  `(app_name, user_id)` in a module registry -- never shared across users;
  `clear_registry()` is tests-only.
- `audit.py`: stdlib-only `AuditLog` (in-memory `events` + optional JSONL),
  per-user registry mirroring the banks (`register_audit`/`get_audit`,
  `clear_audit_registry()` tests-only). `log_turn`/`log_tool_call`/
  `log_tool_result`/`log_memory_write`/`log_rule_fired`/`log_memory_recall`
  are no-ops when no log is registered -- zero overhead by default.
  Prompts/args stored as scrubbed preview + SHA-256 hash of the scrubbed
  text; card numbers redacted to `[REDACTED]` before storage. `replay()`
  re-reads the JSONL to prove the trail is complete and ordered.
- Audit wiring: `_recall_before_agent` logs the `turn`; `ensure_memory_recall`
  logs `memory_recall`; `_audit_before_tool`/`_audit_after_tool` log every
  tool invocation/outcome (they never block -- always return None).
  `request_refund`/`start_return` log `rule_fired`; `remember_fact` logs
  the semantic write; `EpisodicBank.record_session` logs the episodic write
  (with `pii_redacted` flag). A logging failure must never break the agent
  loop -- keep the helpers total and side-effect-free beyond the log.
- Eval: `tests/test_eval.py` holds the 12 scripted scenarios -- each reads
  as a customer story and asserts memory behavior (recall / rule fired /
  fact superseded). Extend it when adding a layer or a rule; the
  `test_eval_audit_replays_full_multisession_story` test pins the exact
  16-event audit sequence, so update it deliberately when the wiring changes.
  The whole eval set also runs against real backends via env vars
  (`SESSION_BACKEND=redis EPISODIC_BACKEND=pgvector SEMANTIC_BACKEND=firestore`),
  skipping cleanly when they are not running.
- Session state discipline: tools write STRUCTURED state only
  (`active_order_id`, `return_in_progress`, `memory_context`,
  `memory_recall_done`). History comes from session events, not state.
- `agent.py`: `build_agent()` and `make_runner()` must construct with NO
  API key (covered by tests). The key is read from the environment at run
  time only. `ensure_memory_recall()` holds the recall logic against a
  plain dict so it stays unit-testable; `_recall_before_agent` is the thin
  ADK wrapper. Instruction is an async provider (not a `{var}` template)
  so JSON/braces in the prompt can't collide with state injection.
- Tests are fully offline: `asyncio.run()` inside sync test functions (no
  pytest-asyncio dependency). FakeToolContext mirrors the ADK surface the
  tools touch (`.state`, `.session.app_name/.user_id`).
- Money rule (verdicts, not execution): the agent drafts refunds above the
  threshold and escalates; it never claims to have executed one.
- GitHub: pushed 2026-10-11 to public repo vinaychawla-ops/adk-memory-support-agent
  (main @6679380, 20/20 files verified). Future pushes go through
  `~/workspace/skills/github/bin/api_push.py` (git-database API; sandbox git
  has no network). Do NOT ask for an API key; the live path reads `GOOGLE_API_KEY`
  from the environment.

## Phase 1+ notes (from ROADMAP.md)
- Phase 1 (done 2026-10-11): `audit.py` JSONL event log, 12-scenario eval
  set (`tests/test_eval.py`), conflict handling verified (upsert->supersede
  ->history; newer preference wins), PII guardrail extended to episodes
  (scrubbed on write) and audit previews.
- Phase 2 (done 2026-10-11): all four boxes behind the ADK seams --
  `backends/redis_sessions.py` (2a), `backends/pgvector_memory.py` (2b,
  Fake/Gemini embedders, hybrid vector+keyword search),
  `backends/firestore_semantics.py` (2c, one doc per user, emulator-ready),
  `rules/policy.yaml` + `rules/procedural.md` (2d, git-versioned). Agent
  code has no backend branches; selection is env-driven with in-memory
  defaults. The Phase-0 keyword-search hack in the recall query now only
  applies when `EPISODIC_BACKEND=memory`.
- When adding a layer, extend the recall block and the eval scenarios,
  not the tool signatures.
