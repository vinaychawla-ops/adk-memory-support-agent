# AGENTS.md — agent-memory-support-agent

## What this repo is
Phase 0 of ROADMAP.md: the video's e-commerce customer support agent rebuilt
as a single Google ADK `LlmAgent` on Gemini, with the four memory layers
(session state, episodic, semantic, procedural) wired through ADK-native
in-memory services. Mock Order Management System, mock customers, zero
infrastructure. Runs locally; live turns need `GOOGLE_API_KEY`.

## Build / test commands
```bash
VENV=/home/hatch/workspace/.venv-adk/bin
$VENV/python -m pytest tests/ -q        # full suite, offline, must be green
$VENV/python demo.py                    # offline smoke of the non-LLM parts
export GOOGLE_API_KEY=...               # real key, never committed
$VENV/python demo.py --live             # scripted two-session live demo
$VENV/python -m py_compile $(find support_agent tests demo.py -name '*.py')
```

## Conventions
- `config.py`: ALL policy thresholds live here (`REFUND_APPROVAL_THRESHOLD`,
  `RETURN_WINDOW_DAYS`, `GOODWILL_CREDIT_MAX`). Never hardcode a threshold
  in rules, tools, or the instruction -- import it.
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
- Recall query: `_recall_before_agent` appends stable domain terms
  ("customer preferences past orders") to the user's latest message before
  `ensure_memory_recall`. Phase-0 search is keyword-based, so without this
  a message like "I want a refund" would never surface the contact
  preference. Phase 2 replaces this with vector search; until then, extend
  the domain terms rather than the tool signatures.
- Eval: `tests/test_eval.py` holds the 12 scripted scenarios -- each reads
  as a customer story and asserts memory behavior (recall / rule fired /
  fact superseded). Extend it when adding a layer or a rule; the
  `test_eval_audit_replays_full_multisession_story` test pins the exact
  16-event audit sequence, so update it deliberately when the wiring changes.
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
- Phase 2 swaps each in-memory service for the video's store behind the
  same ADK seams; agent code must not change to accommodate a backend.
  The audit callbacks resolve logs from the session, so they survive the
  swap untouched.
- When adding a layer, extend the recall block and the eval scenarios,
  not the tool signatures.
