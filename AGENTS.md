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
  Card-number-shaped strings are refused at the bank boundary. Banks are
  keyed `(app_name, user_id)` in a module registry -- never shared across
  users; `clear_registry()` is tests-only.
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
- Do NOT push to GitHub or create any repo -- the user confirms before any
  push. Do NOT ask for an API key; the live path reads `GOOGLE_API_KEY`
  from the environment.

## Phase 1+ notes (from ROADMAP.md)
- `audit.py` JSONL event log comes in Phase 1 (established pattern:
  truncated preview + SHA-256 hash, never credentials).
- Phase 2 swaps each in-memory service for the video's store behind the
  same ADK seams; agent code must not change to accommodate a backend.
- When adding a layer, extend the recall block and the eval scenarios,
  not the tool signatures.
