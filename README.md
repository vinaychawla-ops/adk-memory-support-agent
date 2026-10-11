# Customer Support Agent with Layered Memory

A Google ADK rebuild of the e-commerce customer support agent from
*"The AI Agent Memory Architecture Explained — From State to Long-Term Memory"*
(Applied with AI – Sanjay Kumar). Phase 0 of the roadmap in
[`ROADMAP.md`](ROADMAP.md): one agent, zero infrastructure, ADK-native
services, mock data. Runs locally.

## The four layers, and where each lives

| Video layer | What it holds | Where it lives here |
|---|---|---|
| Conversation history & session state | Structured execution state: `active_order_id`, `return_in_progress`, `memory_context` — never raw transcript | ADK `InMemorySessionService` (`Session.state`); tools write it |
| Episodic memory | What happened in past interactions (session events) | ADK `InMemoryMemoryService` via `EpisodicBank.record_session()` / `.recall()` |
| Semantic memory | Durable customer facts/preferences with upsert + supersede | `SemanticBank` in `support_agent/memory_banks.py` (in-memory; Phase 2c → Firestore) |
| Procedural memory | How the agent must behave: tone, refund thresholds, escalation | System instruction in `support_agent/agent.py` + deterministic `support_agent/rules.py` |

The recall flow: `before_agent_callback` calls `search_memory` once per
session, builds a recall block (facts + relevant episodes), and stores it
in `Session.state["memory_context"]`; the instruction provider renders it
into the system prompt. After a session, `add_session_to_memory` stores
the episode. Tools: a mock Order Management System
(`lookup_order`, `track_order`, `start_return`, `get_return_policy`,
`request_refund`, `escalate_to_human`) plus `remember_fact` /
`recall_memories`.

Policy, enforced deterministically in `rules.py`: refunds/credits at or
below **$50** may be issued; anything above is **drafted and escalated to
a human — never auto-executed**. Returns only within 30 days of delivery.

## Audit log (Phase 1)

`support_agent/audit.py` — every turn, tool call, memory write
(episodic + semantic), procedural rule firing, and memory recall is
logged to an `AuditLog`: in-memory events plus an optional JSONL file.
This is the durable, on-disk episodic record.

- Prompts are stored as a truncated preview + SHA-256 hash, never the
  full text. Card-number-shaped strings are redacted to `[REDACTED]`
  *before* anything is stored — they can't reach any memory layer,
  including this log.
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

## Run it

```bash
# one-time setup (a venv with google-adk already exists at ../.venv-adk;
# or make your own)
/home/hatch/workspace/.venv-adk/bin/pip install -r requirements.txt

# full offline test suite -- no API key needed
/home/hatch/workspace/.venv-adk/bin/python -m pytest tests/ -q

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
  rules.py         deterministic procedural rules (pure functions)
  memory_banks.py  EpisodicBank (ADK memory service, PII-scrubbed) + SemanticBank (fact store)
  fixtures.py      mock customers/orders/policies -- all fictional, no real PII
  config.py        model name, app name, all policy thresholds
tests/             pytest suite, fully offline (67 tests: unit + 12-scenario eval)
demo.py            offline smoke (default) / live two-session demo (--live); --audit PATH
docs/              target-architecture.svg -- the video's Step-5 diagram, redrawn
```

All data is mock and fictional. See [`AGENTS.md`](AGENTS.md) for working
conventions and [`ROADMAP.md`](ROADMAP.md) for Phases 1–4.
