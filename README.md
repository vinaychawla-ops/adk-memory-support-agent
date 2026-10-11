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

## Run it

```bash
# one-time setup (a venv with google-adk already exists at ../.venv-adk;
# or make your own)
/home/hatch/workspace/.venv-adk/bin/pip install -r requirements.txt

# full offline test suite -- no API key needed
/home/hatch/workspace/.venv-adk/bin/python -m pytest tests/ -q

# offline smoke test of the non-LLM parts
/home/hatch/workspace/.venv-adk/bin/python demo.py

# live two-session demo -- needs GOOGLE_API_KEY in the environment
export GOOGLE_API_KEY=...
/home/hatch/workspace/.venv-adk/bin/python demo.py --live
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
  agent.py         LlmAgent, instruction provider, recall callback, make_runner()
  tools.py         mock OMS tools + remember_fact/recall_memories (FunctionTools)
  rules.py         deterministic procedural rules (pure functions)
  memory_banks.py  EpisodicBank (ADK memory service) + SemanticBank (fact store)
  fixtures.py      mock customers/orders/policies -- all fictional, no real PII
  config.py        model name, app name, all policy thresholds
tests/             pytest suite, fully offline (41 tests)
demo.py            offline smoke (default) / live two-session demo (--live)
docs/              target-architecture.svg -- the video's Step-5 diagram, redrawn
```

All data is mock and fictional. See [`AGENTS.md`](AGENTS.md) for working
conventions and [`ROADMAP.md`](ROADMAP.md) for Phases 1–4.
