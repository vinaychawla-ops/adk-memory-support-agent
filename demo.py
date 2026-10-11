"""demo.py -- offline smoke test (default) or live two-session demo (--live).

Offline (no key needed):
    python demo.py
Exercises the non-LLM parts end to end: tools, rules, semantic
upsert/supersede, session-state wiring, episodic record/recall, and
verifies the live-loop pieces (LlmAgent, Runner, services) construct
correctly.

Live (needs GOOGLE_API_KEY in the environment):
    python demo.py --live
Scripted two-session story against Gemini:
  session 1: Maya reports ORD-1001 arrived 5 days late -> $10 goodwill
             credit; she asks for email updates -> preference remembered.
  session 2 (fresh session, same user): agent opens already informed
             (preference recalled, past episode referenced); Maya asks for
             a $75 refund on ORD-1002 -> drafted + escalated, never executed.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types

from support_agent import agent as agent_module
from support_agent import config, fixtures, rules
from support_agent import tools as tool_module
from support_agent.memory_banks import (
    MemoryBanks,
    clear_registry,
    get_banks,
    register_banks,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class _FakeSession:
    def __init__(self, app_name="test_app", user_id="test_user"):
        self.app_name = app_name
        self.user_id = user_id


class _FakeToolContext:
    """Minimal stand-in for ADK ToolContext: .state dict + .session."""

    def __init__(self, app_name="test_app", user_id="test_user"):
        self.state: dict = {}
        self.session = _FakeSession(app_name, user_id)


def _event(author: str, text: str) -> Event:
    return Event(
        author=author,
        content=types.Content(role="user" if author == "user" else "model",
                              parts=[types.Part(text=text)]),
    )


def offline_smoke() -> int:
    from google.adk.memory import InMemoryMemoryService

    print("== offline smoke: tools ==")
    clear_registry()
    fixtures.reset_counters()
    ctx = _FakeToolContext()

    r = tool_module.lookup_order("ORD-1001", ctx)
    check("lookup_order finds ORD-1001", r.get("order_id") == "ORD-1001")
    check("lookup_order sets session state", ctx.state.get("active_order_id") == "ORD-1001")
    check("lookup_order reports lateness", r.get("days_late") == 5)
    check("lookup_order rejects bad id", "error" in tool_module.lookup_order("nope", ctx))
    check("lookup_order unknown id", "error" in tool_module.lookup_order("ORD-9999", ctx))

    r = tool_module.track_order("ORD-1003", ctx)
    check("track_order in_transit", r.get("status") == "in_transit" and len(r["tracking_events"]) == 2)

    r = tool_module.start_return("ORD-1002", "changed my mind", ctx)
    check("start_return eligible", r.get("status") == "return_started")
    check("start_return sets state", ctx.state["return_in_progress"]["order_id"] == "ORD-1002")
    r = tool_module.start_return("ORD-1004", "too late", ctx)
    check("start_return refuses old order", "error" in r and "30-day" in r["rule_reason"])

    check("get_return_policy", "30 days" in tool_module.get_return_policy()["policy"])

    r = tool_module.request_refund("ORD-1001", 10.0, "late delivery", ctx)
    check("request_refund <=$50 approved", r.get("status") == "approved" and r.get("credit_id", "").startswith("CR-"))
    r = tool_module.request_refund("ORD-1002", 75.0, "changed mind", ctx)
    check("request_refund >$50 drafted", r.get("status") == "drafted_for_approval")
    check("request_refund >$50 escalated", r.get("escalation_ticket", "").startswith("TKT-"))

    r = tool_module.escalate_to_human("legal threat", "summary", ctx)
    check("escalate_to_human", r.get("ticket_id", "").startswith("TKT-"))

    print("== offline smoke: rules ==")
    check("refund exactly $50 approves",
          rules.evaluate_refund_request(50.0)["decision"] == "approve")
    check("refund $50.01 drafts",
          rules.evaluate_refund_request(50.01)["decision"] == "draft_for_approval")
    check("refund bad amount errors", "error" in rules.evaluate_refund_request(-5))
    check("late credit for ORD-1001",
          rules.late_delivery_credit(fixtures.get_order("ORD-1001"))["credit"] == 25.0)
    check("no late credit on-time order",
          rules.late_delivery_credit(fixtures.get_order("ORD-1002"))["credit"] == 0.0)
    check("escalate on legal threat",
          rules.should_escalate({"legal_threat": True})["escalate"] is True)
    check("no escalate on empty signals",
          rules.should_escalate({})["escalate"] is False)

    print("== offline smoke: semantic memory ==")
    mem = InMemoryMemoryService()
    banks = register_banks("test_app", "test_user", mem)
    check("banks registered", get_banks("test_app", "test_user") is banks)

    res = banks.semantic.upsert("contact_preference", "prefers email updates")
    check("fact stored", res.get("stored") is True and res.get("superseded") is False)
    res = banks.semantic.upsert("contact_preference", "prefers SMS updates")
    check("fact superseded", res.get("superseded") is True)
    active = banks.semantic.active_facts()
    check("one active fact per slot", len(active) == 1 and active[0].text == "prefers SMS updates")
    check("history keeps both", len(banks.semantic.history()) == 2)
    check("PII refused",
          "error" in banks.semantic.upsert("payment", "card 4111 1111 1111 1111"))
    check("search finds fact",
          any(f.slot == "contact_preference" for f in banks.semantic.search("contact")))

    print("== offline smoke: episodic memory ==")
    async def episodic_roundtrip() -> bool:
        session = Session(
            app_name="test_app", user_id="test_user", id="s1",
            state={}, events=[
                _event("user", "My order ORD-1001 arrived 5 days late"),
                _event("support_agent", "I issued a $10 goodwill credit for the late delivery"),
            ],
        )
        await banks.episodic.record_session(session)
        hits = await banks.episodic.recall("late delivery credit")
        return any("ORD-1001" in h for h in hits)

    check("episodic record + recall", asyncio.run(episodic_roundtrip()))

    print("== offline smoke: recall block + session-state wiring ==")
    async def recall_wiring() -> tuple[bool, bool]:
        state: dict = {}
        await agent_module.ensure_memory_recall(state, "test_app", "test_user", "preferences")
        first = state.get("memory_context", "")
        await agent_module.ensure_memory_recall(state, "test_app", "test_user", "other query")
        second = state.get("memory_context", "")
        return ("prefers SMS updates" in first, first == second)

    has_fact, idempotent = asyncio.run(recall_wiring())
    check("recall block carries semantic fact", has_fact)
    check("recall runs once per session", idempotent)

    print("== offline smoke: live-loop construction (no key) ==")
    try:
        ag = agent_module.build_agent()
        runner, ss, ms, _ = agent_module.make_runner("test_user")
        ok = (
            ag.name == "support_agent"
            and len(ag.tools) == 8
            and runner is not None and ss is not None and ms is not None
        )
        check("agent + Runner construct without key", ok, f"tools={len(ag.tools)}")
    except Exception as e:  # noqa: BLE001 -- smoke must report, not crash
        check("agent + Runner construct without key", False, repr(e))

    print()
    if FAILURES:
        print(f"OFFLINE SMOKE: {len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("OFFLINE SMOKE: all checks passed.")
    return 0


# --- live demo ---------------------------------------------------------------

async def _turn(runner, user_id: str, session_id: str, text: str) -> str:
    """Send one user message; return the agent's final text reply."""
    final = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and event.author == "support_agent":
                    final += part.text
    return final.strip()


async def live_demo() -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Set GOOGLE_API_KEY in the environment to run the live demo.")
        return 2

    user_id = "maya"
    runner, session_service, memory_service, agent = agent_module.make_runner(user_id)
    banks = get_banks(config.APP_NAME, user_id)
    assert banks is not None

    print("=== session 1: late delivery + preference ===")
    await session_service.create_session(
        app_name=config.APP_NAME, user_id=user_id, session_id="s1"
    )
    for text in [
        "Hi, I'm Maya. My order ORD-1001 arrived 5 days late.",
        "Can I get a $10 credit for the trouble?",
        "Also, please email me updates about my orders from now on.",
    ]:
        print(f"\nMaya: {text}")
        reply = await _turn(runner, user_id, "s1", text)
        print(f"Agent: {reply}")

    # Session ends -> episode stored in long-term memory.
    session = await session_service.get_session(
        app_name=config.APP_NAME, user_id=user_id, session_id="s1"
    )
    await banks.episodic.record_session(session)
    print("\n(session 1 stored to episodic memory)")

    print("\n=== session 2: fresh session, same customer ===")
    await session_service.create_session(
        app_name=config.APP_NAME, user_id=user_id, session_id="s2"
    )
    text = "I'd like a $75 refund on order ORD-1002."
    print(f"\nMaya: {text}")
    reply = await _turn(runner, user_id, "s2", text)
    print(f"Agent: {reply}")

    print("\n=== memory check ===")
    s2 = await session_service.get_session(
        app_name=config.APP_NAME, user_id=user_id, session_id="s2"
    )
    block = s2.state.get("memory_context", "")
    print("Recall block injected in session 2:")
    print(block or "(empty)")
    lowered = reply.lower()
    print("\nLive observations (judge these, they depend on the model):")
    print(f"  - reply references the email preference: {'email' in lowered}")
    print(f"  - reply does NOT claim the $75 refund was executed: "
          f"{'draft' in lowered or 'supervisor' in lowered or 'approval' in lowered}")
    return 0


def main() -> int:
    if "--live" in sys.argv:
        return asyncio.run(live_demo())
    return offline_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
