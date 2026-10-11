"""Eval set: scripted support scenarios asserting memory behavior.

Each test reads as a story: what the customer did, what the agent must
remember or enforce, and the assertion that proves it. The agent loop is
exercised through its testable seams (tools, ensure_memory_recall, the
ADK callbacks, record_session) -- the LLM itself is never invoked, so
the whole set runs offline. The live turn loop still needs a key
(demo.py --live); that path is unverified here.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types

from support_agent import agent as agent_module
from support_agent import config, fixtures, rules
from support_agent import tools
from support_agent.audit import (
    clear_audit_registry,
    get_audit,
    register_audit,
)
from support_agent import audit as audit_module


@pytest.fixture()
def audit_log(tmp_path):
    clear_audit_registry()
    log = register_audit("test_app", "test_user", path=str(tmp_path / "audit.jsonl"))
    yield log
    clear_audit_registry()


def _event(author: str, text: str) -> Event:
    return Event(
        author=author,
        content=types.Content(
            role="user" if author == "user" else "model",
            parts=[types.Part(text=text)],
        ),
    )


class _FakeCallbackContext:
    """Stand-in for ADK CallbackContext: .state dict + .session."""

    def __init__(self, app_name, user_id, session_id, events):
        self.state: dict = {}
        self.session = SimpleNamespace(
            app_name=app_name, user_id=user_id, id=session_id, events=events
        )


class _FakeCtx:
    """Stand-in for ReadonlyContext: only .session.state is read."""

    def __init__(self, state):
        self.session = SimpleNamespace(state=state)


async def _run_tool(name: str, ctx, **kwargs):
    """Drive a tool through the audit callbacks, like the real loop does."""
    tool = {t.name: t for t in tools.build_tools()}[name]
    await agent_module._audit_before_tool(tool, kwargs, ctx)
    result = getattr(tools, name)(**kwargs, tool_context=ctx)
    if asyncio.iscoroutine(result):
        result = await result
    await agent_module._audit_after_tool(tool, kwargs, ctx, result)
    return result


# --- 1. preference recalled across sessions ---------------------------------
def test_eval_preference_recalled_across_sessions(ctx, banks):
    async def go():
        # Session 1: Maya states a durable preference.
        r = tools.remember_fact("prefers email updates", "contact_preference", ctx)
        assert r["stored"] is True
        # Session 2: brand-new state, same user -- recall must find it.
        state: dict = {}
        await agent_module.ensure_memory_recall(
            state, "test_app", "test_user", "preferences"
        )
        return state["memory_context"]

    block = asyncio.run(go())
    assert "prefers email updates" in block


# --- 2. conflicting preference supersedes, not appends -----------------------
def test_eval_conflicting_preference_supersedes(ctx, banks):
    async def go():
        tools.remember_fact("prefers email updates", "contact_preference", ctx)
        # Session 2: she changes her mind.
        r = tools.remember_fact("prefers SMS updates", "contact_preference", ctx)
        assert r["superseded"] is True
        # Session 3: only the newest fact may surface.
        state: dict = {}
        await agent_module.ensure_memory_recall(
            state, "test_app", "test_user", "contact"
        )
        return state["memory_context"]

    block = asyncio.run(go())
    assert "prefers SMS updates" in block
    assert "email" not in block
    assert len(banks.semantic.history()) == 2  # old fact kept for audit only


# --- 3. card numbers never enter any memory layer -----------------------------
def test_eval_card_number_never_enters_any_memory_layer(ctx, banks, audit_log):
    r = tools.remember_fact("my card is 4111 1111 1111 1111", "payment", ctx)
    assert "error" in r
    assert banks.semantic.active_facts() == []

    async def go():
        session = Session(
            app_name="test_app",
            user_id="test_user",
            id="s9",
            state={},
            events=[
                _event("user", "here is my card 4111-1111-1111-1111 for the refund")
            ],
        )
        raw = session.events[0].content.parts[0].text
        await banks.episodic.record_session(session)
        assert session.events[0].content.parts[0].text == raw  # original untouched
        return await banks.episodic.recall("card refund")

    hits = asyncio.run(go())
    assert hits, "the (scrubbed) episode should still be recalled"
    assert all("4111" not in h for h in hits)
    assert any("[REDACTED]" in h for h in hits)
    assert "4111" not in json.dumps(audit_log.events)


# --- 4. episodic record/recall round-trip ------------------------------------
def test_eval_episodic_roundtrip(banks):
    async def go():
        session = Session(
            app_name="test_app",
            user_id="test_user",
            id="s1",
            state={},
            events=[
                _event("user", "My order ORD-1001 arrived 5 days late"),
                _event(
                    "support_agent",
                    "Issued a $10 goodwill credit for the late delivery",
                ),
            ],
        )
        await banks.episodic.record_session(session)
        return await banks.episodic.recall("late delivery credit")

    hits = asyncio.run(go())
    assert any("ORD-1001" in h for h in hits)


# --- 5. session state carried across turns ------------------------------------
def test_eval_session_state_carried_across_turns(ctx):
    # Turn 1: look up the late order.
    tools.lookup_order("ORD-1001", ctx)
    assert ctx.state["active_order_id"] == "ORD-1001"
    # Turn 2, same session state: start a return on another order.
    r = tools.start_return("ORD-1002", "changed my mind", ctx)
    assert r["status"] == "return_started"
    assert ctx.state["active_order_id"] == "ORD-1002"
    assert ctx.state["return_in_progress"]["return_id"].startswith("RET-")


# --- 6. $50 refund threshold boundary -----------------------------------------
def test_eval_refund_threshold_boundary(ctx):
    small = tools.request_refund("ORD-1001", 50.00, "late delivery", ctx)
    assert small["status"] == "approved"
    assert small["credit_id"].startswith("CR-")

    large = tools.request_refund("ORD-1002", 50.01, "changed mind", ctx)
    assert large["status"] == "drafted_for_approval"
    assert large["escalation_ticket"].startswith("TKT-")
    assert "credit_id" not in large  # nothing was executed
    assert "draft" in large


# --- 7. escalation triggers ----------------------------------------------------
def test_eval_escalation_triggers(ctx):
    assert rules.should_escalate({"legal_threat": True})["escalate"] is True
    assert rules.should_escalate({"repeat_contacts": 3})["escalate"] is True
    assert rules.should_escalate({"repeat_contacts": 2})["escalate"] is False
    assert rules.should_escalate({"agent_unsure": True})["escalate"] is True

    r = tools.escalate_to_human(
        "legal threat", "Customer mentioned a lawyer.", ctx
    )
    assert r["status"] == "escalated"
    assert r["ticket_id"].startswith("TKT-")


# --- 8. 30-day return window ----------------------------------------------------
def test_eval_return_window_enforced(ctx):
    ok = tools.start_return("ORD-1002", "changed my mind", ctx)
    assert ok["status"] == "return_started"

    old = tools.start_return("ORD-1004", "too late", ctx)
    assert "error" in old
    assert "30-day" in old["rule_reason"]

    transit = tools.start_return("ORD-1003", "impatient", ctx)
    assert "error" in transit
    assert "in_transit" in transit["rule_reason"]


# --- 9. late-delivery goodwill credit -------------------------------------------
def test_eval_goodwill_credit_for_late_delivery(ctx):
    v = rules.late_delivery_credit(fixtures.get_order("ORD-1001"))
    assert v["credit"] == 25.0  # 5 days x $5, capped at the goodwill max

    # The capped credit sits under the $50 threshold, so it stays
    # auto-approvable -- the agent can actually make this right.
    r = tools.request_refund("ORD-1001", v["credit"], "late delivery", ctx)
    assert r["status"] == "approved"

    assert rules.late_delivery_credit(fixtures.get_order("ORD-1002"))["credit"] == 0.0


# --- 10. audit log replays a full multi-session story ---------------------------
def test_eval_audit_replays_full_multisession_story(ctx, banks, audit_log):
    async def go():
        # ---- session 1 ------------------------------------------------
        s1 = _FakeCallbackContext(
            "test_app",
            "test_user",
            "s1",
            [_event("user", "My order ORD-1001 arrived 5 days late")],
        )
        await agent_module._recall_before_agent(s1)

        ctx.session.id = "s1"
        await _run_tool("lookup_order", ctx, order_id="ORD-1001")
        await _run_tool(
            "remember_fact",
            ctx,
            fact="prefers email updates",
            slot="contact_preference",
        )
        r = await _run_tool(
            "request_refund", ctx, order_id="ORD-1001", amount=10.0,
            reason="late delivery",
        )
        assert r["status"] == "approved"

        session = Session(
            app_name="test_app",
            user_id="test_user",
            id="s1",
            state={},
            events=[
                _event("user", "My order ORD-1001 arrived 5 days late"),
                _event(
                    "support_agent",
                    "Issued a $10 goodwill credit for the late delivery",
                ),
            ],
        )
        await banks.episodic.record_session(session)

        # ---- session 2 (fresh session, same customer) ------------------
        s2 = _FakeCallbackContext(
            "test_app",
            "test_user",
            "s2",
            [
                _event(
                    "user",
                    "Hi, following up on my late order -- please keep "
                    "emailing me updates. I'd like a $75 refund on ORD-1002.",
                )
            ],
        )
        await agent_module._recall_before_agent(s2)
        block = s2.state["memory_context"]
        assert "prefers email updates" in block
        assert "ORD-1001" in block

        ctx.session.id = "s2"
        r = await _run_tool(
            "request_refund", ctx, order_id="ORD-1002", amount=75.0,
            reason="changed mind",
        )
        assert r["status"] == "drafted_for_approval"
        assert "credit_id" not in r
        return block

    block = asyncio.run(go())
    assert "prefers email updates" in block

    # The on-disk trail replays the story, event by event, in order.
    replayed = audit_log.replay()
    assert [e["event"] for e in replayed] == [
        "turn", "memory_recall",
        "tool_call", "tool_result",
        "tool_call", "memory_write", "tool_result",
        "tool_call", "rule_fired", "tool_result",
        "memory_write",
        "turn", "memory_recall",
        "tool_call", "rule_fired", "tool_result",
    ]
    for e in replayed:
        assert e["ts"] and e["component"] == "support_agent"

    fired = [e for e in replayed if e["event"] == "rule_fired"]
    assert [(e["rule"], e["decision"]) for e in fired] == [
        ("refund_approval", "approve"),
        ("refund_approval", "draft_for_approval"),
    ]

    writes = [e for e in replayed if e["event"] == "memory_write"]
    assert [(e["kind"], e["detail"].get("session_id", e["detail"].get("slot")))
            for e in writes] == [
        ("semantic", "contact_preference"),
        ("episodic", "s1"),
    ]

    recalls = [e for e in replayed if e["event"] == "memory_recall"]
    assert [e["injected"] for e in recalls] == [False, True]

    summary = audit_log.summary()
    assert summary["total"] == 16
    assert summary["by_event"]["turn"] == 2
    assert summary["by_event"]["rule_fired"] == 2

    with open(audit_log.path, encoding="utf-8") as f:
        assert "4111" not in f.read()


# --- 11. recall block injected into the instruction ------------------------------
def test_eval_recall_injected_into_instruction(banks):
    async def go():
        session = Session(
            app_name="test_app",
            user_id="test_user",
            id="s1",
            state={},
            events=[_event("user", "ORD-1001 was late")],
        )
        await banks.episodic.record_session(session)
        banks.semantic.upsert("contact_preference", "prefers email updates")

        state: dict = {}
        await agent_module.ensure_memory_recall(
            state, "test_app", "test_user", "late order preferences"
        )
        with_memory = await agent_module._instruction_provider(_FakeCtx(state))
        without_memory = await agent_module._instruction_provider(_FakeCtx({}))
        return with_memory, without_memory

    with_memory, without_memory = asyncio.run(go())
    assert "MEMORY -- things remembered" in with_memory
    assert "prefers email updates" in with_memory
    assert "ORD-1001" in with_memory
    assert "MEMORY -- things remembered" not in without_memory
    assert f"${config.REFUND_APPROVAL_THRESHOLD:.2f}" in without_memory


# --- 12. audit previews scrub PII -------------------------------------------------
def test_eval_audit_previews_scrub_pii(audit_log):
    audit_module.log_turn(
        "test_app", "test_user", "s1", "my card is 4111 1111 1111 1111 please"
    )
    audit_module.log_tool_call(
        "test_app",
        "test_user",
        "s1",
        "remember_fact",
        {"fact": "card 4111111111111111", "slot": "payment"},
    )
    dumped = json.dumps(audit_log.events)
    assert "4111" not in dumped
    assert dumped.count("[REDACTED]") >= 2
    assert len(audit_log.events[0]["prompt_hash"]) == 16
