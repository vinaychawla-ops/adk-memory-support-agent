"""Tools: mock OMS behavior, state writes, and error-shaped failures."""

import asyncio

import pytest

from support_agent import tools


def test_lookup_order_sets_active_order_state(ctx):
    r = tools.lookup_order("ORD-1001", ctx)
    assert r["order_id"] == "ORD-1001"
    assert r["days_late"] == 5
    assert r["customer_name"] == "Maya Chen"
    assert ctx.state["active_order_id"] == "ORD-1001"


def test_lookup_order_rejects_bad_id(ctx):
    assert "error" in tools.lookup_order("nope", ctx)
    assert "error" in tools.lookup_order("ORD-9999", ctx)
    assert "active_order_id" not in ctx.state


def test_track_order_timeline(ctx):
    r = tools.track_order("ORD-1003", ctx)
    assert r["status"] == "in_transit"
    assert len(r["tracking_events"]) == 2
    assert "error" in tools.track_order("ORD-0000", ctx)


def test_start_return_happy_path(ctx):
    r = tools.start_return("ORD-1002", "changed my mind", ctx)
    assert r["status"] == "return_started"
    assert r["return_id"].startswith("RET-")
    assert ctx.state["return_in_progress"]["order_id"] == "ORD-1002"


def test_start_return_outside_window_refuses(ctx):
    r = tools.start_return("ORD-1004", "too late", ctx)
    assert "error" in r
    assert "30-day" in r["rule_reason"]
    assert "return_in_progress" not in ctx.state


def test_start_return_not_delivered_refuses(ctx):
    r = tools.start_return("ORD-1003", "impatient", ctx)
    assert "error" in r
    assert "in_transit" in r["rule_reason"]


def test_get_return_policy(ctx):
    assert "30 days" in tools.get_return_policy()["policy"]


def test_request_refund_small_amount_approved(ctx):
    r = tools.request_refund("ORD-1001", 10.0, "late delivery", ctx)
    assert r["status"] == "approved"
    assert r["credit_id"].startswith("CR-")
    assert r["amount"] == 10.0


def test_request_refund_large_amount_drafted_not_executed(ctx):
    r = tools.request_refund("ORD-1002", 75.0, "changed mind", ctx)
    assert r["status"] == "drafted_for_approval"
    assert r["escalation_ticket"].startswith("TKT-")
    assert "draft" in r
    assert "credit_id" not in r  # nothing was executed


def test_request_refund_rejects_bad_amount(ctx):
    assert "error" in tools.request_refund("ORD-1001", -5, "x", ctx)
    assert "error" in tools.request_refund("ORD-1001", "lots", "x", ctx)


def test_escalate_to_human(ctx):
    r = tools.escalate_to_human("legal threat", "Customer mentioned a lawyer.", ctx)
    assert r["status"] == "escalated"
    assert r["ticket_id"].startswith("TKT-")
    assert "error" in tools.escalate_to_human("", "x", ctx)


def test_remember_fact_stores_and_supersedes(ctx, banks):
    r1 = tools.remember_fact("prefers email updates", "contact_preference", ctx)
    assert r1["stored"] is True and r1["superseded"] is False
    r2 = tools.remember_fact("prefers SMS updates", "contact_preference", ctx)
    assert r2["superseded"] is True
    active = banks.semantic.active_facts()
    assert len(active) == 1 and active[0].text == "prefers SMS updates"


def test_remember_fact_rejects_card_numbers(ctx, banks):
    r = tools.remember_fact("card 4111 1111 1111 1111", "payment", ctx)
    assert "error" in r
    assert banks.semantic.active_facts() == []


def test_remember_fact_without_banks_errors():
    from tests.conftest import FakeToolContext

    r = tools.remember_fact("x", "y", FakeToolContext("nope", "nobody"))
    assert "error" in r


def test_recall_memories_returns_block(ctx, banks):
    banks.semantic.upsert("contact_preference", "prefers email updates")
    r = asyncio.run(tools.recall_memories("contact", ctx))
    assert "prefers email updates" in r["recall"]


def test_build_tools_stable_order():
    names = [t.name for t in tools.build_tools()]
    assert names == [
        "lookup_order",
        "track_order",
        "start_return",
        "get_return_policy",
        "request_refund",
        "escalate_to_human",
        "remember_fact",
        "recall_memories",
    ]
