"""Mock Order Management System tools + memory tools.

Module-level, annotated, docstringed functions returning JSON-serializable
dicts -- ADK builds FunctionTool schemas from them. Every failure returns
{"error": ...}, never a traceback. Signatures are the public contract:
agent.py and demo.py depend on them, so keep them stable.

Session state discipline: tools write STRUCTURED execution state only
(active_order_id, return_in_progress) -- never raw transcript. The
conversation history itself comes from session events.
"""

from __future__ import annotations

import re
from typing import Any

from google.adk.tools import FunctionTool, ToolContext

from support_agent import config, fixtures, rules
from support_agent.audit import log_memory_write, log_rule_fired
from support_agent.memory_banks import get_banks

_ORDER_ID_RE = re.compile(r"^ORD-\d{4}$")


def _audit_ids(tool_context) -> tuple[str, str, str | None]:
    """(app_name, user_id, session_id) from a ToolContext, defensively."""
    try:
        session = tool_context.session
        return (
            session.app_name,
            session.user_id,
            getattr(session, "id", None),
        )
    except Exception:
        return "", "", None


def _banks_or_error(tool_context: ToolContext) -> tuple[Any | None, dict | None]:
    banks = get_banks(
        tool_context.session.app_name, tool_context.session.user_id
    )
    if banks is None:
        return None, {"error": "Memory is not initialized for this user."}
    return banks, None


def _id_error(order_id: str) -> dict | None:
    if not _ORDER_ID_RE.match(order_id or ""):
        return {"error": f"Invalid order id {order_id!r}: expected like 'ORD-1001'."}
    return None


def lookup_order(order_id: str, tool_context: ToolContext) -> dict:
    """Look up an order by id. Records it as the active order in session state."""
    if err := _id_error(order_id):
        return err
    order = fixtures.get_order(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id!r}."}
    tool_context.state["active_order_id"] = order_id
    customer = fixtures.get_customer(order["customer_id"]) or {}
    return {
        "order_id": order["order_id"],
        "status": order["status"],
        "customer_name": customer.get("name"),
        "total": order["total"],
        "items": order["items"],
        "promised_date": str(order["promised_date"]),
        "delivered_date": str(order["delivered_date"]) if order["delivered_date"] else None,
        "days_late": fixtures.days_late(order),
    }


def track_order(order_id: str, tool_context: ToolContext) -> dict:
    """Get tracking status and the event timeline for an order."""
    if err := _id_error(order_id):
        return err
    order = fixtures.get_order(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id!r}."}
    return {
        "order_id": order_id,
        "status": order["status"],
        "tracking_events": [
            {"date": d, "event": e} for d, e in order["tracking_events"]
        ],
    }


def start_return(order_id: str, reason: str, tool_context: ToolContext) -> dict:
    """Start a return. Refuses ineligible orders with the rule's reason."""
    if err := _id_error(order_id):
        return err
    order = fixtures.get_order(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id!r}."}
    verdict = rules.check_return_eligibility(order)
    app_name, user_id, session_id = _audit_ids(tool_context)
    if verdict.get("error"):
        decision = "error"
    elif verdict.get("eligible"):
        decision = "eligible"
    else:
        decision = "ineligible"
    log_rule_fired(
        app_name,
        user_id,
        session_id,
        "return_eligibility",
        decision,
        verdict.get("reason", verdict.get("error", "")),
    )
    if verdict.get("error") or not verdict.get("eligible"):
        return {
            "error": "Return not eligible.",
            "rule_reason": verdict.get("reason", verdict.get("error")),
        }
    return_id = fixtures.new_return_id()
    tool_context.state["return_in_progress"] = {
        "return_id": return_id,
        "order_id": order_id,
        "reason": reason,
    }
    tool_context.state["active_order_id"] = order_id
    return {
        "return_id": return_id,
        "order_id": order_id,
        "status": "return_started",
        "next_step": "Customer ships the item back; refund issues on receipt.",
    }


def get_return_policy() -> dict:
    """Return the store's return policy text."""
    return {"policy": fixtures.RETURN_POLICY}


def request_refund(
    order_id: str, amount: float, reason: str, tool_context: ToolContext
) -> dict:
    """Request a refund/credit. <= $50 issues via the mock OMS; above that
    the request is DRAFTED and escalated -- never auto-executed."""
    if err := _id_error(order_id):
        return err
    order = fixtures.get_order(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id!r}."}
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return {"error": "amount must be a number"}

    verdict = rules.evaluate_refund_request(amount, order)
    app_name, user_id, session_id = _audit_ids(tool_context)
    if verdict.get("error"):
        log_rule_fired(
            app_name, user_id, session_id,
            "refund_approval", "error", verdict.get("error", ""),
        )
        return verdict
    log_rule_fired(
        app_name,
        user_id,
        session_id,
        "refund_approval",
        verdict["decision"],
        verdict["reason"],
    )

    if verdict["decision"] == "approve":
        credit_id = fixtures.new_credit_id()
        return {
            "status": "approved",
            "credit_id": credit_id,
            "order_id": order_id,
            "amount": round(amount, 2),
            "rule_reason": verdict["reason"],
        }

    # Above the threshold: draft + human ticket, no execution.
    ticket_id = fixtures.new_ticket_id()
    return {
        "status": "drafted_for_approval",
        "order_id": order_id,
        "amount": round(amount, 2),
        "draft": (
            f"Refund request: ${amount:.2f} on {order_id} ({reason}). "
            "Prepared by support agent; awaiting supervisor approval."
        ),
        "escalation_ticket": ticket_id,
        "rule_reason": verdict["reason"],
    }


def escalate_to_human(reason: str, summary: str, tool_context: ToolContext) -> dict:
    """Hand the conversation to a human agent with a summary."""
    if not (reason or "").strip():
        return {"error": "reason is required"}
    ticket_id = fixtures.new_ticket_id()
    return {
        "ticket_id": ticket_id,
        "status": "escalated",
        "reason": reason,
        "summary": summary,
    }


def remember_fact(fact: str, slot: str, tool_context: ToolContext) -> dict:
    """Store a durable customer fact/preference (semantic memory).

    Call this when the customer states something that should hold across
    sessions: contact preferences, shipping preferences, etc. A new fact
    for an existing slot supersedes the old one.
    """
    banks, err = _banks_or_error(tool_context)
    if err:
        return err
    result = banks.semantic.upsert(slot=slot, text=fact)
    app_name, user_id, _ = _audit_ids(tool_context)
    log_memory_write(
        app_name,
        user_id,
        "semantic",
        {
            "slot": slot,
            "stored": result.get("stored", False),
            "superseded": result.get("superseded", False),
            "error": result.get("error"),
        },
    )
    return result


async def recall_memories(query: str, tool_context: ToolContext) -> dict:
    """Search long-term memory: durable facts + relevant past interactions."""
    banks, err = _banks_or_error(tool_context)
    if err:
        return err
    block = await banks.build_recall_block(query or "")
    return {"recall": block}


def build_tools() -> list[FunctionTool]:
    """All tools, in a stable order."""
    return [
        FunctionTool(lookup_order),
        FunctionTool(track_order),
        FunctionTool(start_return),
        FunctionTool(get_return_policy),
        FunctionTool(request_refund),
        FunctionTool(escalate_to_human),
        FunctionTool(remember_fact),
        FunctionTool(recall_memories),
    ]
