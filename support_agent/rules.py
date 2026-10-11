"""Deterministic procedural rules (the checks.py pattern).

Pure functions only: no I/O, no network, no LLM. Every rule returns a
dict with a "rule" key; malformed input returns {"rule", "error"}.
Thresholds come from config.py -- never hardcode one here.

These rules ARE the agent's procedural memory: how it must behave,
independent of what it remembers about any customer.
"""

from __future__ import annotations

from datetime import date

from support_agent import config
from support_agent import fixtures


def evaluate_refund_request(amount: float, order: dict | None = None) -> dict:
    """Decide whether a refund/credit may be issued directly or needs a human.

    - amount <= REFUND_APPROVAL_THRESHOLD: the mock OMS may issue it.
    - amount >  REFUND_APPROVAL_THRESHOLD: draft the request and escalate;
      the agent must never execute it itself.
    """
    if not isinstance(amount, (int, float)) or amount <= 0:
        return {"rule": "refund_approval", "error": "amount must be a positive number"}
    if order is not None and not isinstance(order, dict):
        return {"rule": "refund_approval", "error": "order must be a dict or None"}

    if amount <= config.REFUND_APPROVAL_THRESHOLD:
        return {
            "rule": "refund_approval",
            "decision": "approve",
            "reason": (
                f"${amount:.2f} is at or below the ${config.REFUND_APPROVAL_THRESHOLD:.2f} "
                "auto-approval limit."
            ),
        }
    return {
        "rule": "refund_approval",
        "decision": "draft_for_approval",
        "reason": (
            f"${amount:.2f} exceeds the ${config.REFUND_APPROVAL_THRESHOLD:.2f} limit: "
            "prepare the request and escalate to a human supervisor. "
            "Do NOT execute the refund."
        ),
    }


def check_return_eligibility(order: dict, today: date | None = None) -> dict:
    """A return is eligible only if delivered and within the return window."""
    if not isinstance(order, dict) or "order_id" not in order:
        return {"rule": "return_eligibility", "error": "order must be an order dict"}
    today = today or fixtures.TODAY

    if order.get("status") != "delivered":
        return {
            "rule": "return_eligibility",
            "eligible": False,
            "reason": f"Order {order['order_id']} has status "
            f"'{order.get('status')}' -- only delivered orders can be returned.",
        }
    age = fixtures.days_since_delivery(order, today)
    if age is None:
        return {
            "rule": "return_eligibility",
            "eligible": False,
            "reason": "No delivery date on record; cannot verify the return window.",
        }
    if age > config.RETURN_WINDOW_DAYS:
        return {
            "rule": "return_eligibility",
            "eligible": False,
            "reason": (
                f"Delivered {age} days ago -- outside the "
                f"{config.RETURN_WINDOW_DAYS}-day return window. Escalate instead."
            ),
        }
    return {
        "rule": "return_eligibility",
        "eligible": True,
        "reason": f"Delivered {age} days ago -- within the return window.",
    }


def should_escalate(signals: dict) -> dict:
    """Escalation triggers. signals keys (all optional booleans/ints):

    - legal_threat: customer mentions lawyer / regulator / chargeback threat
    - repeat_contacts: number of prior contacts about the same issue
    - vip_high_value: VIP customer AND amount above the refund threshold
    - agent_unsure: the agent cannot resolve with available tools
    """
    if not isinstance(signals, dict):
        return {"rule": "escalation", "error": "signals must be a dict"}

    reasons: list[str] = []
    if signals.get("legal_threat"):
        reasons.append("Customer raised a legal/regulatory threat.")
    if isinstance(signals.get("repeat_contacts"), int) and signals["repeat_contacts"] >= 3:
        reasons.append(
            f"Customer has contacted {signals['repeat_contacts']} times about the same issue."
        )
    if signals.get("vip_high_value"):
        reasons.append("VIP customer with a high-value request.")
    if signals.get("agent_unsure"):
        reasons.append("The agent cannot resolve this with available tools.")

    return {
        "rule": "escalation",
        "escalate": bool(reasons),
        "reasons": reasons,
    }


def late_delivery_credit(order: dict) -> dict:
    """Goodwill credit for a late delivery, capped by policy."""
    if not isinstance(order, dict) or "order_id" not in order:
        return {"rule": "late_delivery_credit", "error": "order must be an order dict"}
    late = fixtures.days_late(order)
    if late <= 0:
        return {
            "rule": "late_delivery_credit",
            "credit": 0.0,
            "reason": "Order was not delivered late; no goodwill credit applies.",
        }
    # $5 per late day, capped at the goodwill max (which sits under the
    # refund approval threshold, so it stays auto-approvable).
    credit = min(5.0 * late, config.GOODWILL_CREDIT_MAX)
    return {
        "rule": "late_delivery_credit",
        "credit": round(credit, 2),
        "reason": f"Delivered {late} days late: ${credit:.2f} goodwill credit.",
    }
