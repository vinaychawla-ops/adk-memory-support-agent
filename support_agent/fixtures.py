"""Mock Order Management System data. Everything here is fictional.

Deterministic by design: TODAY is fixed so return-window and lateness
math never depends on the wall clock, and id counters reset for tests.
No real PII anywhere in this module.
"""

from __future__ import annotations

from datetime import date
from itertools import count

# Fixed "today" keeps every date computation deterministic in tests/demos.
TODAY = date(2026, 10, 11)

CUSTOMERS = {
    "CUST-1001": {
        "customer_id": "CUST-1001",
        "name": "Maya Chen",
        "email": "maya.chen@example.com",
        "tier": "standard",
    },
    "CUST-1002": {
        "customer_id": "CUST-1002",
        "name": "Dev Patel",
        "email": "dev.patel@example.com",
        "tier": "vip",
    },
}

ORDERS = {
    # Delivered 5 days late: promised 2026-10-01, arrived 2026-10-06.
    "ORD-1001": {
        "order_id": "ORD-1001",
        "customer_id": "CUST-1001",
        "status": "delivered",
        "items": [{"sku": "SKU-HEADPHONES", "name": "Aurora Headphones", "qty": 1}],
        "total": 89.99,
        "promised_date": date(2026, 10, 1),
        "delivered_date": date(2026, 10, 6),
        "tracking_events": [
            ("2026-09-29", "Label created"),
            ("2026-10-01", "Departed fulfillment center"),
            ("2026-10-04", "Arrived at regional hub (delayed)"),
            ("2026-10-06", "Delivered"),
        ],
    },
    # Delivered on time.
    "ORD-1002": {
        "order_id": "ORD-1002",
        "customer_id": "CUST-1001",
        "status": "delivered",
        "items": [{"sku": "SKU-KEYBOARD", "name": "Drift Keyboard", "qty": 1}],
        "total": 149.99,
        "promised_date": date(2026, 10, 9),
        "delivered_date": date(2026, 10, 9),
        "tracking_events": [
            ("2026-10-07", "Label created"),
            ("2026-10-08", "Departed fulfillment center"),
            ("2026-10-09", "Delivered"),
        ],
    },
    # In transit.
    "ORD-1003": {
        "order_id": "ORD-1003",
        "customer_id": "CUST-1001",
        "status": "in_transit",
        "items": [{"sku": "SKU-MOUSE", "name": "Nomad Mouse", "qty": 2}],
        "total": 59.98,
        "promised_date": date(2026, 10, 14),
        "delivered_date": None,
        "tracking_events": [
            ("2026-10-10", "Label created"),
            ("2026-10-11", "Departed fulfillment center"),
        ],
    },
    # Delivered long ago: outside the 30-day return window on TODAY.
    "ORD-1004": {
        "order_id": "ORD-1004",
        "customer_id": "CUST-1002",
        "status": "delivered",
        "items": [{"sku": "SKU-MONITOR", "name": "Vista Monitor 27in", "qty": 1}],
        "total": 329.00,
        "promised_date": date(2026, 8, 1),
        "delivered_date": date(2026, 8, 1),
        "tracking_events": [("2026-08-01", "Delivered")],
    },
}

RETURN_POLICY = """\
Returns are accepted within 30 days of delivery for unopened items in
original packaging. Refunds go to the original payment method within
5-10 business days. Goodwill credits up to $25 may be issued directly
by support for service failures (e.g. late delivery). Refund or credit
amounts above $50 require a human supervisor's approval -- support may
prepare the request but never executes it. Shipping fees are
non-refundable unless the error was ours.
""".strip()

# Deterministic id counters (reset in tests).
_ticket_counter = count(1)
_return_counter = count(1)
_credit_counter = count(1)


def reset_counters() -> None:
    """Reset id counters -- tests only, keeps ids deterministic."""
    global _ticket_counter, _return_counter, _credit_counter
    _ticket_counter = count(1)
    _return_counter = count(1)
    _credit_counter = count(1)


def new_ticket_id() -> str:
    return f"TKT-{next(_ticket_counter):04d}"


def new_return_id() -> str:
    return f"RET-{next(_return_counter):04d}"


def new_credit_id() -> str:
    return f"CR-{next(_credit_counter):04d}"


def get_order(order_id: str) -> dict | None:
    return ORDERS.get(order_id)


def get_customer(customer_id: str) -> dict | None:
    return CUSTOMERS.get(customer_id)


def days_late(order: dict) -> int:
    """Days between promised and actual delivery (0 if on time / not delivered)."""
    promised = order.get("promised_date")
    delivered = order.get("delivered_date")
    if not promised or not delivered:
        return 0
    return max(0, (delivered - promised).days)


def days_since_delivery(order: dict, today: date = TODAY) -> int | None:
    delivered = order.get("delivered_date")
    if not delivered:
        return None
    return (today - delivered).days
