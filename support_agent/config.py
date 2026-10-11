"""Central configuration: model, app identity, and policy thresholds.

All policy numbers live here. Never hardcode a threshold in rules.py,
tools.py, or the agent instruction -- import it from here so a policy
change is a one-line diff.
"""

APP_NAME = "memory_support_agent"

# Gemini model used for the live agent. The key comes from the environment
# (GOOGLE_API_KEY); nothing in this package reads a key itself.
MODEL = "gemini-2.5-flash"

# --- Procedural policy (the deterministic rules layer) ----------------------
# Refunds / goodwill credits at or below this amount may be issued by the
# mock OMS directly. Anything above is drafted and routed to a human --
# the agent never auto-executes it.
REFUND_APPROVAL_THRESHOLD = 50.0

# Returns are accepted within this many days of delivery.
RETURN_WINDOW_DAYS = 30

# Goodwill credits at or below this amount may be issued directly for
# service failures (e.g. late delivery). Sits under REFUND_APPROVAL_THRESHOLD.
GOODWILL_CREDIT_MAX = 25.0

# How many past episodes / facts the recall block carries at most.
MAX_EPISODIC_HITS = 5
MAX_SEMANTIC_FACTS = 10
