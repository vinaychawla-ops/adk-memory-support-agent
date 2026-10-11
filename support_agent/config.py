"""Central configuration: model, app identity, policy, and backends.

Policy thresholds live in rules/policy.yaml (the git-versioned source of
truth for procedural memory) and are loaded + validated at import -- see
support_agent/backends/git_rules.py. Never hardcode a threshold in
rules.py, tools.py, or the agent instruction; import it from here.

Backend selection is env-driven (Phase 2). Defaults keep everything
in-memory so the agent runs with zero infrastructure:
  SESSION_BACKEND=memory|redis        (session state + history)
  EPISODIC_BACKEND=memory|pgvector     (episodic long-term)
  SEMANTIC_BACKEND=memory|firestore    (semantic long-term)
  EMBEDDER=fake|gemini                 (pgvector embeddings; fake is offline)
"""

import os

from support_agent.backends import git_rules

APP_NAME = "memory_support_agent"

# Gemini model used for the live agent. The key comes from the environment
# (GOOGLE_API_KEY); nothing in this package reads a key itself.
MODEL = "gemini-2.5-flash"

# --- Procedural policy (loaded from rules/policy.yaml, validated) ------------
POLICY = git_rules.load_policy()

# Refunds / goodwill credits at or below this amount may be issued by the
# mock OMS directly. Anything above is drafted and routed to a human --
# the agent never auto-executes it.
REFUND_APPROVAL_THRESHOLD = float(POLICY["refund_approval_threshold"])

# Returns are accepted within this many days of delivery.
RETURN_WINDOW_DAYS = int(POLICY["return_window_days"])

# Goodwill credits at or below this amount may be issued directly for
# service failures (e.g. late delivery). Sits under REFUND_APPROVAL_THRESHOLD.
GOODWILL_CREDIT_MAX = float(POLICY["goodwill_credit_max"])

# How many past episodes / facts the recall block carries at most.
MAX_EPISODIC_HITS = int(POLICY["max_episodic_hits"])
MAX_SEMANTIC_FACTS = int(POLICY["max_semantic_facts"])

# --- Durable backends (Phase 2) ------------------------------------------------
SESSION_BACKEND = os.environ.get("SESSION_BACKEND", "memory")
EPISODIC_BACKEND = os.environ.get("EPISODIC_BACKEND", "memory")
SEMANTIC_BACKEND = os.environ.get("SEMANTIC_BACKEND", "memory")
EMBEDDER = os.environ.get("EMBEDDER", "fake")

# 2a Redis: sliding TTL per session (30 days); refreshed on every write.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(30 * 24 * 3600)))

# 2b pgvector: episodic rows embedded with EMBEDDING_MODEL at EMBEDDING_DIM.
# Keep EMBEDDING_DIM stable once the table exists (it sizes the column).
PGVECTOR_DSN = os.environ.get(
    "PGVECTOR_DSN", "postgresql://memory:memory@localhost:5432/memory"
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

# 2c Firestore: semantic facts. The client honors FIRESTORE_EMULATOR_HOST
# natively -- point it at the local emulator for Docker, unset it (plus
# real credentials) for cloud. Pure connection swap, no code change.
FIRESTORE_PROJECT = os.environ.get("FIRESTORE_PROJECT", "demo-memory-agent")
