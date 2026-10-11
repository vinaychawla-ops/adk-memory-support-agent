"""Durable memory backends (Phase 2) -- the video's Memory Handling panel.

Each backend sits behind an ADK seam, so the agent code never changes to
accommodate one:

- 2a session state + conversation history -> RedisSessionService
  (BaseSessionService), selected by SESSION_BACKEND=redis
- 2b episodic long-term                 -> PgVectorMemoryService
  (BaseMemoryService), selected by EPISODIC_BACKEND=pgvector
- 2c semantic long-term                 -> FirestoreSemanticBank
  (SemanticBank-compatible), selected by SEMANTIC_BACKEND=firestore
- 2d procedural long-term               -> rules/policy.yaml +
  rules/procedural.md, versioned in git (see git_rules.py)

Defaults are the in-memory services: everything still runs with zero
infrastructure. Imports are lazy (inside the factory functions) so that
config.py can import git_rules at module load without import cycles, and
so importing this package never requires redis/psycopg/firestore libs.
"""

from __future__ import annotations


def make_session_service():
    """ADK session service per SESSION_BACKEND (memory | redis)."""
    from support_agent import config

    if config.SESSION_BACKEND == "redis":
        import redis

        from support_agent.backends.redis_sessions import RedisSessionService

        return RedisSessionService(
            # Short timeouts: only matter when the server is down, in which
            # case probes fail fast (tests skip) instead of hanging.
            redis.Redis.from_url(
                config.REDIS_URL, socket_connect_timeout=3, socket_timeout=3
            ),
            ttl_seconds=config.SESSION_TTL_SECONDS,
        )
    from google.adk.sessions import InMemorySessionService

    return InMemorySessionService()


def make_memory_service():
    """ADK memory service per EPISODIC_BACKEND (memory | pgvector)."""
    from support_agent import config

    if config.EPISODIC_BACKEND == "pgvector":
        from support_agent.backends.pgvector_memory import (
            FakeEmbedder,
            GeminiEmbedder,
            PgVectorEpisodeStore,
            PgVectorMemoryService,
        )

        dim = config.EMBEDDING_DIM
        embedder = (
            GeminiEmbedder(model=config.EMBEDDING_MODEL, dim=dim)
            if config.EMBEDDER == "gemini"
            else FakeEmbedder(dim=dim)
        )
        return PgVectorMemoryService(
            PgVectorEpisodeStore(config.PGVECTOR_DSN, dim), embedder
        )
    from google.adk.memory import InMemoryMemoryService

    return InMemoryMemoryService()


def make_semantic_store(app_name: str, user_id: str):
    """Semantic fact store per SEMANTIC_BACKEND (memory | firestore)."""
    from support_agent import config

    if config.SEMANTIC_BACKEND == "firestore":
        from google.cloud import firestore

        from support_agent.backends.firestore_semantics import (
            FirestoreSemanticBank,
        )

        # Honors FIRESTORE_EMULATOR_HOST natively: local Docker vs. real
        # Firestore is a pure connection swap.
        db = firestore.Client(project=config.FIRESTORE_PROJECT)
        return FirestoreSemanticBank(db, app_name, user_id)
    from support_agent.memory_banks import SemanticBank

    return SemanticBank()


def make_embedder():
    """Embedder per EMBEDDER (fake | gemini). Offline-safe by default."""
    from support_agent import config
    from support_agent.backends.pgvector_memory import FakeEmbedder, GeminiEmbedder

    if config.EMBEDDER == "gemini":
        return GeminiEmbedder(model=config.EMBEDDING_MODEL, dim=config.EMBEDDING_DIM)
    return FakeEmbedder(dim=config.EMBEDDING_DIM)
