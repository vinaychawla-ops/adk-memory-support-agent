"""Shared fakes for the offline suite."""

import pytest

from support_agent import config, fixtures
from support_agent.memory_banks import clear_registry, register_banks


class FakeSession:
    def __init__(self, app_name="test_app", user_id="test_user"):
        self.app_name = app_name
        self.user_id = user_id


class FakeToolContext:
    """Minimal stand-in for ADK ToolContext: .state dict + .session."""

    def __init__(self, app_name="test_app", user_id="test_user"):
        self.state: dict = {}
        self.session = FakeSession(app_name, user_id)


def _ensure_reachable(memory_service, semantic_store) -> None:
    """Skip the test when an env-selected backend is not running.

    With default (in-memory) backends this is a no-op, so the offline
    suite is unaffected. Lets the whole eval set run against real
    backends via:
      SESSION_BACKEND=redis EPISODIC_BACKEND=pgvector SEMANTIC_BACKEND=firestore \
        pytest tests/test_eval.py -q
    """
    from support_agent.backends.firestore_semantics import FirestoreSemanticBank
    from support_agent.backends.pgvector_memory import PgVectorMemoryService

    if isinstance(semantic_store, FirestoreSemanticBank):
        try:
            semantic_store.ping()
        except Exception as e:
            pytest.skip(f"firestore backend unreachable: {e}")
    if isinstance(memory_service, PgVectorMemoryService):
        import asyncio

        try:
            asyncio.run(memory_service.ping())
        except Exception as e:
            pytest.skip(f"pgvector backend unreachable: {e}")


@pytest.fixture()
def ctx():
    clear_registry()
    fixtures.reset_counters()
    return FakeToolContext()


@pytest.fixture()
def banks():
    """Registered MemoryBanks via the backend factories (env-selected).

    Defaults to the in-memory services, so the offline suite is unchanged;
    with EPISODIC_BACKEND/SEMANTIC_BACKEND set it exercises the real
    backends (skipping cleanly when they are not running).
    """
    from support_agent.backends import make_memory_service, make_semantic_store

    clear_registry()
    memory_service = make_memory_service()
    try:
        semantic_store = make_semantic_store("test_app", "test_user")
    except Exception as e:
        # e.g. firestore.Client() with no emulator and no credentials.
        pytest.skip(f"semantic backend unavailable: {e}")
    _ensure_reachable(memory_service, semantic_store)
    return register_banks(
        "test_app", "test_user", memory_service, semantic=semantic_store
    )
