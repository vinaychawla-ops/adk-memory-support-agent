"""Integration tests against the real Phase-2 backends in local Docker.

Each test self-skips when its backend is not running -- the offline suite
stays green with zero infrastructure. Run them with:

  docker compose up -d            # redis :6379, pgvector :5432, firestore :8080
  V=~/workspace/.venv-adk/bin
  $V/python -m pytest tests/ -q -m integration

Or the whole suite (unit + integration):
  SESSION_BACKEND=redis EPISODIC_BACKEND=pgvector SEMANTIC_BACKEND=firestore \\
    $V/python -m pytest tests/ -q

To prove the kill/restart acceptance by hand:
  docker compose restart redis pgvector   # firestore emulator is ephemeral by design
  $V/python -m pytest tests/ -q -m integration   # long-term memory must survive

Note: pgvector tests use FakeEmbedder (no GOOGLE_API_KEY needed); set
EMBEDDER=gemini for real Gemini embeddings.
"""

import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.integration

from support_agent import config
from support_agent.backends.pgvector_memory import (
    FakeEmbedder,
    PgVectorEpisodeStore,
    PgVectorMemoryService,
)
from support_agent.backends.redis_sessions import RedisSessionService
from support_agent.memory_banks import EpisodicBank, MemoryBanks

from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types


def _event(author: str, text: str, event_id: str) -> Event:
    return Event(
        id=event_id,
        author=author,
        content=types.Content(
            role="user" if author == "user" else "model",
            parts=[types.Part(text=text)],
        ),
    )


def _session(app: str, user: str, sid: str, *texts: str) -> Session:
    return Session(
        app_name=app,
        user_id=user,
        id=sid,
        state={},
        events=[_event("user", t, f"{sid}-{i}") for i, t in enumerate(texts)],
    )


def run(coro):
    return asyncio.run(coro)


def _uid() -> str:
    return "itest_" + uuid.uuid4().hex[:8]


# --- Redis ----------------------------------------------------------------------


def _redis_service():
    import redis

    try:
        client = redis.Redis.from_url(
            "redis://localhost:6379/0", socket_connect_timeout=3, socket_timeout=3
        )
        client.ping()
    except Exception as e:
        pytest.skip(f"redis not running (docker compose up -d): {e}")
    return RedisSessionService(client, ttl_seconds=config.SESSION_TTL_SECONDS)


def test_integration_redis_roundtrip_and_replica():
    svc1, svc2, user = _redis_service(), _redis_service(), _uid()

    async def go():
        s = await svc1.create_session(app_name="a", user_id=user)
        event = _event("user", "track my order", "e1")
        event.actions.state_delta = {"active_order_id": "ORD-1001"}
        await svc1.append_event(s, event)
        # A different service instance over the same server (any replica).
        got = await svc2.get_session(app_name="a", user_id=user, session_id=s.id)
        assert got.state["active_order_id"] == "ORD-1001"
        assert len(got.events) == 1
        await svc1.delete_session(app_name="a", user_id=user, session_id=s.id)
        assert await svc2.get_session(app_name="a", user_id=user, session_id=s.id) is None

    run(go())


def test_integration_redis_survives_reconnect():
    """Client reconnect (and `docker compose restart redis`) loses nothing:
    data lives server-side."""
    user = _uid()

    async def go():
        svc1 = _redis_service()
        s = await svc1.create_session(
            app_name="a", user_id=user, state={"active_order_id": "ORD-1002"}
        )
        del svc1  # drop the client entirely
        svc2 = _redis_service()  # fresh connection, same server
        got = await svc2.get_session(app_name="a", user_id=user, session_id=s.id)
        assert got.state["active_order_id"] == "ORD-1002"
        await svc2.delete_session(app_name="a", user_id=user, session_id=s.id)

    run(go())


# --- pgvector ----------------------------------------------------------------------


def _pg_service(user_dim: int | None = None):
    store = PgVectorEpisodeStore(config.PGVECTOR_DSN, user_dim or config.EMBEDDING_DIM)
    try:
        store.ping()
    except Exception as e:
        pytest.skip(f"pgvector not running (docker compose up -d): {e}")
    return PgVectorMemoryService(store, FakeEmbedder(dim=user_dim or config.EMBEDDING_DIM))


def test_integration_pgvector_add_and_search():
    svc, user = _pg_service(), _uid()

    async def go():
        await svc.add_session_to_memory(
            _session("a", user, "s1", "my order ORD-1001 arrived 5 days late")
        )
        resp = await svc.search_memory(app_name="a", user_id=user, query="late delivery")
        assert any("ORD-1001" in m.content.parts[0].text for m in resp.memories)
        # Another user sees nothing.
        resp2 = await svc.search_memory(app_name="a", user_id=user + "_other", query="late")
        assert resp2.memories == []

    run(go())


def test_integration_pgvector_scrubs_pii():
    svc, user = _pg_service(), _uid()

    async def go():
        await svc.add_session_to_memory(
            _session("a", user, "s9", "my card is 4111 1111 1111 1111 please")
        )
        resp = await svc.search_memory(app_name="a", user_id=user, query="card")
        texts = [m.content.parts[0].text for m in resp.memories]
        assert texts, "the scrubbed episode should still be found"
        assert all("4111" not in t for t in texts)

    run(go())


# --- Firestore (emulator) ----------------------------------------------------------------------


def _firestore_bank(user: str):
    os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"
    from google.cloud import firestore

    from support_agent.backends.firestore_semantics import FirestoreSemanticBank

    try:
        db = firestore.Client(project="demo-memory-agent")
        bank = FirestoreSemanticBank(db, "test_app", user)
        bank.ping()
    except Exception as e:
        pytest.skip(f"firestore emulator not running (docker compose up -d): {e}")
    return bank


def test_integration_firestore_upsert_and_supersede():
    user = _uid()
    bank = _firestore_bank(user)
    assert bank.upsert("contact_preference", "prefers email updates")["stored"] is True
    assert bank.upsert("contact_preference", "prefers SMS updates")["superseded"] is True
    # A second bank instance over the same emulator sees the writes.
    bank2 = _firestore_bank(user)
    assert [f.text for f in bank2.active_facts()] == ["prefers SMS updates"]
    assert len(bank2.history()) == 2


# --- combined: the full recall block over real backends ----------------------------------------------------------------------


def test_integration_combined_recall_block():
    """Episodic (pgvector) + semantic (firestore) feed one recall block."""
    user = _uid()
    pg = _pg_service()
    fs = _firestore_bank(user)
    banks = MemoryBanks(pg, "test_app", user, semantic=fs)

    async def go():
        banks.semantic.upsert("contact_preference", "prefers email updates")
        await banks.episodic.record_session(
            _session("test_app", user, "s1", "ORD-1001 arrived 5 days late")
        )
        return await banks.build_recall_block("late order preferences")

    block = run(go())
    assert "prefers email updates" in block
    assert "ORD-1001" in block
