"""Unit tests for the pgvector episodic backend (2b).

FakeEmbedder (deterministic, offline) + FakeEpisodeStore (real cosine
math in memory) exercise the service plumbing: batch embedding, PII
scrubbing before insert, MemoryEntry shaping, and hybrid ranking. The
semantic-quality property (vectors that understand meaning) belongs to
the integration tests with real Gemini embeddings.
"""

import asyncio
import math

import pytest
from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types

from support_agent import config
from support_agent.backends.pgvector_memory import (
    FakeEmbedder,
    FakeEpisodeStore,
    GeminiEmbedder,
    PgVectorMemoryService,
)
from support_agent.memory_banks import EpisodicBank


def _event(author: str, text: str, event_id: str = "e1") -> Event:
    return Event(
        id=event_id,
        author=author,
        content=types.Content(
            role="user" if author == "user" else "model",
            parts=[types.Part(text=text)],
        ),
    )


def _session(sid: str, *texts: str) -> Session:
    return Session(
        app_name="test_app",
        user_id="test_user",
        id=sid,
        state={},
        events=[_event("user", t, event_id=f"{sid}-{i}") for i, t in enumerate(texts)],
    )


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def service():
    return PgVectorMemoryService(FakeEpisodeStore(), FakeEmbedder(dim=8))


# --- FakeEmbedder ------------------------------------------------------------------
def test_fake_embedder_deterministic_and_normalized():
    e = FakeEmbedder(dim=8)
    v1 = e.embed(["hello world"])[0]
    v2 = e.embed(["hello world"])[0]
    v3 = e.embed(["different text"])[0]
    assert v1 == v2
    assert v1 != v3
    assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-6)
    assert e.dim == 8


def test_gemini_embedder_constructs_without_key():
    # The key is only read when embed() runs, never at construction.
    g = GeminiEmbedder(model="gemini-embedding-001", dim=768)
    assert g.dim == 768


# --- add_session_to_memory ------------------------------------------------------------
def test_add_session_stores_one_row_per_event(service):
    async def go():
        await service.add_session_to_memory(_session("s1", "late order", "refund please"))
        return service._store._rows

    rows = run(go())
    assert len(rows) == 2
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["text"] == "late order"
    assert rows[0]["author"] == "user"
    assert len(rows[0]["embedding"]) == 8


def test_add_session_scrubs_pii_before_insert(service):
    async def go():
        await service.add_session_to_memory(
            _session("s2", "my card is 4111-1111-1111-1111 thanks")
        )
        return service._store._rows

    rows = run(go())
    assert len(rows) == 1
    assert "4111" not in rows[0]["text"]
    assert "[REDACTED]" in rows[0]["text"]


def test_add_session_skips_empty_events(service):
    async def go():
        s = Session(app_name="test_app", user_id="test_user", id="s3", state={}, events=[])
        await service.add_session_to_memory(s)
        return service._store._rows

    assert run(go()) == []


def test_rows_are_namespaced_per_user():
    store = FakeEpisodeStore()
    svc = PgVectorMemoryService(store, FakeEmbedder(dim=8))

    async def go():
        other = Session(
            app_name="test_app", user_id="other_user", id="sx", state={},
            events=[_event("user", "my secret order", event_id="sx-0")],
        )
        await svc.add_session_to_memory(other)
        resp = await svc.search_memory(
            app_name="test_app", user_id="test_user", query="secret order"
        )
        return resp.memories

    assert run(go()) == []


# --- search_memory -----------------------------------------------------------------------
def test_search_returns_memory_entries(service):
    async def go():
        await service.add_session_to_memory(_session("s1", "my order arrived late"))
        resp = await service.search_memory(
            app_name="test_app", user_id="test_user", query="late delivery"
        )
        return resp.memories

    memories = run(go())
    assert len(memories) == 1
    assert memories[0].content.parts[0].text == "my order arrived late"
    assert memories[0].author == "user"


def test_search_respects_limit(service):
    async def go():
        await service.add_session_to_memory(
            _session("s1", *[f"message number {i}" for i in range(10)])
        )
        resp = await service.search_memory(
            app_name="test_app", user_id="test_user", query="message"
        )
        return resp.memories

    assert len(run(go())) == config.MAX_EPISODIC_HITS


def test_hybrid_keyword_boost_can_reorder():
    """Same store, hand-set vectors: the token boost is real math."""
    store = FakeEpisodeStore()
    store.insert_episode(
        app_name="a", user_id="u", session_id="s", author="user", ts=None,
        text="totally unrelated words here", embedding=[1.0, 0.0], meta={},
    )
    store.insert_episode(
        app_name="a", user_id="u", session_id="s", author="user", ts=None,
        text="mentions the refund keyword", embedding=[0.0, 1.0], meta={},
    )
    # Query vector points almost exactly at row 1: vector distance wins
    # (0.0) over row 2's boosted score (~0.49).
    rows = store.search(
        app_name="a", user_id="u", query_vector=[0.99, 0.01],
        tokens=["refund"], limit=2,
    )
    assert rows[0]["text"] == "totally unrelated words here"
    # A closer call: row 1 scores ~0.17 on vectors alone, row 2 scores
    # ~0.45 - 0.5 = ~-0.05 with the keyword boost, so the boost flips it.
    rows2 = store.search(
        app_name="a", user_id="u", query_vector=[0.6, 0.4],
        tokens=["refund"], limit=2,
    )
    assert rows2[0]["text"] == "mentions the refund keyword"


def test_recall_through_episodic_bank():
    """EpisodicBank.recall works unchanged over the pgvector service."""
    bank = EpisodicBank(
        PgVectorMemoryService(FakeEpisodeStore(), FakeEmbedder(dim=8)),
        "test_app", "test_user",
    )

    async def go():
        await bank.record_session(_session("s1", "ORD-1001 arrived 5 days late"))
        return await bank.recall("late delivery")

    hits = run(go())
    assert any("ORD-1001" in h for h in hits)
    assert all("[REDACTED]" not in h for h in hits)


def test_service_ping_on_fake_store():
    run(PgVectorMemoryService(FakeEpisodeStore(), FakeEmbedder(dim=8)).ping())


# --- recall query: keyword hack is gone under vector search -------------------
from types import SimpleNamespace


class _FakeCallbackContext:
    def __init__(self, events):
        self.state: dict = {}
        self.session = SimpleNamespace(
            app_name="test_app", user_id="test_user", id="s9", events=events
        )


def _ctx_with_text(text: str):
    return _FakeCallbackContext([_event("user", text, event_id="q1")])


def test_recall_query_appends_domain_terms_on_keyword_backend(monkeypatch):
    from support_agent import agent as agent_module
    from support_agent import config as config_module
    from support_agent.memory_banks import clear_registry

    monkeypatch.setattr(config_module, "EPISODIC_BACKEND", "memory")
    clear_registry()
    captured = {}

    async def fake_recall(state, app_name, user_id, query, session_id=None):
        captured["query"] = query
        state["memory_recall_done"] = True

    monkeypatch.setattr(agent_module, "ensure_memory_recall", fake_recall)
    asyncio.run(agent_module._recall_before_agent(_ctx_with_text("I want a refund")))
    assert captured["query"] == "I want a refund customer preferences past orders"


def test_recall_query_is_raw_text_on_vector_backend(monkeypatch):
    from support_agent import agent as agent_module
    from support_agent import config as config_module
    from support_agent.memory_banks import clear_registry

    monkeypatch.setattr(config_module, "EPISODIC_BACKEND", "pgvector")
    clear_registry()
    captured = {}

    async def fake_recall(state, app_name, user_id, query, session_id=None):
        captured["query"] = query
        state["memory_recall_done"] = True

    monkeypatch.setattr(agent_module, "ensure_memory_recall", fake_recall)
    asyncio.run(agent_module._recall_before_agent(_ctx_with_text("I want a refund")))
    assert captured["query"] == "I want a refund"
