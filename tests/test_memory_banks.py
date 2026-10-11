"""Memory banks: semantic upsert/supersede, episodic record/recall."""

import asyncio

from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types

from support_agent.memory_banks import (
    MemoryBanks,
    clear_registry,
    get_banks,
    register_banks,
)


def _event(author: str, text: str) -> Event:
    return Event(
        author=author,
        content=types.Content(
            role="user" if author == "user" else "model",
            parts=[types.Part(text=text)],
        ),
    )


def test_registry_is_per_user(banks):
    from google.adk.memory import InMemoryMemoryService

    other = register_banks("test_app", "someone_else", InMemoryMemoryService())
    assert get_banks("test_app", "test_user") is banks
    assert get_banks("test_app", "someone_else") is other
    assert get_banks("test_app", "ghost") is None


def test_semantic_upsert_and_supersede(banks):
    banks.semantic.upsert("contact_preference", "prefers email updates")
    banks.semantic.upsert("shipping_preference", "standard shipping")
    r = banks.semantic.upsert("contact_preference", "prefers SMS updates")
    assert r["superseded"] is True

    active = {f.slot: f.text for f in banks.semantic.active_facts()}
    assert active == {
        "contact_preference": "prefers SMS updates",
        "shipping_preference": "standard shipping",
    }
    # Superseded facts stay in history (audit trail) but never in recall.
    assert len(banks.semantic.history()) == 3
    assert all("email" not in f.text for f in banks.semantic.active_facts())


def test_semantic_upsert_same_text_is_not_supersede(banks):
    banks.semantic.upsert("contact_preference", "prefers email")
    r = banks.semantic.upsert("contact_preference", "prefers email")
    assert r["superseded"] is False
    assert len(banks.semantic.active_facts()) == 1


def test_semantic_pii_guard(banks):
    r = banks.semantic.upsert("payment", "my card is 4111-1111-1111-1111")
    assert "error" in r
    assert banks.semantic.active_facts() == []


def test_semantic_search(banks):
    banks.semantic.upsert("contact_preference", "prefers email updates")
    banks.semantic.upsert("shipping_preference", "express shipping")
    hits = banks.semantic.search("email contact")
    assert [f.slot for f in hits] == ["contact_preference"]


def test_episodic_record_and_recall(banks):
    async def go():
        session = Session(
            app_name="test_app",
            user_id="test_user",
            id="s1",
            state={},
            events=[
                _event("user", "My order ORD-1001 arrived 5 days late"),
                _event("support_agent", "Issued a $10 goodwill credit for the late delivery"),
            ],
        )
        await banks.episodic.record_session(session)
        return await banks.episodic.recall("late delivery credit")

    hits = asyncio.run(go())
    assert any("ORD-1001" in h for h in hits)


def test_build_recall_block_combines_banks(banks):
    async def go():
        session = Session(
            app_name="test_app",
            user_id="test_user",
            id="s1",
            state={},
            events=[_event("user", "ORD-1001 was late")],
        )
        await banks.episodic.record_session(session)
        banks.semantic.upsert("contact_preference", "prefers email updates")
        return await banks.build_recall_block("preferences late order")

    block = asyncio.run(go())
    assert "prefers email updates" in block
    assert "ORD-1001" in block
    assert "MEMORY" in block


def test_build_recall_block_empty_when_nothing_known(banks):
    block = asyncio.run(banks.build_recall_block("anything"))
    assert block == ""


def test_clear_registry():
    clear_registry()
    assert get_banks("test_app", "test_user") is None
