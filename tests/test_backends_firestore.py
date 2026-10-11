"""Unit tests for FirestoreSemanticBank (2c) with an in-memory FakeFirestore.

Covers the SemanticBank contract parity (upsert/supersede/idempotent,
card refusal, search), per-user isolation, expiry/purge, and the ping
used by test skips.
"""

import pytest

from support_agent import config
from support_agent.backends.firestore_semantics import (
    FakeFirestore,
    FirestoreSemanticBank,
)
from support_agent.memory_banks import SemanticBank


@pytest.fixture()
def db():
    return FakeFirestore()


@pytest.fixture()
def bank(db):
    return FirestoreSemanticBank(db, "test_app", "test_user")


def test_upsert_and_active_facts(bank):
    r = bank.upsert("contact_preference", "prefers email updates")
    assert r == {"stored": True, "slot": "contact_preference", "superseded": False}
    facts = bank.active_facts()
    assert len(facts) == 1
    assert facts[0].text == "prefers email updates"
    assert facts[0].confidence == 1.0


def test_supersede_keeps_history(bank):
    bank.upsert("contact_preference", "prefers email updates")
    r = bank.upsert("contact_preference", "prefers SMS updates")
    assert r["superseded"] is True
    active = bank.active_facts()
    assert [f.text for f in active] == ["prefers SMS updates"]
    assert len(bank.history()) == 2  # old fact kept for audit only


def test_identical_reupsert_is_idempotent(bank):
    bank.upsert("contact_preference", "prefers email updates")
    r = bank.upsert("contact_preference", "prefers email updates")
    assert r == {"stored": True, "slot": "contact_preference", "superseded": False}
    assert len(bank.history()) == 1


def test_card_number_refused(bank):
    r = bank.upsert("payment", "my card is 4111 1111 1111 1111")
    assert "error" in r
    assert bank.active_facts() == []
    assert bank.history() == []


def test_validation_errors(bank):
    assert "error" in bank.upsert("", "text")
    assert "error" in bank.upsert("slot", "")
    assert "error" in bank.upsert("slot", "text", confidence=1.5)


def test_users_are_isolated(db):
    b1 = FirestoreSemanticBank(db, "test_app", "alice")
    b2 = FirestoreSemanticBank(db, "test_app", "bob")
    b1.upsert("contact_preference", "prefers email")
    assert b2.active_facts() == []
    # Same user, different app is also isolated.
    b3 = FirestoreSemanticBank(db, "other_app", "alice")
    assert b3.active_facts() == []


def test_search_parity_with_semantic_bank(db):
    """Same data, same query -> same result order as the in-memory bank."""
    mem = SemanticBank()
    fs = FirestoreSemanticBank(db, "test_app", "test_user")
    data = [
        ("contact_preference", "prefers email updates"),
        ("shipping_preference", "prefers express shipping"),
        ("language", "prefers Spanish"),
    ]
    for slot, text in data:
        mem.upsert(slot, text)
        fs.upsert(slot, text)
    for query in ["preferences", "email", "shipping", ""]:
        mem_texts = [f.text for f in mem.search(query)]
        fs_texts = [f.text for f in fs.search(query, limit=config.MAX_SEMANTIC_FACTS)]
        assert fs_texts == mem_texts, f"query {query!r} diverged"


def test_search_limit_respected(bank):
    for i in range(5):
        bank.upsert(f"slot_{i}", f"customer prefers option {i}")
    assert len(bank.search("prefers", limit=2)) == 2


def test_expired_facts_do_not_surface(bank):
    bank.upsert("promo", "10% off this week")
    # Backdate the expiry directly in the document.
    raw = bank._read()
    raw[0]["expires"] = "2000-01-01"
    bank._write(raw)
    assert bank.active_facts() == []
    assert len(bank.history()) == 1  # still visible in history until purged


def test_purge_expired(bank):
    bank.upsert("promo", "10% off this week")
    bank.upsert("contact_preference", "prefers email")
    raw = bank._read()
    raw[0]["expires"] = "2000-01-01"
    bank._write(raw)
    assert bank.purge_expired() == 1
    assert [f.slot for f in bank.active_facts()] == ["contact_preference"]
    assert bank.purge_expired() == 0  # second run is a no-op


def test_ping(db):
    FirestoreSemanticBank(db, "test_app", "test_user").ping()


def test_new_bank_reads_existing_document(db):
    """Durability across instances: a second bank sees the first's writes."""
    b1 = FirestoreSemanticBank(db, "test_app", "test_user")
    b1.upsert("contact_preference", "prefers email updates")
    b2 = FirestoreSemanticBank(db, "test_app", "test_user")
    assert [f.text for f in b2.active_facts()] == ["prefers email updates"]
