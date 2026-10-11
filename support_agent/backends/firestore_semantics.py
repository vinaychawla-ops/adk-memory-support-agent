"""Semantic long-term memory on Firestore (Phase 2c).

A drop-in replacement for the in-memory SemanticBank with the same public
contract (upsert / active_facts / history / search). One document per user:
`semantic_memory/{app_name}__{user_id}` holding a `facts` array -- no
composite indexes, no queries, single-document reads/writes. The real
Firestore client honors FIRESTORE_EMULATOR_HOST natively, so local Docker
vs. real Firestore is a pure connection swap with zero code change.

Fact shape: {slot, text, confidence, created_at, updated_at,
superseded_by, expires}. `expires` is an optional ISO date; expired facts
stop surfacing (Phase 4 will purge them via purge_expired()).

The db client is injected (real google.cloud.firestore.Client or the
FakeFirestore in tests); this module imports no Firestore library itself.
"""

from __future__ import annotations

import dataclasses

from support_agent import config
from support_agent.audit import CARD_RE
from support_agent.memory_banks import Fact, _utcnow, score_facts


class FirestoreSemanticBank:
    """Durable facts with upsert/supersede, stored in Firestore."""

    COLLECTION = "semantic_memory"

    def __init__(self, db, app_name: str, user_id: str) -> None:
        self._doc = db.collection(self.COLLECTION).document(
            f"{app_name}__{user_id}"
        )
        self._app_name = app_name
        self._user_id = user_id

    # -- document I/O ----------------------------------------------------
    def _read(self) -> list[dict]:
        snap = self._doc.get()
        if not snap.exists:
            return []
        data = snap.to_dict() or {}
        facts = data.get("facts") or []
        return [f for f in facts if isinstance(f, dict)]

    def _write(self, facts: list[dict]) -> None:
        self._doc.set({"facts": facts})

    @staticmethod
    def _to_fact(raw: dict) -> Fact:
        allowed = {f.name for f in dataclasses.fields(Fact)}
        return Fact(**{k: v for k, v in raw.items() if k in allowed})

    # -- SemanticBank-compatible API --------------------------------------
    def upsert(self, slot: str, text: str, confidence: float = 1.0) -> dict:
        """Same contract as SemanticBank.upsert (stored/superseded/error)."""
        slot = (slot or "").strip()
        text = (text or "").strip()
        if not slot or not text:
            return {"error": "slot and text are both required"}
        if CARD_RE.search(text):
            return {
                "error": "Refused: text looks like it contains a card number. "
                "Card numbers must never enter memory."
            }
        if not 0.0 <= confidence <= 1.0:
            return {"error": "confidence must be between 0 and 1"}

        facts = self._read()
        superseded = False
        for raw in facts:
            if raw.get("slot") != slot or raw.get("superseded_by") is not None:
                continue
            if raw.get("text") == text:
                return {"stored": True, "slot": slot, "superseded": False}
            raw["superseded_by"] = text
            raw["updated_at"] = _utcnow()
            superseded = True

        now = _utcnow()
        facts.append(
            {
                "slot": slot,
                "text": text,
                "confidence": confidence,
                "created_at": now,
                "updated_at": now,
                "superseded_by": None,
                "expires": None,
            }
        )
        self._write(facts)
        return {"stored": True, "slot": slot, "superseded": superseded}

    def active_facts(self) -> list[Fact]:
        return [f for f in (self._to_fact(r) for r in self._read()) if f.active]

    def history(self) -> list[Fact]:
        return [self._to_fact(r) for r in self._read()]

    def search(self, query: str, limit: int = config.MAX_SEMANTIC_FACTS) -> list[Fact]:
        """Keyword search over active facts -- same scoring as SemanticBank.

        Facts are short slot/text pairs, so keyword matching is the right
        tool here; the vector upgrade (Phase 2b) applies to episodes, not
        facts.
        """
        return score_facts(self.active_facts(), query, limit)

    def purge_expired(self) -> int:
        """Drop expired facts (kept for audit in history until purged).

        Returns the number removed. Phase 4 calls this on a schedule.
        """
        facts = self._read()
        kept = []
        removed = 0
        for raw in facts:
            fact = self._to_fact(raw)
            # Expired and not superseded: drop it. Superseded facts stay as
            # the audit trail (Phase 4 decides their lifecycle).
            if fact.expires and fact.superseded_by is None and not fact.active:
                removed += 1
                continue
            kept.append(raw)
        if removed:
            self._write(kept)
        return removed

    def ping(self) -> None:
        """Raise if Firestore is unreachable (used by test skips).

        Probes the emulator over raw TCP first: the gRPC deadline is not
        always honored against a blackholed port, which would hang the
        skip instead of failing fast.
        """
        import os
        import socket

        emu = os.environ.get("FIRESTORE_EMULATOR_HOST", "")
        if emu:
            host, _, port = emu.partition(":")
            with socket.create_connection((host, int(port or 8080)), timeout=3):
                return
        self._doc.get(timeout=10)


# --- Fake Firestore (offline unit tests) ------------------------------------


class _FakeSnapshot:
    def __init__(self, data: dict | None) -> None:
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict | None:
        return dict(self._data) if self._data is not None else None


class _FakeDocument:
    def __init__(self, store: dict, doc_id: str) -> None:
        self._store = store
        self._id = doc_id

    def get(self, **kwargs) -> _FakeSnapshot:
        return _FakeSnapshot(self._store.get(self._id))

    def set(self, data: dict) -> None:
        self._store[self._id] = dict(data)


class _FakeCollection:
    def __init__(self, store: dict) -> None:
        self._store = store

    def document(self, doc_id: str) -> _FakeDocument:
        return _FakeDocument(self._store, doc_id)


class FakeFirestore:
    """In-memory stand-in implementing the tiny Firestore surface we use."""

    def __init__(self) -> None:
        self._collections: dict[str, dict] = {}

    def collection(self, name: str) -> _FakeCollection:
        return _FakeCollection(self._collections.setdefault(name, {}))
