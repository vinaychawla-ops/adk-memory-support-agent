"""The two long-term memory banks, kept conceptually separate.

Episodic memory -- what happened in past interactions -- lives in ADK's
BaseMemoryService (InMemoryMemoryService in Phase 0): raw session events
are stored with add_session_to_memory() and retrieved by keyword search
with search_memory().

Semantic memory -- durable facts and preferences about the customer --
lives in SemanticBank: a small fact store with upsert/supersede logic.
A new fact for an already-known slot (e.g. "contact_preference")
supersedes the old one instead of appending; superseded facts are kept
for audit but never surface in recall. (Phase 2c moves this to Firestore
with the same record shape.)

Both banks are keyed by (app_name, user_id). A module-level registry maps
that key to a MemoryBanks instance; make_runner()/demo.py register one
per user via register_banks(). Tools resolve the banks from
tool_context.session, so no global user state leaks across users.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from support_agent import config

# Card-number-shaped strings must never enter any memory layer.
_PII_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")


def _stem(word: str) -> str:
    """Tiny stemmer: strips common English suffixes for recall matching."""
    w = word.lower()
    for suffix in ("ing", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)]
    return w


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Fact:
    """One semantic fact: a durable, slot-scoped statement about the customer."""

    slot: str
    text: str
    confidence: float = 1.0
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    superseded_by: str | None = None  # text of the newer fact, if superseded

    @property
    def active(self) -> bool:
        return self.superseded_by is None


class SemanticBank:
    """Durable facts with upsert/supersede. In-memory in Phase 0."""

    def __init__(self) -> None:
        self._facts: list[Fact] = []

    def upsert(self, slot: str, text: str, confidence: float = 1.0) -> dict:
        """Store a fact. A new fact for an existing slot supersedes the old.

        Returns {"stored", "slot", "superseded"} or {"error"}.
        """
        slot = (slot or "").strip()
        text = (text or "").strip()
        if not slot or not text:
            return {"error": "slot and text are both required"}
        if _PII_RE.search(text):
            return {
                "error": "Refused: text looks like it contains a card number. "
                "Card numbers must never enter memory."
            }
        if not 0.0 <= confidence <= 1.0:
            return {"error": "confidence must be between 0 and 1"}

        superseded = False
        for old in self._facts:
            if old.slot != slot or not old.active:
                continue
            if old.text == text:
                # Identical re-statement: idempotent, nothing changes.
                return {"stored": True, "slot": slot, "superseded": False}
            old.superseded_by = text
            old.updated_at = _utcnow()
            superseded = True

        self._facts.append(Fact(slot=slot, text=text, confidence=confidence))
        return {"stored": True, "slot": slot, "superseded": superseded}

    def active_facts(self) -> list[Fact]:
        return [f for f in self._facts if f.active]

    def history(self) -> list[Fact]:
        """All facts including superseded ones (audit trail)."""
        return list(self._facts)

    def search(self, query: str, limit: int = config.MAX_SEMANTIC_FACTS) -> list[Fact]:
        """Keyword search over active facts. Empty query returns all active.

        Matches on stemmed tokens so "preferences" finds "prefers".
        """
        words = {_stem(w) for w in re.findall(r"\w+", query or "") if len(w) > 2}

        def _matches(fact: Fact) -> int:
            tokens = re.findall(r"\w+", f"{fact.slot} {fact.text}".lower())
            score = 0
            for q in words:
                if any(q in t or t in q for t in tokens):
                    score += 1
            return score

        scored = [(_matches(f), f) for f in self.active_facts()]
        if words:
            scored = [(s, f) for s, f in scored if s]
        scored.sort(key=lambda s: (-s[0], s[1].created_at))
        return [f for _, f in scored[:limit]]


class EpisodicBank:
    """Past interactions, via ADK's memory service (raw session events)."""

    def __init__(self, memory_service, app_name: str, user_id: str) -> None:
        self._service = memory_service
        self._app_name = app_name
        self._user_id = user_id

    async def record_session(self, session) -> None:
        """Store a finished session's events as one episode."""
        await self._service.add_session_to_memory(session)

    async def recall(self, query: str, limit: int = config.MAX_EPISODIC_HITS) -> list[str]:
        """Keyword search over past episodes -> short human-readable lines."""
        response = await self._service.search_memory(
            app_name=self._app_name, user_id=self._user_id, query=query
        )
        lines: list[str] = []
        for entry in response.memories[:limit]:
            text = " ".join(
                part.text for part in (entry.content.parts or []) if part.text
            ).strip()
            if not text:
                continue
            if len(text) > 220:
                text = text[:217] + "..."
            author = entry.author or "unknown"
            lines.append(f"[{entry.timestamp or 'past'}] {author}: {text}")
        return lines


class MemoryBanks:
    """Both long-term banks for one (app_name, user_id)."""

    def __init__(self, memory_service, app_name: str, user_id: str) -> None:
        self.episodic = EpisodicBank(memory_service, app_name, user_id)
        self.semantic = SemanticBank()
        self.app_name = app_name
        self.user_id = user_id

    async def build_recall_block(self, query: str) -> str:
        """The text injected into the agent's context at session start."""
        facts = self.semantic.search(query)
        episodes = await self.episodic.recall(query)

        parts: list[str] = []
        if facts:
            parts.append("What I know about this customer (durable facts):")
            parts.extend(f"- [{f.slot}] {f.text}" for f in facts)
        if episodes:
            parts.append("Relevant past interactions:")
            parts.extend(f"- {line}" for line in episodes)
        if not parts:
            return ""
        return (
            "MEMORY -- things remembered from earlier conversations. "
            "Use them; do not ask the customer to repeat what is stated here.\n"
            + "\n".join(parts)
        )


# --- Registry ---------------------------------------------------------------
# One MemoryBanks per (app_name, user_id). Registered by make_runner()/demo;
# resolved by tools from tool_context.session. Never shared across users.

_REGISTRY: dict[tuple[str, str], MemoryBanks] = {}


def register_banks(app_name: str, user_id: str, memory_service) -> MemoryBanks:
    """Create (or return the existing) banks for a user. Idempotent."""
    key = (app_name, user_id)
    banks = _REGISTRY.get(key)
    if banks is None:
        banks = MemoryBanks(memory_service, app_name, user_id)
        _REGISTRY[key] = banks
    return banks


def get_banks(app_name: str, user_id: str) -> MemoryBanks | None:
    return _REGISTRY.get((app_name, user_id))


def clear_registry() -> None:
    """Tests only: drop all registered banks."""
    _REGISTRY.clear()
