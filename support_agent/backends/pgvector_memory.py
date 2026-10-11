"""Episodic long-term memory on PostgreSQL + pgvector (Phase 2b).

A drop-in ADK BaseMemoryService: each event of an ingested session becomes
one row (app_name, user_id, session_id, author, ts, text, embedding, meta),
embedded once in batch. search_memory() embeds the query and orders by
cosine distance with a small keyword-overlap boost (hybrid) -- this is
what replaces the Phase-0 keyword-search hack in the recall path.

Embeddings: GeminiEmbedder (google.genai, key from the environment at call
time) or FakeEmbedder (deterministic, offline). The store is injectable:
PgVectorEpisodeStore (real psycopg) or FakeEpisodeStore (in-memory, for
unit tests). Blocking DB calls run in asyncio.to_thread so the event loop
stays free.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import math
import random
import re
from datetime import datetime, timezone

from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types

from support_agent import config
from support_agent.memory_banks import scrub_event


# --- Embedders ------------------------------------------------------------


class Embedder(abc.ABC):
    """Embed a batch of texts -> one unit-ish vector per text."""

    @property
    @abc.abstractmethod
    def dim(self) -> int: ...

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder(Embedder):
    """Deterministic offline embedder (no key, no network).

    sha256(text) seeds a PRNG -> gaussian vector, L2-normalized. Same text
    always yields the same vector; similar texts are NOT guaranteed to be
    close -- it tests plumbing and ranking math, not semantic quality.
    """

    def __init__(self, dim: int = 768) -> None:
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            vec = [rng.gauss(0.0, 1.0) for _ in range(self._dim)]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class GeminiEmbedder(Embedder):
    """Real embeddings via google.genai. The client reads GOOGLE_API_KEY
    from the environment when embed() is first called -- never at import."""

    def __init__(self, model: str, dim: int) -> None:
        self._model = model
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google import genai

        client = genai.Client()
        response = client.models.embed_content(
            model=self._model, contents=list(texts)
        )
        return [list(e.values) for e in response.embeddings]


# --- Episode stores ---------------------------------------------------------


class EpisodeStore(abc.ABC):
    """Persistence for embedded episode rows."""

    @abc.abstractmethod
    def ensure_schema(self) -> None: ...

    @abc.abstractmethod
    def insert_episode(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        author: str | None,
        ts: str | None,
        text: str,
        embedding: list[float],
        meta: dict,
    ) -> None: ...

    @abc.abstractmethod
    def search(
        self,
        *,
        app_name: str,
        user_id: str,
        query_vector: list[float],
        tokens: list[str],
        limit: int,
    ) -> list[dict]: ...

    @abc.abstractmethod
    def list_episodes_older_than(
        self, *, app_name: str, user_id: str, cutoff_iso: str
    ) -> list[dict]:
        """Episodes with a timestamp older than cutoff_iso (Phase 4).

        Returns dicts with id/session_id/author/timestamp/text. Rows with
        no timestamp are excluded (their age is unknown).
        """
        ...

    @abc.abstractmethod
    def delete_episode(self, *, episode_id) -> None:
        """Drop one episode row by id (Phase 4 summarization)."""
        ...

    def ping(self) -> None:
        """Raise if the store is unreachable (used by test skips)."""


class PgVectorEpisodeStore(EpisodeStore):
    """Real PostgreSQL + pgvector store (psycopg, sync calls in threads)."""

    def __init__(self, dsn: str, dim: int) -> None:
        self._dsn = dsn
        self._dim = int(dim)

    def _connect(self):
        import psycopg

        # Short connect timeout: probes must fail fast (skip) when the
        # database is not running, instead of hanging on a blackholed SYN.
        return psycopg.connect(self._dsn, connect_timeout=3)

    def ensure_schema(self) -> None:
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS episodes (
          id BIGSERIAL PRIMARY KEY,
          app_name TEXT NOT NULL,
          user_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          author TEXT,
          ts TIMESTAMPTZ,
          text TEXT NOT NULL,
          embedding VECTOR({self._dim}) NOT NULL,
          meta JSONB NOT NULL DEFAULT '{{}}'
        );
        CREATE INDEX IF NOT EXISTS episodes_owner_idx
          ON episodes (app_name, user_id);
        """
        # ivfflat needs a populated table to be useful; create it lazily so
        # an empty database never errors on first search.
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
                cur.execute("SELECT count(*) FROM episodes")
                if cur.fetchone()[0] > 0:
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS episodes_vec_idx "
                        "ON episodes USING ivfflat (embedding vector_cosine_ops) "
                        "WITH (lists = 100)"
                    )
            conn.commit()

    def insert_episode(self, **kwargs) -> None:
        import json as _json

        vec = "[" + ",".join(f"{v:.6f}" for v in kwargs["embedding"]) + "]"
        ts = kwargs.get("ts")
        ts_val = (
            datetime.fromisoformat(ts) if isinstance(ts, str) and ts else None
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO episodes "
                    "(app_name, user_id, session_id, author, ts, text, embedding, meta) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s::vector,%s::jsonb)",
                    (
                        kwargs["app_name"],
                        kwargs["user_id"],
                        kwargs["session_id"],
                        kwargs.get("author"),
                        ts_val,
                        kwargs["text"],
                        vec,
                        _json.dumps(kwargs.get("meta") or {}),
                    ),
                )
            conn.commit()

    def search(self, *, app_name, user_id, query_vector, tokens, limit) -> list[dict]:
        vec = "[" + ",".join(f"{v:.6f}" for v in query_vector) + "]"
        if tokens:
            likes = [f"%{t}%" for t in tokens]
            order = (
                "(e.embedding <=> %s::vector) + "
                "CASE WHEN e.text ILIKE ANY(%s) THEN 0.0 ELSE 0.5 END"
            )
        else:
            order = "(e.embedding <=> %s::vector)"
        sql = (
            "SELECT e.id, e.author, e.ts, e.text "
            "FROM episodes e WHERE e.app_name=%s AND e.user_id=%s "
            f"ORDER BY {order} LIMIT %s"
        )
        # Placeholder order: vector (and token patterns) first, then
        # app_name, user_id, limit -- matching the SQL above.
        if tokens:
            params = (vec, likes, app_name, user_id, limit)
        else:
            params = (vec, app_name, user_id, limit)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        out = []
        for rid, author, ts, text in rows:
            out.append(
                {
                    "id": rid,
                    "author": author,
                    "timestamp": ts.isoformat() if ts else None,
                    "text": text,
                }
            )
        return out

    def list_episodes_older_than(
        self, *, app_name: str, user_id: str, cutoff_iso: str
    ) -> list[dict]:
        """Episodes with ts older than cutoff_iso (Phase 4 summarization).

        Rows with NULL ts are excluded -- their age is unknown.
        """
        sql = (
            "SELECT e.id, e.session_id, e.author, e.ts, e.text "
            "FROM episodes e "
            "WHERE e.app_name=%s AND e.user_id=%s "
            "AND e.ts IS NOT NULL AND e.ts < %s::timestamptz "
            "ORDER BY e.ts"
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (app_name, user_id, cutoff_iso))
                rows = cur.fetchall()
        return [
            {
                "id": rid,
                "session_id": sid,
                "author": author,
                "timestamp": ts.isoformat() if ts else None,
                "text": text,
            }
            for rid, sid, author, ts, text in rows
        ]

    def delete_episode(self, *, episode_id) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM episodes WHERE id=%s", (episode_id,))
            conn.commit()

    def ping(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")


class FakeEpisodeStore(EpisodeStore):
    """In-memory store with real cosine math, for offline unit tests."""

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._next_id = 1

    def ensure_schema(self) -> None:
        pass

    def insert_episode(self, **kwargs) -> None:
        row = dict(kwargs)
        row["id"] = self._next_id
        self._next_id += 1
        self._rows.append(row)

    def search(self, *, app_name, user_id, query_vector, tokens, limit) -> list[dict]:
        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        scored = []
        for row in self._rows:
            if row["app_name"] != app_name or row["user_id"] != user_id:
                continue
            dist = 1.0 - cosine(query_vector, row["embedding"])
            boost = 0.0
            if tokens:
                hay = row["text"].lower()
                if any(t.lower() in hay for t in tokens):
                    boost = -0.5
            scored.append((dist + boost, row))
        scored.sort(key=lambda s: s[0])
        return [
            {
                "id": r["id"],
                "author": r.get("author"),
                "timestamp": r.get("ts"),
                "text": r["text"],
            }
            for _, r in scored[:limit]
        ]

    def list_episodes_older_than(
        self, *, app_name: str, user_id: str, cutoff_iso: str
    ) -> list[dict]:
        """Episodes with ts older than cutoff_iso (Phase 4, offline)."""
        out = []
        for row in self._rows:
            if row["app_name"] != app_name or row["user_id"] != user_id:
                continue
            ts = row.get("ts")
            if not ts or ts >= cutoff_iso:
                continue
            out.append(
                {
                    "id": row["id"],
                    "session_id": row.get("session_id"),
                    "author": row.get("author"),
                    "timestamp": ts,
                    "text": row["text"],
                }
            )
        out.sort(key=lambda r: r["timestamp"] or "")
        return out

    def delete_episode(self, *, episode_id) -> None:
        self._rows = [r for r in self._rows if r["id"] != episode_id]


# --- The ADK memory service -------------------------------------------------


def _event_text(event) -> str:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return ""
    return " ".join(p.text for p in parts if getattr(p, "text", None)).strip()


def _event_ts(event) -> str | None:
    ts = getattr(event, "timestamp", None)
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


class PgVectorMemoryService(BaseMemoryService):
    """Episodic memory on pgvector. One row per event; vector-ranked recall."""

    def __init__(self, store: EpisodeStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder
        self._ensured = False

    @property
    def episode_store(self) -> EpisodeStore:
        """The underlying store (Phase 4 maintenance needs list/delete)."""
        return self._store

    async def _ensure(self) -> None:
        if not self._ensured:
            await asyncio.to_thread(self._store.ensure_schema)
            self._ensured = True

    async def add_session_to_memory(self, session) -> None:
        await self._ensure()
        events = session.events or []
        texts: list[str] = []
        metas: list[dict] = []
        for event in events:
            clean, _ = scrub_event(event)  # PII never reaches the backend
            text = _event_text(clean)
            if not text:
                continue
            texts.append(text)
            metas.append(
                {"event": clean, "author": getattr(clean, "author", None)}
            )
        if not texts:
            return
        vectors = await asyncio.to_thread(self._embedder.embed, texts)
        for meta, text, vec in zip(metas, texts, vectors):
            await asyncio.to_thread(
                self._store.insert_episode,
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.id,
                author=meta["author"],
                ts=_event_ts(meta["event"]),
                text=text,
                embedding=vec,
                meta={},
            )

    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        await self._ensure()
        vectors = await asyncio.to_thread(self._embedder.embed, [query or ""])
        tokens = [w for w in re.findall(r"\w+", (query or "").lower()) if len(w) > 2]
        rows = await asyncio.to_thread(
            self._store.search,
            app_name=app_name,
            user_id=user_id,
            query_vector=vectors[0],
            tokens=tokens,
            limit=config.MAX_EPISODIC_HITS,
        )
        memories = [
            MemoryEntry(
                id=str(r["id"]),
                author=r.get("author"),
                timestamp=r.get("timestamp"),
                content=types.Content(
                    role="user", parts=[types.Part(text=r["text"])]
                ),
            )
            for r in rows
        ]
        return SearchMemoryResponse(memories=memories)

    async def ping(self) -> None:
        await asyncio.to_thread(self._store.ping)
