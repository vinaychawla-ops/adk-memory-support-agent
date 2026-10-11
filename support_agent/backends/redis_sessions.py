"""Session state + conversation history on Redis (Phase 2a).

A drop-in ADK BaseSessionService: sessions are JSON-serialized
(Session.model_dump_json / model_validate_json) under
`msa:sess:{app}:{user}:{session_id}` with a sliding TTL, so ANY replica
can serve ANY session -- this is what unblocks Phase 3. App-scoped and
user-scoped state live in shared hashes, merged into the session on read
exactly like InMemorySessionService does.

PII: event text is scrubbed (card numbers -> [REDACTED]) before anything
is written to Redis. The caller's in-memory session object is never
mutated by the scrub.

The client is injected (real redis.Redis or the FakeRedis in tests), so
this module imports no redis library itself.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from google.adk.events.event import Event
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.errors.session_not_found_error import SessionNotFoundError
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session
from google.adk.sessions.state import State

from google.adk.sessions import _session_util as session_util

from support_agent.memory_banks import scrub_event


def _now() -> float:
    return time.time()


class RedisSessionService(BaseSessionService):
    """ADK session service backed by Redis with per-session TTL."""

    def __init__(self, client, ttl_seconds: int) -> None:
        self._r = client
        self._ttl = int(ttl_seconds)

    # -- key layout ------------------------------------------------------
    @staticmethod
    def _skey(app_name: str, user_id: str, session_id: str) -> str:
        return f"msa:sess:{app_name}:{user_id}:{session_id}"

    @staticmethod
    def _idx(app_name: str, user_id: str) -> str:
        return f"msa:sessidx:{app_name}:{user_id}"

    @staticmethod
    def _ukey(app_name: str, user_id: str) -> str:
        return f"msa:userstate:{app_name}:{user_id}"

    @staticmethod
    def _akey(app_name: str) -> str:
        return f"msa:appstate:{app_name}"

    # -- persistence helpers ---------------------------------------------
    def _save(self, session: Session) -> None:
        """Persist a scrubbed copy with a sliding TTL; refresh the index."""
        events = []
        for event in session.events or []:
            clean, _ = scrub_event(event)
            events.append(clean)
        payload = session.model_copy(update={"events": events}).model_dump_json()
        key = self._skey(session.app_name, session.user_id, session.id)
        self._r.set(key, payload, ex=self._ttl)
        idx = self._idx(session.app_name, session.user_id)
        self._r.sadd(idx, session.id)
        self._r.expire(idx, self._ttl)

    def _load(self, app_name: str, user_id: str, session_id: str) -> Optional[Session]:
        raw = self._r.get(self._skey(app_name, user_id, session_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return Session.model_validate_json(raw)

    def _merge_state(self, app_name: str, user_id: str, session: Session) -> Session:
        """Merge shared app/user state into the session (prefixed keys)."""
        app_raw = self._r.hgetall(self._akey(app_name)) or {}
        for key, value in app_raw.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            session.state[State.APP_PREFIX + key] = json.loads(value)
        user_raw = self._r.hgetall(self._ukey(app_name, user_id)) or {}
        for key, value in user_raw.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            session.state[State.USER_PREFIX + key] = json.loads(value)
        return session

    def _store_shared_deltas(
        self, app_name: str, user_id: str, deltas: dict[str, dict[str, Any]]
    ) -> None:
        if deltas.get("app"):
            self._r.hset(
                self._akey(app_name),
                mapping={k: json.dumps(v) for k, v in deltas["app"].items()},
            )
        if deltas.get("user"):
            self._r.hset(
                self._ukey(app_name, user_id),
                mapping={k: json.dumps(v) for k, v in deltas["user"].items()},
            )

    # -- BaseSessionService API ------------------------------------------
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        sid = (session_id or "").strip() or uuid.uuid4().hex
        if self._load(app_name, user_id, sid) is not None:
            raise AlreadyExistsError(f"Session with id {sid} already exists.")
        deltas = session_util.extract_state_delta(state or {})
        self._store_shared_deltas(app_name, user_id, deltas)
        session = Session(
            app_name=app_name,
            user_id=user_id,
            id=sid,
            state=deltas["session"] or {},
            last_update_time=_now(),
        )
        self._save(session)
        return self._merge_state(app_name, user_id, session)

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        sid = (session_id or "").strip()
        session = self._load(app_name, user_id, sid)
        if session is None:
            return None
        if config:
            if config.num_recent_events is not None:
                if config.num_recent_events == 0:
                    session.events = []
                else:
                    session.events = session.events[-config.num_recent_events :]
            if config.after_timestamp:
                session.events = [
                    e
                    for e in session.events
                    if (e.timestamp or 0) >= config.after_timestamp
                ]
        return self._merge_state(app_name, user_id, session)

    async def list_sessions(
        self, *, app_name: str, user_id: Optional[str] = None
    ) -> ListSessionsResponse:
        if user_id is None:
            # Index is per-user; without a user we cannot enumerate.
            return ListSessionsResponse(sessions=[])
        idx = self._idx(app_name, user_id)
        members = self._r.smembers(idx) or set()
        sessions: list[Session] = []
        for member in members:
            sid = member.decode("utf-8") if isinstance(member, bytes) else member
            session = self._load(app_name, user_id, sid)
            if session is None:
                self._r.srem(idx, sid)  # prune expired entries
                continue
            sessions.append(session)
        sessions.sort(key=lambda s: s.last_update_time or 0.0)
        return ListSessionsResponse(sessions=sessions)

    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        self._r.delete(self._skey(app_name, user_id, session_id))
        self._r.srem(self._idx(app_name, user_id), session_id)

    async def get_user_state(
        self, *, app_name: str, user_id: str
    ) -> dict[str, Any]:
        """Raw (un-prefixed) user-scoped state, like InMemorySessionService."""
        raw = self._r.hgetall(self._ukey(app_name, user_id)) or {}
        out: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            out[key] = json.loads(value)
        return out

    def ping(self) -> None:
        """Raise if Redis is unreachable (used by test skips)."""
        self._r.ping()

    async def append_event(self, session: Session, event: Event) -> Event:
        if event.partial:
            return event
        stored = self._load(session.app_name, session.user_id, session.id)
        if stored is None:
            raise SessionNotFoundError(f"Session {session.id} not found.")
        if any(e == event for e in stored.events if e.id == event.id):
            return event  # re-delivery: drop, like InMemorySessionService
        # Apply temp/state-delta logic to the caller's session object.
        await super().append_event(session=session, event=event)
        session.last_update_time = event.timestamp or _now()
        if event.actions and event.actions.state_delta:
            deltas = session_util.extract_state_delta(event.actions.state_delta)
            self._store_shared_deltas(session.app_name, session.user_id, deltas)
        self._save(session)  # scrubbed copy, sliding TTL refresh
        return event
