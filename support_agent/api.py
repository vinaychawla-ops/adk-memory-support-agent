"""Phase 3: the agent as an HTTP service.

`POST /chat` runs one user turn through the ADK Runner; `GET /health`
reports liveness plus per-backend reachability. The runner, session
service, and memory service come from the existing `make_runner()`
factories, so backend selection stays env-driven (in-memory by default,
Redis/pgvector/Firestore via env -- see AGENTS.md).

Design notes:
- One Runner per process; banks and audit logs are registered per user
  (both idempotent), so any replica serves any user -- no sticky sessions.
- After each turn, only the *new* events are written to episodic memory.
  The count already recorded lives in session state
  (`episodic_recorded_upto`), bumped via a bookkeeping event, so a
  restart or a different replica never double-records.
- PII: user text reaches Redis only through the scrubbing session
  service, episodes are scrubbed in `EpisodicBank.record_session`, and
  the audit log stores scrubbed previews + hashes (see audit.py).
- The live turn needs GOOGLE_API_KEY at run time; constructing the app
  (and /health) does not.

Run locally (needs the venv; Docker for the durable backends):
    uvicorn support_agent.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import Session
from google.genai import types
from pydantic import BaseModel, Field

from support_agent import agent as agent_module
from support_agent import config
from support_agent.audit import get_audit, register_audit
from support_agent.memory_banks import get_banks

# Bookkeeping key: how many session events are already in episodic memory.
RECORDED_UPTO_KEY = "episodic_recorded_upto"


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    reply: str
    model: str
    replica_id: str
    backends: dict[str, str]
    memory_recall: bool
    events_recorded: int


REPLICA_ID = os.environ.get("REPLICA_ID", "local")


def _backend_names() -> dict[str, str]:
    return {
        "session": config.SESSION_BACKEND,
        "episodic": config.EPISODIC_BACKEND,
        "semantic": config.SEMANTIC_BACKEND,
        "embedder": config.EMBEDDER,
    }


def _default_runner_factory():
    """(runner, session_service, memory_service, agent), no key needed."""
    return agent_module.make_runner()


async def _record_new_events(session_service, session) -> int:
    """Write this turn's new events to episodic memory (deduplicated).

    Each turn's delta is stored under a distinct episode id
    ("<session_id>#e<offset>"): the in-memory backend keys episodes by
    session id, so reusing the session id would overwrite the previous
    turn's episode; pgvector accumulates rows either way. The id is
    deterministic in the recorded offset, so a retry can never duplicate.

    Returns the number of events recorded. The caller's session object is
    refreshed by the caller afterwards if it needs the new counter.
    """
    banks = get_banks(session.app_name, session.user_id)
    events = list(session.events or [])
    try:
        recorded_upto = int(session.state.get(RECORDED_UPTO_KEY) or 0)
    except (TypeError, ValueError):
        recorded_upto = 0
    new_events = events[recorded_upto:]
    if new_events and banks is not None:
        partial = Session(
            app_name=session.app_name,
            user_id=session.user_id,
            id=f"{session.id}#e{recorded_upto}",
            state={},
            events=new_events,
        )
        # record_session PII-scrubs before storage (memory_banks.py).
        await banks.episodic.record_session(partial)
    # Bump the counter past the new events AND this bookkeeping event, so
    # neither is ever recorded twice -- by this or any other replica.
    marker = Event(author="system", actions=EventActions(state_delta={RECORDED_UPTO_KEY: len(events) + 1}))
    await session_service.append_event(session, marker)
    return len(new_events) if banks is not None else 0


def _final_reply(events, agent_name: str) -> str:
    """Concatenate the agent's final text parts from a run's events."""
    chunks: list[str] = []
    for event in events or []:
        if event.author != agent_name or not event.content:
            continue
        for part in event.content.parts or []:
            if part.text:
                chunks.append(part.text)
    return "".join(chunks).strip()


def create_app(runner_factory=_default_runner_factory) -> FastAPI:
    """Build the FastAPI app. runner_factory is injectable for tests."""
    app = FastAPI(title="memory-support-agent", version="3")
    services: dict = {}

    def _services():
        if not services:
            runner, session_service, memory_service, agent = runner_factory()
            services.update(
                runner=runner,
                session_service=session_service,
                memory_service=memory_service,
                agent=agent,
            )
        return services

    @app.get("/health")
    async def health():
        # Never raise: every backend probe is best-effort with a timeout.
        # In-memory selections have no ping -> reported as "n/a".
        status: dict[str, str] = {}

        async def _probe(name: str, ping):
            if ping is None:
                return "n/a (in-memory)"
            try:
                if asyncio.iscoroutinefunction(ping):
                    await asyncio.wait_for(ping(), timeout=3)
                else:
                    await asyncio.to_thread(ping)
                return "up"
            except Exception:
                return "down"

        try:
            svcs = _services()
            session_service = svcs["session_service"]
            memory_service = svcs["memory_service"]
            user_probe = getattr(session_service, "ping", None)
            mem_probe = getattr(memory_service, "ping", None)
            status["session"] = await _probe("session", user_probe)
            status["episodic"] = await _probe("episodic", mem_probe)
            # Semantic store needs a user to construct; report selection only.
            status["semantic"] = f"selected: {config.SEMANTIC_BACKEND}"
        except Exception as e:  # noqa: BLE001 -- health must stay total
            status["error"] = str(e)[:200]
        return {
            "status": "ok",
            "model": config.MODEL,
            "replica_id": REPLICA_ID,
            "backends": _backend_names(),
            "backend_status": status,
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        try:
            svcs = _services()
            runner = svcs["runner"]
            session_service = svcs["session_service"]
            memory_service = svcs["memory_service"]
            agent = svcs["agent"]

            # Per-user wiring; both are idempotent, so any replica can
            # serve any user on any turn.
            agent_module.init_user_memory(config.APP_NAME, req.user_id, memory_service)
            register_audit(config.APP_NAME, req.user_id)

            session = await session_service.get_session(
                app_name=config.APP_NAME,
                user_id=req.user_id,
                session_id=req.session_id,
            )
            if session is None:
                session = await session_service.create_session(
                    app_name=config.APP_NAME,
                    user_id=req.user_id,
                    session_id=req.session_id,
                )

            seen: list = []
            async for event in runner.run_async(
                user_id=req.user_id,
                session_id=req.session_id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=req.message)]
                ),
            ):
                seen.append(event)
            reply = _final_reply(seen, agent.name)

            # Refresh the session (the run appended events), then bank the
            # new ones in episodic memory exactly once.
            session = await session_service.get_session(
                app_name=config.APP_NAME,
                user_id=req.user_id,
                session_id=req.session_id,
            )
            events_recorded = await _record_new_events(session_service, session)
            memory_recall = bool((session.state or {}).get("memory_context"))

            return ChatResponse(
                user_id=req.user_id,
                session_id=req.session_id,
                reply=reply,
                model=config.MODEL,
                replica_id=REPLICA_ID,
                backends=_backend_names(),
                memory_recall=memory_recall,
                events_recorded=events_recorded,
            )
        except Exception as e:  # noqa: BLE001 -- API errors, never tracebacks
            return JSONResponse(status_code=500, content={"error": str(e)[:500]})

    return app


app = create_app()
