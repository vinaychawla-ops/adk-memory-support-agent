"""Phase 3 API tests -- fully offline (no key, no Docker).

The FastAPI app is built with an injected fake runner; the session and
memory services are the real in-memory ADK services, so the session
lifecycle, episodic recording/dedupe, PII scrubbing, and audit wiring are
exercised for real. The only fake is the LLM turn itself.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from google.adk.events import Event
from google.genai import types

from support_agent import agent as agent_module
from support_agent import config
from support_agent.api import create_app
from support_agent.audit import clear_audit_registry, get_audit
from support_agent.memory_banks import clear_registry, get_banks


class FakeRunner:
    """Stands in for ADK Runner: appends user+agent events, yields the reply.

    Mirrors what Runner.run_async does (events land in the session via the
    session service) so the API's session/episodic/audit wiring is tested
    for real. state_delta, when given, is applied on the user event --
    simulating a tool that wrote structured session state.
    """

    def __init__(self, session_service, reply="canned reply", state_delta=None,
                 fail=None):
        self._ss = session_service
        self._reply = reply
        self._state_delta = state_delta
        self._fail = fail

    async def run_async(self, *, user_id, session_id, new_message=None, **kwargs):
        if self._fail is not None:
            raise self._fail
        session = await self._ss.get_session(
            app_name=config.APP_NAME, user_id=user_id, session_id=session_id
        )
        user_event = Event(author="user", content=new_message)
        if self._state_delta:
            user_event.actions.state_delta = dict(self._state_delta)
        await self._ss.append_event(session, user_event)
        agent_event = Event(
            author="support_agent",
            content=types.Content(
                role="model", parts=[types.Part(text=self._reply)]
            ),
        )
        await self._ss.append_event(session, agent_event)
        yield agent_event


def _shared_services():
    from google.adk.memory import InMemoryMemoryService
    from google.adk.sessions import InMemorySessionService

    return InMemorySessionService(), InMemoryMemoryService()


def _app_with(session_service, memory_service, **runner_kwargs):
    agent = agent_module.build_agent()
    runner = FakeRunner(session_service, **runner_kwargs)
    return create_app(lambda: (runner, session_service, memory_service, agent))


@pytest.fixture()
def app_services():
    clear_registry()
    clear_audit_registry()
    ss, ms = _shared_services()
    app = _app_with(ss, ms)
    return TestClient(app), ss, ms


def _chat(client, user="u1", session="s1", message="hello"):
    return client.post(
        "/chat", json={"user_id": user, "session_id": session, "message": message}
    )


def test_health_ok_without_key(app_services):
    client, _, _ = app_services
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == config.MODEL
    assert set(body["backends"]) >= {"session", "episodic", "semantic"}
    # Default env = in-memory selections, which have no ping.
    assert body["backend_status"]["session"] == "n/a (in-memory)"
    assert body["backend_status"]["episodic"] == "n/a (in-memory)"


def test_chat_turn_shape_and_episodic_dedupe(app_services):
    client, ss, _ = app_services
    r = _chat(client, message="track my order ORD-1001")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == "u1" and body["session_id"] == "s1"
    assert body["reply"] == "canned reply"
    assert body["model"] == config.MODEL
    assert body["replica_id"] == "local"
    assert body["events_recorded"] == 2  # user + agent events

    # Second turn records only its own 2 new events -- never double-records.
    r2 = _chat(client, message="any update?")
    assert r2.status_code == 200
    assert r2.json()["events_recorded"] == 2

    async def check():
        s = await ss.get_session(
            app_name=config.APP_NAME, user_id="u1", session_id="s1"
        )
        # 2 turns x (user + agent) + 2 bookkeeping events = 6 total.
        assert len(s.events) == 6
        assert s.state["episodic_recorded_upto"] == 6

    asyncio.run(check())


def test_chat_validation(app_services):
    client, _, _ = app_services
    assert client.post("/chat", json={}).status_code == 422
    assert client.post(
        "/chat", json={"user_id": "u", "session_id": "s", "message": ""}
    ).status_code == 422


def test_chat_runner_error_is_500_without_traceback():
    clear_registry()
    clear_audit_registry()
    ss, ms = _shared_services()
    app = _app_with(ss, ms, fail=RuntimeError("boom"))
    r = TestClient(app).post(
        "/chat", json={"user_id": "u", "session_id": "s", "message": "hi"}
    )
    assert r.status_code == 500
    assert r.json() == {"error": "boom"}
    assert "Traceback" not in r.text


def test_interleaved_replicas_share_everything():
    """Phase 3 acceptance (offline): two replicas, one shared backend.

    Replica A takes turn 1 (a tool writes structured session state);
    replica B takes turn 2 for the same user/session. Nothing is lost:
    B sees the state, the history, and the episodic memory from A.
    """
    clear_registry()
    clear_audit_registry()
    ss, ms = _shared_services()
    client_a = TestClient(
        _app_with(ss, ms, state_delta={"active_order_id": "ORD-1001"})
    )
    client_b = TestClient(_app_with(ss, ms))

    r1 = client_a.post(
        "/chat",
        json={"user_id": "u9", "session_id": "s9", "message": "track my order"},
    )
    assert r1.status_code == 200

    r2 = client_b.post(
        "/chat",
        json={"user_id": "u9", "session_id": "s9", "message": "any update?"},
    )
    assert r2.status_code == 200
    # B recorded only its own 2 new events (A's were already banked).
    assert r2.json()["events_recorded"] == 2

    async def check():
        s = await ss.get_session(
            app_name=config.APP_NAME, user_id="u9", session_id="s9"
        )
        # Structured state written on A's turn is visible to B.
        assert s.state["active_order_id"] == "ORD-1001"
        # History from both turns is intact.
        texts = [
            p.text
            for e in s.events
            for p in (e.content.parts if e.content else [])
            if p.text
        ]
        assert "track my order" in texts and "any update?" in texts
        # Episodic memory (shared service) holds A's turn for B's recall.
        banks = get_banks(config.APP_NAME, "u9")
        hits = await banks.episodic.recall("track my order")
        assert any("track my order" in h for h in hits)

    asyncio.run(check())


def test_audit_receives_episodic_write(app_services):
    client, _, _ = app_services
    _chat(client, message="hello")
    log = get_audit(config.APP_NAME, "u1")
    assert log is not None
    kinds = [e["event"] for e in log.events]
    assert "memory_write" in kinds


def test_pii_never_reaches_episodic_layer(app_services):
    client, _, _ = app_services
    _chat(client, message="my card is 4111 1111 1111 1111 please refund")
    banks = get_banks(config.APP_NAME, "u1")

    async def check():
        hits = await banks.episodic.recall("card")
        assert hits, "the scrubbed episode should still be found"
        assert all("4111" not in h for h in hits)

    asyncio.run(check())
