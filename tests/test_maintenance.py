"""Phase 4 tests: the forgetting layer.

Unit tests run fully offline (fakes everywhere). The forgetting drill is
the acceptance test: aged-out data is seeded into every store, maintenance
runs, the data is provably gone, and the eval set still passes. The
real-backend variant is marked integration and self-skips without Docker.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from support_agent import config, maintenance, rules
from support_agent.audit import CARD_RE
from support_agent.backends import git_rules
from support_agent.backends.pgvector_memory import (
    FakeEmbedder,
    FakeEpisodeStore,
    PgVectorMemoryService,
)
from support_agent.backends.redis_sessions import RedisSessionService
from support_agent.memory_banks import Fact, SemanticBank

from google.adk.events import Event
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(coro):
    return asyncio.run(coro)


def _uid(prefix="mtest"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _old_iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


# --- policy validation --------------------------------------------------------


def _temp_policy(tmp_path, **overrides) -> Path:
    """Copy the real policy.yaml with overrides applied."""
    lines = Path("rules/policy.yaml").read_text().splitlines()
    out = []
    seen = set()
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if stripped and ":" in stripped:
            key = stripped.partition(":")[0].strip()
            if key in overrides:
                out.append(f"{key}: {overrides[key]}")
                seen.add(key)
                continue
        out.append(line)
    for key, value in overrides.items():
        if key not in seen:
            out.append(f"{key}: {value}")
    path = tmp_path / "policy.yaml"
    path.write_text("\n".join(out) + "\n")
    return path


def test_policy_lifecycle_keys_validated(tmp_path):
    base = _temp_policy(tmp_path)
    assert git_rules.load_policy(base)["session_abandon_days"] == 14
    for bad in [
        {"session_abandon_days": 0},
        {"episodic_summary_days": -1},
        {"fact_min_confidence": 1.5},
        {"fact_reconfirm_days": 0},
        {"superseded_retention_days": 0},
        {"deprecated_no_such_rule": True},
        {"deprecated_escalation": "maybe"},
        {"llm_cost_per_mtok_input_usd": -0.1},
    ]:
        with pytest.raises(ValueError):
            git_rules.load_policy(_temp_policy(tmp_path, **bad))


def test_deprecated_rule_does_not_fire(monkeypatch):
    monkeypatch.setattr(
        config, "DEPRECATED_RULES", frozenset({"late_delivery_credit"})
    )
    out = rules.late_delivery_credit({"order_id": "ORD-1"})
    assert out["deprecated"] is True
    assert out["decision"] == "not_fired"
    # Active rules are unaffected.
    assert rules.evaluate_refund_request(10.0)["decision"] == "approve"


def test_render_drops_deprecated_prompt_block(tmp_path):
    path = _temp_policy(tmp_path, deprecated_late_delivery_credit="true")
    policy = git_rules.load_policy(path)
    assert git_rules.deprecated_rules(policy) == frozenset({"late_delivery_credit"})
    rendered = git_rules.render_system_prompt(policy)
    assert "goodwill credit the rules compute" not in rendered
    assert "may be issued through request_refund" in rendered  # others kept


def test_check_deprecated_rules_empty_by_default():
    assert maintenance.check_deprecated_rules()["deprecated_checks"] == {}


# --- sessions -----------------------------------------------------------------


def _make_redis_stack(ttl=30 * 24 * 3600):
    from tests.test_backends_redis import FakeRedis

    fake = FakeRedis()
    return fake, RedisSessionService(fake, ttl_seconds=ttl)


def _backdate_session(fake, app, user, sid, days_ago):
    key = RedisSessionService._skey(app, user, sid)
    raw = json.loads(fake.strings[key].decode("utf-8"))
    raw["last_update_time"] = time.time() - days_ago * 86400
    fake.strings[key] = json.dumps(raw).encode("utf-8")


def test_find_abandoned_splits_unknown_age():
    class S:
        def __init__(self, sid, lut):
            self.id = sid
            self.last_update_time = lut

    now = time.time()
    old = S("old", now - 20 * 86400)
    fresh = S("fresh", now - 3600)
    unknown = S("unknown", None)
    abandoned, unk = maintenance.find_abandoned([old, fresh, unknown], 14, now=now)
    assert [s.id for s in abandoned] == ["old"]
    assert unk == ["unknown"]


def test_compact_sessions_dry_run_then_execute():
    async def go():
        fake, svc = _make_redis_stack()
        app, user = "a", _uid()
        old = await svc.create_session(app_name=app, user_id=user, session_id="old")
        fresh = await svc.create_session(app_name=app, user_id=user, session_id="new")
        _backdate_session(fake, app, user, "old", days_ago=20)

        dry = await maintenance.compact_sessions(
            svc, app, user, abandon_days=14, dry_run=True
        )
        assert dry["dry_run"] is True
        assert dry["abandoned"] == ["old"]
        assert dry["deleted"] == 0
        # Dry run changes nothing.
        assert await svc.get_session(app_name=app, user_id=user, session_id="old") is not None

        done = await maintenance.compact_sessions(
            svc, app, user, abandon_days=14, dry_run=False
        )
        assert done["deleted"] == 1
        assert await svc.get_session(app_name=app, user_id=user, session_id="old") is None
        assert await svc.get_session(app_name=app, user_id=user, session_id="new") is not None

    run(go())


# --- episodic summarization ---------------------------------------------------


def _seed_episodes(store, app, user, dim=8):
    """3 old episodes (100d) + 1 recent; one old episode holds a card number."""
    for i, (days, text) in enumerate(
        [
            (100, "ORD-1001 arrived 5 days late, customer upset"),
            (100, "ORD-1001 late again, card 4111 1111 1111 1111 mentioned"),
            (100, "ORD-1002 damaged item, refund issued"),
            (1, "ORD-1003 tracking requested"),
        ]
    ):
        store.insert_episode(
            app_name=app,
            user_id=user,
            session_id=f"s{i}",
            author="user",
            ts=_old_iso(days),
            text=text,
            embedding=[0.0] * dim,
            meta={},
        )


def test_summarize_episodes_dry_run_changes_nothing():
    async def go():
        store = FakeEpisodeStore()
        app, user = "a", _uid()
        _seed_episodes(store, app, user)
        semantic = SemanticBank()
        rep = await maintenance.summarize_episodes(
            store, semantic, maintenance.FakeSummarizer(), app, user,
            summary_days=90, dry_run=True,
        )
        assert rep["dry_run"] is True
        assert rep["candidates"] == 3
        assert rep["dropped"] == 3  # would drop
        assert len(store.list_episodes_older_than(
            app_name=app, user_id=user, cutoff_iso=_old_iso(90))) == 3
        assert semantic.active_facts() == []

    run(go())


def test_summarize_episodes_drops_old_keeps_recent_and_scrubs_pii():
    async def go():
        store = FakeEpisodeStore()
        app, user = "a", _uid()
        _seed_episodes(store, app, user)
        semantic = SemanticBank()
        rep = await maintenance.summarize_episodes(
            store, semantic, maintenance.FakeSummarizer(), app, user,
            summary_days=90, dry_run=False,
        )
        assert rep["dropped"] == 3
        # Old episodes provably gone; recent one kept.
        remaining = store.list_episodes_older_than(
            app_name=app, user_id=user, cutoff_iso=_old_iso(90))
        assert remaining == []
        recent = store.list_episodes_older_than(
            app_name=app, user_id=user, cutoff_iso=_old_iso(0))
        assert len(recent) == 1
        assert "ORD-1003" in recent[0]["text"]
        # One rolling summary fact, superseding nothing yet.
        facts = [f for f in semantic.active_facts()
                 if f.slot == maintenance.SUMMARY_SLOT]
        assert len(facts) == 1
        assert "3 past customer support interactions" in facts[0].text
        # PII never leaked into the summary.
        assert not CARD_RE.search(facts[0].text)
        assert "4111" not in facts[0].text

    run(go())


def test_fake_summarizer_deterministic():
    s = maintenance.FakeSummarizer()
    texts = ["ORD-1001 arrived late again", "ORD-1001 late delivery"]
    assert s.summarize(texts) == s.summarize(texts)
    assert "2 past customer support interactions" in s.summarize(texts)


def test_episodic_lifecycle_unsupported_for_memory_backend():
    from google.adk.memory import InMemoryMemoryService

    ok, reason = maintenance.episodic_lifecycle_supported(InMemoryMemoryService())
    assert ok is False
    assert "pgvector" in reason


def test_episodic_lifecycle_supported_for_pgvector_store():
    svc = PgVectorMemoryService(FakeEpisodeStore(), FakeEmbedder(dim=8))
    ok, _ = maintenance.episodic_lifecycle_supported(svc)
    assert ok is True


# --- semantic review ----------------------------------------------------------


def _seed_semantic():
    bank = SemanticBank()
    bank.upsert("contact_preference", "prefers email updates")  # healthy
    bank.upsert("low_conf", "maybe prefers sms", confidence=0.2)  # flagged
    bank.upsert("old_slot", "first version")
    bank.upsert("old_slot", "second version")  # supersedes first
    # Backdate the superseded fact beyond retention.
    for f in bank.history():
        if f.slot == "old_slot" and f.superseded_by is not None:
            f.updated_at = _old_iso(200)
    # Expired fact (white-box: upsert has no expires param by design).
    bank._facts.append(Fact(slot="gone", text="old news", expires="2020-01-01"))
    return bank


def test_semantic_review_classifies_correctly():
    bank = _seed_semantic()
    items = maintenance.review_semantic(bank)
    by_action = {}
    for i in items:
        by_action.setdefault(i.action, []).append(i.slot)
    assert by_action["purge_expired"] == ["gone"]
    assert by_action["purge_superseded"] == ["old_slot"]
    assert "low_conf" in by_action["flag_low_confidence"]
    assert "contact_preference" not in {i.slot for i in items}


def test_apply_semantic_review_purges_but_only_drafts_flags():
    bank = _seed_semantic()
    items = maintenance.review_semantic(bank)
    rep = maintenance.apply_semantic_review(bank, items, dry_run=False)
    assert rep["purged_expired"] == 1
    assert rep["purged_superseded"] == 1
    slots = {f.slot for f in bank.active_facts()}
    assert "gone" not in slots
    # Low-confidence fact is flagged, NOT purged.
    assert "low_conf" in slots
    assert any(f["action"] == "flag_low_confidence" for f in rep["flags"])
    # Healthy fact untouched.
    assert "contact_preference" in slots


def test_stale_fact_flagged_and_reconfirm_clears_it():
    bank = SemanticBank()
    bank.upsert("stale_slot", "old preference")
    for f in bank.history():
        if f.slot == "stale_slot":
            f.last_confirmed = _old_iso(200)[:10]
            f.created_at = _old_iso(200)
    items = maintenance.review_semantic(bank)
    assert any(i.action == "flag_stale" and i.slot == "stale_slot" for i in items)
    assert bank.reconfirm("stale_slot")["reconfirmed"] is True
    items2 = maintenance.review_semantic(bank)
    assert not any(i.slot == "stale_slot" for i in items2)


def test_conflicted_slot_flagged():
    bank = SemanticBank()
    bank._facts.append(Fact(slot="dup", text="one"))
    bank._facts.append(Fact(slot="dup", text="two"))
    items = maintenance.review_semantic(bank)
    assert sum(1 for i in items if i.action == "flag_conflicted") == 2


# --- metrics ------------------------------------------------------------------


def _synthetic_audit_events():
    base = "2026-10-11T10:00:00+00:00"
    return [
        {"ts": base, "component": "support_agent", "event": "turn",
         "session_id": "s1", "prompt_chars": 40},
        {"ts": base, "component": "support_agent", "event": "memory_recall",
         "session_id": "s1", "injected": True, "block_chars": 120,
         "episodic_chars": 80, "semantic_chars": 40,
         "episodic_hits": 2, "semantic_hits": 1},
        {"ts": base, "component": "support_agent", "event": "turn",
         "session_id": "s1", "prompt_chars": 60},
        {"ts": base, "component": "support_agent", "event": "memory_recall",
         "session_id": "s1", "injected": False, "block_chars": 0,
         "episodic_chars": 0, "semantic_chars": 0,
         "episodic_hits": 0, "semantic_hits": 0},
    ]


def test_audit_metrics_computation():
    m = maintenance.audit_metrics(_synthetic_audit_events())
    assert m["turns"] == 2
    assert m["tickets"] == 1
    assert m["recall"]["overall_hit_rate"] == 0.5
    assert m["recall"]["episodic_hit_rate"] == 0.5
    assert m["recall"]["semantic_hit_rate"] == 0.5
    assert m["recall"]["avg_episodic_chars"] == 40.0
    assert m["recall"]["avg_episodic_tokens_est"] == 10.0
    assert m["est_input_cost_usd_per_turn"] > 0
    assert "input-side estimate" in m["cost_note"]


def test_retrieval_probes_precision():
    rep = run(maintenance.run_retrieval_probes())
    assert rep["precision_at_k"] == 1.0
    assert rep["hits"] == rep["probes"] == 3


def test_maintenance_report_renders(tmp_path):
    path = tmp_path / "audit.jsonl"
    with open(path, "w") as f:
        for e in _synthetic_audit_events():
            f.write(json.dumps(e) + "\n")
    rep = maintenance.report(str(path), precision=True)
    text = maintenance.render_report(rep)
    assert "turns: 2" in text
    assert "recall hit rate: overall 50%" in text
    assert "retrieval precision@5: 100%" in text
    assert "est. input cost" in text


def test_run_requires_user():
    with pytest.raises(ValueError, match="--user is required"):
        maintenance.run(user_id=None, dry_run=True)


def test_run_dry_run_offline_defaults():
    rep = maintenance.run(user_id=_uid(), dry_run=True)
    assert rep["dry_run"] is True
    assert rep["layers"]["session"]["deleted"] == 0
    assert rep["layers"]["episodic"]["supported"] is False  # in-memory backend
    assert rep["layers"]["procedural"]["deprecated"] == []


# --- the forgetting drill (acceptance) ----------------------------------------


async def _drill_seed(app, user):
    """Seed aged-out data in every store. Returns the handles."""
    fake_redis, session_svc = _make_redis_stack()
    await session_svc.create_session(
        app_name=app, user_id=user, session_id="drill-old")
    _backdate_session(fake_redis, app, user, "drill-old", days_ago=20)
    await session_svc.create_session(
        app_name=app, user_id=user, session_id="drill-fresh")

    store = FakeEpisodeStore()
    _seed_episodes(store, app, user)

    semantic = SemanticBank()
    semantic.upsert("contact_preference", "prefers email updates")
    semantic.upsert("low_conf", "maybe prefers sms", confidence=0.2)
    semantic._facts.append(Fact(slot="gone", text="old news", expires="2020-01-01"))
    return fake_redis, session_svc, store, semantic


def _drill_assert_gone(app, user, session_svc, store, semantic):
    async def go():
        assert await session_svc.get_session(
            app_name=app, user_id=user, session_id="drill-old") is None
        assert await session_svc.get_session(
            app_name=app, user_id=user, session_id="drill-fresh") is not None
        assert store.list_episodes_older_than(
            app_name=app, user_id=user, cutoff_iso=_old_iso(90)) == []
        slots = {f.slot for f in semantic.active_facts()}
        assert "gone" not in slots
        assert "low_conf" in slots  # flagged, not purged
        summaries = [f for f in semantic.active_facts()
                     if f.slot == maintenance.SUMMARY_SLOT]
        assert len(summaries) == 1
        assert not CARD_RE.search(summaries[0].text)
    run(go())


def test_forgetting_drill():
    """Acceptance: aged-out data is provably gone from every store."""
    app, user = "drill_app", _uid("drill")

    async def execute():
        fake_redis, session_svc, store, semantic = await _drill_seed(app, user)
        r1 = await maintenance.compact_sessions(
            session_svc, app, user, abandon_days=14, dry_run=False)
        assert r1["deleted"] == 1
        r2 = await maintenance.summarize_episodes(
            store, semantic, maintenance.FakeSummarizer(), app, user,
            summary_days=90, dry_run=False)
        assert r2["dropped"] == 3
        items = maintenance.review_semantic(semantic)
        r3 = maintenance.apply_semantic_review(semantic, items, dry_run=False)
        assert r3["purged_expired"] == 1
        return session_svc, store, semantic

    session_svc, store, semantic = run(execute())
    _drill_assert_gone(app, user, session_svc, store, semantic)


def test_forgetting_drill_then_eval_still_passes():
    """The drill must not break the agent: the eval set still passes."""
    app, user = "drill_app2", _uid("drill")
    async def execute():
        _, session_svc, store, semantic = await _drill_seed(app, user)
        await maintenance.compact_sessions(
            session_svc, app, user, abandon_days=14, dry_run=False)
        await maintenance.summarize_episodes(
            store, semantic, maintenance.FakeSummarizer(), app, user,
            summary_days=90, dry_run=False)
        items = maintenance.review_semantic(semantic)
        maintenance.apply_semantic_review(semantic, items, dry_run=False)
    run(execute())

    # The drill used isolated stores; the eval suite runs against the
    # default in-memory backends and must stay green.
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("SESSION_BACKEND", "EPISODIC_BACKEND",
                     "SEMANTIC_BACKEND", "EMBEDDER", "SUMMARIZER")
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_eval.py", "-q",
         "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


# --- integration: the drill against real Docker backends ----------------------


def _real_session_service():
    import redis

    client = redis.Redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        socket_connect_timeout=3, socket_timeout=3,
    )
    client.ping()
    return client, RedisSessionService(client, ttl_seconds=30 * 24 * 3600)


def _real_episode_store(dim=8):
    from support_agent.backends.pgvector_memory import PgVectorEpisodeStore

    store = PgVectorEpisodeStore(
        os.environ.get(
            "PGVECTOR_DSN", "postgresql://memory:memory@localhost:5432/memory"
        ),
        dim,
    )
    store.ping()
    store.ensure_schema()
    return store


def _real_semantic_bank(app, user):
    from google.cloud import firestore

    from support_agent.backends.firestore_semantics import FirestoreSemanticBank

    bank = FirestoreSemanticBank(
        firestore.Client(project=os.environ.get(
            "FIRESTORE_PROJECT", "demo-memory-agent")),
        app, user,
    )
    bank.ping()
    return bank


@pytest.mark.integration
def test_forgetting_drill_real_backends():
    """Same drill as offline, against the real Docker backends.

    Run with:
      docker compose up -d
      SESSION_BACKEND=redis EPISODIC_BACKEND=pgvector SEMANTIC_BACKEND=firestore \
        FIRESTORE_EMULATOR_HOST=localhost:8080 \
        ~/workspace/.venv-adk/bin/python -m pytest tests/test_maintenance.py -q -m integration
    """
    app, user = "drill_app", _uid("idrill")
    layers = {}

    try:
        client, session_svc = _real_session_service()
        layers["session"] = (client, session_svc)
    except Exception as e:
        pytest.skip(f"redis unreachable: {e}")
    try:
        layers["episodic"] = _real_episode_store()
    except Exception as e:
        pytest.skip(f"pgvector unreachable: {e}")
    try:
        layers["semantic"] = _real_semantic_bank(app, user)
    except Exception as e:
        pytest.skip(f"firestore unreachable: {e}")

    async def go():
        # --- session layer: an abandoned session is compacted --------------
        client, session_svc = layers["session"]
        sid = f"old-{uuid.uuid4().hex[:6]}"
        await session_svc.create_session(app_name=app, user_id=user, session_id=sid)
        key = RedisSessionService._skey(app, user, sid)
        raw = json.loads(client.get(key).decode("utf-8"))
        raw["last_update_time"] = time.time() - 20 * 86400
        client.set(key, json.dumps(raw))
        rep = await maintenance.compact_sessions(
            session_svc, app, user, abandon_days=14, dry_run=False)
        assert rep["deleted"] >= 1
        assert await session_svc.get_session(
            app_name=app, user_id=user, session_id=sid) is None

        # --- episodic layer: old episodes summarized + dropped -------------
        store = layers["episodic"]
        for i in range(2):
            store.insert_episode(
                app_name=app, user_id=user, session_id=f"is{i}",
                author="user", ts=_old_iso(100),
                text=f"integration old episode {i} about ORD-1001",
                embedding=[0.0] * 8, meta={})
        semantic = layers["semantic"]
        rep = await maintenance.summarize_episodes(
            store, semantic, maintenance.FakeSummarizer(), app, user,
            summary_days=90, dry_run=False)
        assert rep["dropped"] == 2
        assert store.list_episodes_older_than(
            app_name=app, user_id=user, cutoff_iso=_old_iso(90)) == []
        assert any(f.slot == maintenance.SUMMARY_SLOT
                   for f in semantic.active_facts())

        # --- semantic layer: expired facts purged --------------------------
        raw_doc = semantic._read()
        raw_doc.append({
            "slot": "idrill-gone", "text": "expired", "confidence": 1.0,
            "created_at": _old_iso(200), "updated_at": _old_iso(200),
            "superseded_by": None, "expires": "2020-01-01",
            "last_confirmed": _old_iso(200),
        })
        semantic._write(raw_doc)
        items = maintenance.review_semantic(semantic)
        rep = maintenance.apply_semantic_review(semantic, items, dry_run=False)
        assert rep["purged_expired"] >= 1
        assert "idrill-gone" not in {f.slot for f in semantic.active_facts()}

    run(go())
