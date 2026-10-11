"""AuditLog: previews, hashes, PII scrubbing, JSONL, registry."""

import json

import pytest

from support_agent import audit as audit_module
from support_agent.audit import (
    AuditLog,
    clear_audit_registry,
    get_audit,
    register_audit,
)


@pytest.fixture(autouse=True)
def clean_registry():
    clear_audit_registry()
    yield
    clear_audit_registry()


def test_record_stamps_ts_and_component():
    log = AuditLog()
    evt = log.record({"event": "turn", "session_id": "s1"})
    assert evt["ts"]
    assert evt["component"] == "support_agent"
    assert evt["event"] == "turn"
    assert len(log) == 1


def test_record_writes_jsonl(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = AuditLog(path=path)
    log.record({"event": "turn"})
    log.record({"event": "tool_call"})
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert [e["event"] for e in lines] == ["turn", "tool_call"]
    assert all(e["ts"] for e in lines)


def test_no_file_without_path(tmp_path):
    log = AuditLog()
    log.record({"event": "turn"})
    assert list(tmp_path.iterdir()) == []


def test_prompt_preview_truncates():
    long_prompt = "x" * 500
    assert len(audit_module.prompt_preview(long_prompt)) == 200


def test_prompt_hash_stable_and_short():
    h1 = audit_module.prompt_hash("hello")
    h2 = audit_module.prompt_hash("hello")
    assert h1 == h2 and len(h1) == 16
    assert audit_module.prompt_hash("different") != h1


def test_scrub_pii_redacts_all_groupings():
    for text in (
        "card 4111 1111 1111 1111",
        "card 4111-1111-1111-1111",
        "card 4111111111111111",
    ):
        scrubbed = audit_module.scrub_pii(text)
        assert "4111" not in scrubbed
        assert "[REDACTED]" in scrubbed
    # Non-card numbers are untouched.
    assert audit_module.scrub_pii("order ORD-1001") == "order ORD-1001"


def test_turn_preview_never_carries_card_number():
    log = AuditLog()
    audit_module.log_turn("a", "u", "s1", "my card is 4111 1111 1111 1111 please")
    # No log registered for ("a", "u") -> no-op; register and retry.
    register_audit("a", "u")
    audit_module.log_turn("a", "u", "s1", "my card is 4111 1111 1111 1111 please")
    evt = get_audit("a", "u").events[0]
    assert "4111" not in evt["prompt_preview"]
    assert "[REDACTED]" in evt["prompt_preview"]
    assert len(evt["prompt_hash"]) == 16


def test_helpers_noop_without_registered_log():
    # Must never raise, must return None.
    assert audit_module.log_turn("a", "ghost", "s", "hi") is None
    assert audit_module.log_tool_call("a", "ghost", "s", "t", {}) is None
    assert audit_module.log_tool_result("a", "ghost", "s", "t", {}) is None
    assert audit_module.log_memory_write("a", "ghost", "semantic", {}) is None
    assert audit_module.log_rule_fired("a", "ghost", "s", "r", "d", "") is None
    assert audit_module.log_memory_recall("a", "ghost", "s", True, 10) is None


def test_registry_is_per_user():
    log_a = register_audit("app", "alice")
    log_b = register_audit("app", "bob", path=None)
    assert get_audit("app", "alice") is log_a
    assert get_audit("app", "bob") is log_b
    assert get_audit("app", "carol") is None
    assert register_audit("app", "alice") is log_a  # idempotent


def test_summary_counts_by_event():
    log = register_audit("app", "u")
    audit_module.log_turn("app", "u", "s", "hi")
    audit_module.log_tool_call("app", "u", "s", "lookup_order", {"order_id": "ORD-1001"})
    audit_module.log_tool_result("app", "u", "s", "lookup_order", {"order_id": "ORD-1001"})
    s = log.summary()
    assert s["total"] == 3
    assert s["by_event"] == {"turn": 1, "tool_call": 1, "tool_result": 1}


def test_tool_result_marks_error_status():
    log = register_audit("app", "u")
    audit_module.log_tool_result("app", "u", "s", "lookup_order", {"error": "nope"})
    assert log.events[0]["status"] == "error"
    audit_module.log_tool_result("app", "u", "s", "lookup_order", {"order_id": "x"})
    assert log.events[1]["status"] == "ok"


def test_replay_reads_jsonl_in_order(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    log = register_audit("app", "u", path=path)
    audit_module.log_turn("app", "u", "s1", "first")
    audit_module.log_tool_call("app", "u", "s1", "t", {})
    replayed = log.replay()
    assert [e["event"] for e in replayed] == ["turn", "tool_call"]
    assert replayed == log.events


def test_events_are_json_serializable_with_weird_values():
    log = AuditLog()
    evt = log.record({"event": "tool_call", "args": {"when": object()}})
    json.dumps(evt)  # must not raise


def test_record_deep_copies_caller_dict():
    log = AuditLog()
    args = {"order_id": "ORD-1001"}
    log.record({"event": "tool_call", "args": args})
    args["order_id"] = "MUTATED"
    assert log.events[0]["args"]["order_id"] == "ORD-1001"
