"""AuditLog: durable JSONL event log for every agent action.

The established pattern (from the Jev repos), adapted for the support
agent. Two sinks:

- in-memory ``events`` list (inspect in-process; handy in tests)
- optional JSONL file (``register_audit(app, user, path="audit.jsonl")``)
  for a persistent, line-per-event trail you can grep, replay, or load
  into a dataframe. This is the durable, on-disk episodic record.

Every event is a JSON-serializable dict with a UTC timestamp, a
``component`` ("support_agent") and an ``event`` type:

- ``turn``           user prompt arrived (preview + hash, never the full text)
- ``tool_call``      a tool was invoked (name + scrubbed args)
- ``tool_result``    a tool returned (status ok/error + scrubbed preview)
- ``memory_write``   episodic or semantic write (kind + detail)
- ``rule_fired``     a procedural rule produced a verdict
- ``memory_recall``  long-term memory injected at session start

Privacy: prompts and args are stored as truncated previews plus a
SHA-256 hash. Card-number-shaped strings are redacted to ``[REDACTED]``
*before* anything is stored -- they must never reach any memory layer,
and that includes this log.

Stdlib only. Logging helpers (``log_turn`` etc.) are no-ops when no
``AuditLog`` is registered for the user, so the agent runs with zero
overhead by default -- same convention as the Jev ``audit=None`` default.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import datetime, timezone

PROMPT_PREVIEW_CHARS = 200
REDACTED = "[REDACTED]"

# Card-number-shaped strings: 16 digits, optionally grouped with spaces
# or dashes. Single source of truth -- memory_banks.py imports this for
# its refusal check and episode scrubbing.
CARD_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def scrub_pii(text: str) -> str:
    """Redact card-number-shaped strings. Safe on non-strings."""
    if not isinstance(text, str):
        return text
    return CARD_RE.sub(REDACTED, text)


def _scrub_value(value, _depth: int = 0):
    """Recursively scrub + truncate a JSON-ish value for safe logging."""
    if isinstance(value, str):
        v = scrub_pii(value)
        return v[:PROMPT_PREVIEW_CHARS] + ("..." if len(v) > PROMPT_PREVIEW_CHARS else "")
    if isinstance(value, dict):
        return {k: _scrub_value(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_value(v, _depth + 1) for v in value[:20]]
    return value


def prompt_preview(prompt: str, chars: int = PROMPT_PREVIEW_CHARS) -> str:
    """Scrubbed, truncated preview -- enough to audit, not a prompt store."""
    return scrub_pii(prompt or "")[:chars]


def prompt_hash(prompt: str) -> str:
    """16 hex chars of SHA-256 over the *scrubbed* prompt.

    Correlates repeat prompts without storing anything sensitive; hashing
    the scrubbed text means a card number can never be brute-forced back
    from the hash.
    """
    return hashlib.sha256(scrub_pii(prompt or "").encode("utf-8")).hexdigest()[:16]


class AuditLog:
    """Collects agent events in memory and/or a JSONL file."""

    def __init__(self, path: str | None = None) -> None:
        self.events: list[dict] = []
        self.path = path

    def record(self, event: dict) -> dict:
        """Stamp an event with ``ts``, store it, and append it to the JSONL
        file if one was configured. Returns the stamped event."""
        evt = {"ts": utc_now_iso(), "component": "support_agent", **event}
        # Deep-copy so later mutation of the caller's dict can't rewrite
        # history; default=str keeps everything JSON-serializable.
        evt = json.loads(json.dumps(evt, default=str))
        self.events.append(evt)
        if self.path:
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(evt) + "\n")
        return evt

    def summary(self) -> dict:
        """Counts per event type, e.g. ``{'total': 9, 'by_event':
        {'turn': 2, 'tool_call': 3, ...}}``."""
        by_event: dict[str, int] = {}
        for e in self.events:
            t = e.get("event", "unknown")
            by_event[t] = by_event.get(t, 0) + 1
        return {"total": len(self.events), "by_event": by_event}

    def replay(self) -> list[dict]:
        """Re-read the JSONL file (if any) and return its events in order.

        Proves the on-disk trail replays the story the in-memory log saw.
        """
        if not self.path or not os.path.exists(self.path):
            return list(self.events)
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def __len__(self) -> int:
        return len(self.events)


# --- Registry ---------------------------------------------------------------
# One AuditLog per (app_name, user_id), mirroring the memory-banks
# registry. Callbacks resolve the log from the session, so no global
# user state leaks across users.

_REGISTRY: dict[tuple[str, str], AuditLog] = {}


def register_audit(
    app_name: str, user_id: str, path: str | None = None
) -> AuditLog:
    """Create (or return the existing) audit log for a user. Idempotent."""
    key = (app_name, user_id)
    log = _REGISTRY.get(key)
    if log is None:
        log = AuditLog(path=path)
        _REGISTRY[key] = log
    return log


def get_audit(app_name: str, user_id: str) -> AuditLog | None:
    return _REGISTRY.get((app_name, user_id))


def clear_audit_registry() -> None:
    """Tests only: drop all registered audit logs."""
    _REGISTRY.clear()


# --- Logging helpers --------------------------------------------------------
# All no-op when no log is registered for the user. Never raise: a
# logging failure must not break the agent loop.


def _emit(app_name: str, user_id: str, event: dict) -> dict | None:
    try:
        log = get_audit(app_name, user_id)
        if log is None:
            return None
        return log.record(event)
    except Exception:
        return None


def log_turn(app_name: str, user_id: str, session_id: str | None, prompt: str) -> None:
    _emit(
        app_name,
        user_id,
        {
            "event": "turn",
            "session_id": session_id,
            "prompt_preview": prompt_preview(prompt),
            "prompt_hash": prompt_hash(prompt),
        },
    )


def log_tool_call(
    app_name: str, user_id: str, session_id: str | None, tool_name: str, args: dict
) -> None:
    _emit(
        app_name,
        user_id,
        {
            "event": "tool_call",
            "session_id": session_id,
            "tool": tool_name,
            "args": _scrub_value(copy.deepcopy(args or {})),
        },
    )


def log_tool_result(
    app_name: str,
    user_id: str,
    session_id: str | None,
    tool_name: str,
    result: dict,
) -> None:
    status = "error" if isinstance(result, dict) and "error" in result else "ok"
    _emit(
        app_name,
        user_id,
        {
            "event": "tool_result",
            "session_id": session_id,
            "tool": tool_name,
            "status": status,
            "result_preview": _scrub_value(result),
        },
    )


def log_memory_write(
    app_name: str, user_id: str, kind: str, detail: dict
) -> None:
    _emit(
        app_name,
        user_id,
        {"event": "memory_write", "kind": kind, "detail": _scrub_value(detail)},
    )


def log_rule_fired(
    app_name: str,
    user_id: str,
    session_id: str | None,
    rule: str,
    decision: str,
    reason: str,
) -> None:
    _emit(
        app_name,
        user_id,
        {
            "event": "rule_fired",
            "session_id": session_id,
            "rule": rule,
            "decision": decision,
            "reason_preview": prompt_preview(reason),
        },
    )


def log_memory_recall(
    app_name: str,
    user_id: str,
    session_id: str | None,
    injected: bool,
    block_chars: int,
) -> None:
    _emit(
        app_name,
        user_id,
        {
            "event": "memory_recall",
            "session_id": session_id,
            "injected": injected,
            "block_chars": block_chars,
        },
    )
