"""Procedural memory, versioned in git (Phase 2d).

The source of truth for the deterministic rules layer is two files under
rules/:
- policy.yaml      -- numeric policy thresholds (loaded + validated here)
- procedural.md    -- the procedural prose for the system prompt, with
                      {refund_threshold} / {return_window_days} slots

Rule changes go through git -- the commit history IS the audit trail.
This module is stdlib-only and imports nothing from the package, so
config.py can load it at import time without import cycles.
"""

from __future__ import annotations

from pathlib import Path

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"
POLICY_FILE = RULES_DIR / "policy.yaml"
PROCEDURAL_FILE = RULES_DIR / "procedural.md"

# key -> acceptable python types after parsing
_REQUIRED: dict[str, tuple[type, ...]] = {
    "refund_approval_threshold": (int, float),
    "return_window_days": int,
    "goodwill_credit_max": (int, float),
    "max_episodic_hits": int,
    "max_semantic_facts": int,
    "session_abandon_days": int,
    "episodic_summary_days": int,
    "fact_min_confidence": (int, float),
    "fact_reconfirm_days": int,
    "superseded_retention_days": int,
    "llm_cost_per_mtok_input_usd": (int, float),
    "llm_cost_per_mtok_output_usd": (int, float),
}

# Rule names that may carry a `deprecated_<name>: true|false` flag.
KNOWN_RULES = (
    "refund_approval",
    "return_eligibility",
    "escalation",
    "late_delivery_credit",
)


def _parse_value(raw: str):
    """Parse a scalar YAML value (bool, int, float, or plain string)."""
    raw = raw.strip().strip("\"'")
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def load_policy(path: str | Path = POLICY_FILE) -> dict:
    """Load and validate rules/policy.yaml. Raises ValueError on any problem.

    Fail-fast by design: a bad policy must break startup loudly, never run
    the agent under silently wrong thresholds.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"policy file not found: {path}")
    policy: dict = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"{path}:{lineno}: expected 'key: value', got {line!r}")
        key, _, raw = stripped.partition(":")
        key, raw = key.strip(), raw.strip()
        if not key or not raw:
            raise ValueError(f"{path}:{lineno}: empty key or value in {line!r}")
        if key in policy:
            raise ValueError(f"{path}:{lineno}: duplicate key {key!r}")
        policy[key] = _parse_value(raw)

    for key, types in _REQUIRED.items():
        if key not in policy:
            raise ValueError(f"{path}: missing required policy key {key!r}")
        if not isinstance(policy[key], types) or isinstance(policy[key], bool):
            raise ValueError(
                f"{path}: {key!r} must be numeric, got {policy[key]!r}"
            )
    if policy["refund_approval_threshold"] <= 0:
        raise ValueError("refund_approval_threshold must be positive")
    if policy["goodwill_credit_max"] <= 0:
        raise ValueError("goodwill_credit_max must be positive")
    if policy["goodwill_credit_max"] > policy["refund_approval_threshold"]:
        raise ValueError(
            "goodwill_credit_max must sit under refund_approval_threshold"
        )
    if policy["return_window_days"] < 1:
        raise ValueError("return_window_days must be >= 1")
    if policy["max_episodic_hits"] < 1 or policy["max_semantic_facts"] < 1:
        raise ValueError("recall budgets must be >= 1")
    for key in ("session_abandon_days", "episodic_summary_days",
                "fact_reconfirm_days", "superseded_retention_days"):
        if policy[key] < 1:
            raise ValueError(f"{key} must be >= 1")
    if not 0.0 <= policy["fact_min_confidence"] <= 1.0:
        raise ValueError("fact_min_confidence must be between 0 and 1")
    for key in ("llm_cost_per_mtok_input_usd", "llm_cost_per_mtok_output_usd"):
        if policy[key] < 0:
            raise ValueError(f"{key} must be >= 0")
    # Deprecation flags must name a known rule and be boolean.
    for key, value in policy.items():
        if not key.startswith("deprecated_"):
            continue
        rule = key[len("deprecated_"):]
        if rule not in KNOWN_RULES:
            raise ValueError(
                f"{path}: {key!r} does not name a known rule "
                f"(known: {', '.join(KNOWN_RULES)})"
            )
        if not isinstance(value, bool):
            raise ValueError(f"{path}: {key!r} must be true or false")
    return policy


def deprecated_rules(policy: dict) -> frozenset:
    """Rule names flagged `deprecated_<name>: true` in the policy."""
    return frozenset(
        name for name in KNOWN_RULES if policy.get(f"deprecated_{name}") is True
    )


def _drop_deprecated_blocks(template: str, policy: dict) -> str:
    """Remove `<!-- rule:name --> ... <!-- /rule -->` blocks for deprecated
    rules, so deprecated rules stop firing at the prompt level too. They
    stay readable in git history."""
    import re

    deprecated = deprecated_rules(policy)

    def _sub(match: re.Match) -> str:
        return "" if match.group(1) in deprecated else match.group(0)

    return re.sub(
        r"<!-- rule:([a-z_]+) -->(.*?)<!-- /rule -->",
        _sub,
        template,
        flags=re.DOTALL,
    )


def render_system_prompt(policy: dict) -> str:
    """Render rules/procedural.md with the policy thresholds.

    The markdown is the prose source of truth; the numbers always come
    from the validated policy, so prompt and deterministic rules can never
    drift apart. Deprecated rules' prose blocks are dropped.
    """
    template = _drop_deprecated_blocks(PROCEDURAL_FILE.read_text(), policy)
    return template.format(
        refund_threshold=f"{float(policy['refund_approval_threshold']):.2f}",
        return_window_days=int(policy["return_window_days"]),
    )
