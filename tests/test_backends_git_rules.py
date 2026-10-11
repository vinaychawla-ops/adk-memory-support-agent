"""Unit tests for the git-versioned procedural memory (2d).

policy.yaml is the validated source of truth for thresholds;
procedural.md is the prose source of truth for the system prompt.
"""

import pytest

from support_agent import agent as agent_module
from support_agent import config
from support_agent.backends import git_rules


def test_policy_loads_with_expected_values():
    policy = git_rules.load_policy()
    assert policy["refund_approval_threshold"] == 50.0
    assert policy["return_window_days"] == 30
    assert policy["goodwill_credit_max"] == 25.0
    assert policy["max_episodic_hits"] == 5
    assert policy["max_semantic_facts"] == 10


def test_config_constants_come_from_policy():
    assert config.REFUND_APPROVAL_THRESHOLD == 50.0
    assert config.RETURN_WINDOW_DAYS == 30
    assert config.GOODWILL_CREDIT_MAX == 25.0
    assert config.MAX_EPISODIC_HITS == 5
    assert config.MAX_SEMANTIC_FACTS == 10


def test_policy_validation_rejects_bad_files(tmp_path):
    good = git_rules.POLICY_FILE.read_text()

    def bad(content: str):
        p = tmp_path / "policy.yaml"
        p.write_text(content)
        with pytest.raises(ValueError):
            git_rules.load_policy(p)

    bad("")  # missing keys
    bad("refund_approval_threshold: 50.0\n")  # missing the rest
    bad(good.replace("return_window_days: 30", "return_window_days: soon"))
    bad(good.replace("refund_approval_threshold: 50.0", "refund_approval_threshold: -5"))
    bad(good.replace("return_window_days: 30", "return_window_days: 0"))
    bad(good + "refund_approval_threshold: 60.0\n")  # duplicate key
    bad(good.replace(
        "goodwill_credit_max: 25.0", "goodwill_credit_max: 99.0"
    ))  # above the refund threshold

    with pytest.raises(ValueError):
        git_rules.load_policy(tmp_path / "does-not-exist.yaml")


def test_rendered_prompt_matches_thresholds():
    prompt = git_rules.render_system_prompt(git_rules.load_policy())
    assert "$50.00" in prompt
    assert "30 days" in prompt
    assert "{" not in prompt and "}" not in prompt  # no unfilled slots
    assert "NEVER claim a large refund was executed" in prompt
    assert "MEMORY DISCIPLINE" in prompt


def test_agent_system_prompt_is_the_rendered_markdown():
    # The wiring test already asserts the values land in the prompt; here
    # we pin that the prompt IS the rendered file, byte for byte.
    assert agent_module.SYSTEM_PROMPT_BASE == git_rules.render_system_prompt(
        git_rules.load_policy()
    )


def test_rules_module_uses_policy_thresholds():
    from support_agent import rules

    # Behavior identical to the old hardcoded constants, now via policy.
    assert rules.evaluate_refund_request(50.0)["decision"] == "approve"
    assert rules.evaluate_refund_request(50.01)["decision"] == "draft_for_approval"
