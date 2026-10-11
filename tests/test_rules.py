"""Rules: pure, deterministic, boundary-tested."""

from support_agent import fixtures, rules


def test_refund_threshold_boundary():
    assert rules.evaluate_refund_request(50.0)["decision"] == "approve"
    assert rules.evaluate_refund_request(50.01)["decision"] == "draft_for_approval"
    assert rules.evaluate_refund_request(500)["decision"] == "draft_for_approval"


def test_refund_rule_never_says_execute_above_threshold():
    verdict = rules.evaluate_refund_request(75.0)
    assert "Do NOT execute" in verdict["reason"]


def test_refund_rule_rejects_garbage():
    assert "error" in rules.evaluate_refund_request(-1)
    assert "error" in rules.evaluate_refund_request(0)
    assert "error" in rules.evaluate_refund_request("lots")


def test_return_eligibility_happy_path():
    v = rules.check_return_eligibility(fixtures.get_order("ORD-1002"))
    assert v["eligible"] is True


def test_return_eligibility_outside_window():
    v = rules.check_return_eligibility(fixtures.get_order("ORD-1004"))
    assert v["eligible"] is False
    assert "30-day" in v["reason"]


def test_return_eligibility_not_delivered():
    v = rules.check_return_eligibility(fixtures.get_order("ORD-1003"))
    assert v["eligible"] is False


def test_return_eligibility_rejects_garbage():
    assert "error" in rules.check_return_eligibility({})
    assert "error" in rules.check_return_eligibility(None)


def test_escalation_triggers():
    assert rules.should_escalate({"legal_threat": True})["escalate"] is True
    assert rules.should_escalate({"repeat_contacts": 3})["escalate"] is True
    assert rules.should_escalate({"repeat_contacts": 2})["escalate"] is False
    assert rules.should_escalate({"vip_high_value": True})["escalate"] is True
    assert rules.should_escalate({"agent_unsure": True})["escalate"] is True
    assert rules.should_escalate({})["escalate"] is False
    assert "error" in rules.should_escalate("nope")


def test_late_delivery_credit():
    v = rules.late_delivery_credit(fixtures.get_order("ORD-1001"))
    assert v["credit"] == 25.0  # 5 days late x $5, capped at goodwill max
    v = rules.late_delivery_credit(fixtures.get_order("ORD-1002"))
    assert v["credit"] == 0.0
    assert "error" in rules.late_delivery_credit({})
