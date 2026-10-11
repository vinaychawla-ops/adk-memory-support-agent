"""Shared fakes for the offline suite."""

import pytest

from support_agent import fixtures
from support_agent.memory_banks import clear_registry, register_banks


class FakeSession:
    def __init__(self, app_name="test_app", user_id="test_user"):
        self.app_name = app_name
        self.user_id = user_id


class FakeToolContext:
    """Minimal stand-in for ADK ToolContext: .state dict + .session."""

    def __init__(self, app_name="test_app", user_id="test_user"):
        self.state: dict = {}
        self.session = FakeSession(app_name, user_id)


@pytest.fixture()
def ctx():
    clear_registry()
    fixtures.reset_counters()
    return FakeToolContext()


@pytest.fixture()
def banks():
    """Registered MemoryBanks with a real ADK InMemoryMemoryService."""
    from google.adk.memory import InMemoryMemoryService

    clear_registry()
    return register_banks("test_app", "test_user", InMemoryMemoryService())
