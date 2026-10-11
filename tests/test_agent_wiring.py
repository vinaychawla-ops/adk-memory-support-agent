"""Agent wiring: construction, instruction rendering, recall injection.

Everything here runs offline. The LLM itself is never invoked; we test
the seams around it: the agent builds, the Runner builds, the instruction
provider renders the memory block, and ensure_memory_recall populates
session state exactly once.
"""

import asyncio

import pytest

from support_agent import agent as agent_module
from support_agent import config


def test_build_agent_constructs_without_key():
    agent = agent_module.build_agent()
    assert agent.name == "support_agent"
    assert agent.model == config.MODEL
    assert len(agent.tools) == 8
    assert agent.before_agent_callback is not None


def test_make_runner_constructs_without_key(banks):
    runner, session_service, memory_service, agent = agent_module.make_runner(
        "test_user"
    )
    assert runner is not None
    assert session_service is not None
    assert memory_service is not None
    assert agent.name == "support_agent"


def test_make_runner_registers_banks():
    from support_agent.memory_banks import clear_registry, get_banks

    clear_registry()
    runner, _, memory_service, _ = agent_module.make_runner("maya")
    banks = get_banks(config.APP_NAME, "maya")
    assert banks is not None
    assert banks.episodic._service is memory_service


def test_ensure_memory_recall_populates_state_once(banks):
    banks.semantic.upsert("contact_preference", "prefers email updates")

    async def go():
        state: dict = {}
        await agent_module.ensure_memory_recall(
            state, "test_app", "test_user", "preferences"
        )
        first = state["memory_context"]
        await agent_module.ensure_memory_recall(
            state, "test_app", "test_user", "different query"
        )
        return first, state["memory_context"], state["memory_recall_done"]

    first, second, done = asyncio.run(go())
    assert "prefers email updates" in first
    assert first == second  # second call is a no-op
    assert done is True


def test_ensure_memory_recall_without_banks_is_empty():
    from support_agent.memory_banks import clear_registry

    clear_registry()

    async def go():
        state: dict = {}
        await agent_module.ensure_memory_recall(state, "nope", "nobody", "q")
        return state

    state = asyncio.run(go())
    assert state["memory_context"] == ""
    assert state["memory_recall_done"] is True


class _FakeCtx:
    """Stand-in for ReadonlyContext: only .session.state is read."""

    def __init__(self, state):
        self.session = type("S", (), {"state": state})()


def test_instruction_provider_renders_memory_block():
    async def go():
        with_memory = await agent_module._instruction_provider(
            _FakeCtx({"memory_context": "MEMORY -- prefers email"})
        )
        without_memory = await agent_module._instruction_provider(_FakeCtx({}))
        return with_memory, without_memory

    with_memory, without_memory = asyncio.run(go())
    assert "MEMORY -- prefers email" in with_memory
    assert "MEMORY -- prefers email" not in without_memory
    assert "$50.00" in without_memory  # procedural rules always present
    assert "request_refund" in without_memory


def test_system_prompt_states_thresholds():
    assert f"${config.REFUND_APPROVAL_THRESHOLD:.2f}" in agent_module.SYSTEM_PROMPT_BASE
    assert str(config.RETURN_WINDOW_DAYS) in agent_module.SYSTEM_PROMPT_BASE
