"""The support agent: one LlmAgent with all four memory layers wired.

Where each layer lives:
- Session state + conversation history: ADK InMemorySessionService. Tools
  keep structured state (active_order_id, return_in_progress); history is
  the session's events.
- Episodic + semantic memory: ADK InMemoryMemoryService plus SemanticBank.
  before_agent_callback injects a recall block into session state once per
  session; the instruction provider renders it into the system prompt.
  After a session, the caller stores it via banks.episodic.record_session().
- Procedural memory: the system instruction below (tone, thresholds,
  escalation) + the deterministic rules in rules.py, enforced by tools.

The agent reads the model/key from the environment (GOOGLE_API_KEY) at
run time; constructing the agent and Runner needs no key.
"""

from __future__ import annotations

from typing import Any, MutableMapping

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.runners import Runner

from support_agent import config, tools as tool_module
from support_agent.audit import (
    log_memory_recall,
    log_tool_call,
    log_tool_result,
    log_turn,
)
from support_agent.backends import (
    make_memory_service,
    make_semantic_store,
    make_session_service,
)
from support_agent.backends import git_rules
from support_agent.memory_banks import get_banks, register_banks

# Procedural memory, versioned in git: rules/procedural.md is the prose
# source of truth, rendered with the validated thresholds from
# rules/policy.yaml so prompt and deterministic rules cannot drift apart.
SYSTEM_PROMPT_BASE = git_rules.render_system_prompt(config.POLICY)


async def _instruction_provider(ctx: ReadonlyContext) -> str:
    """System prompt = base instruction + this session's memory block."""
    memory_block = ""
    try:
        state = ctx.session.state if ctx.session else {}
        memory_block = state.get("memory_context") or ""
    except Exception:
        memory_block = ""
    if memory_block:
        return f"{SYSTEM_PROMPT_BASE}\n{memory_block}\n"
    return SYSTEM_PROMPT_BASE


def _latest_user_text(ctx: CallbackContext) -> str:
    """Best-effort: the most recent user message in this session."""
    try:
        for event in reversed(ctx.session.events or []):
            if event.author == "user" and event.content and event.content.parts:
                text = " ".join(p.text for p in event.content.parts if p.text).strip()
                if text:
                    return text
    except Exception:
        pass
    return ""


async def ensure_memory_recall(
    state: MutableMapping[str, Any],
    app_name: str,
    user_id: str,
    query: str,
    session_id: str | None = None,
) -> None:
    """Populate state["memory_context"] once per session (idempotent).

    Kept separate from the ADK callback so it is unit-testable with a
    plain dict. The callback below is the thin ADK wrapper.
    """
    if state.get("memory_recall_done"):
        return
    banks = get_banks(app_name, user_id)
    block = await banks.build_recall_block(query) if banks else ""
    state["memory_context"] = block
    state["memory_recall_done"] = True
    log_memory_recall(app_name, user_id, session_id, bool(block), len(block))


def _session_ids(ctx) -> tuple[str, str, str | None]:
    """(app_name, user_id, session_id) from a callback context, defensively."""
    try:
        session = ctx.session
        return (
            session.app_name,
            session.user_id,
            getattr(session, "id", None),
        )
    except Exception:
        return "", "", None


async def _recall_before_agent(ctx: CallbackContext) -> None:
    """before_agent_callback: log the turn, then inject long-term memory.

    The recall query is the user's latest message. On the in-memory
    backend (keyword search) stable domain terms ("customer preferences
    past orders") are appended, because keyword search only matches on word
    overlap -- without them a message like "I want a refund" would never
    surface the customer's contact preference. With EPISODIC_BACKEND=
    pgvector the query goes to vector search, which needs no such crutch,
    so the raw user text is used.
    """
    app_name, user_id, session_id = _session_ids(ctx)
    user_text = _latest_user_text(ctx)
    log_turn(app_name, user_id, session_id, user_text)
    if config.EPISODIC_BACKEND == "memory":
        query = f"{user_text} customer preferences past orders".strip()
    else:
        query = user_text
    await ensure_memory_recall(
        ctx.state, app_name, user_id, query, session_id=session_id
    )
    return None


async def _audit_before_tool(tool, args: dict, tool_context) -> None:
    """before_tool_callback: log the invocation. Always returns None so the
    tool runs -- this callback never blocks, it only observes."""
    app_name, user_id, session_id = _session_ids(tool_context)
    try:
        name = getattr(tool, "name", repr(tool))
    except Exception:
        name = "unknown"
    log_tool_call(app_name, user_id, session_id, name, args or {})
    return None


async def _audit_after_tool(tool, args: dict, tool_context, tool_response: dict) -> None:
    """after_tool_callback: log the outcome (ok/error + preview)."""
    app_name, user_id, session_id = _session_ids(tool_context)
    try:
        name = getattr(tool, "name", repr(tool))
    except Exception:
        name = "unknown"
    log_tool_result(app_name, user_id, session_id, name, tool_response or {})
    return None


def build_agent() -> LlmAgent:
    """Construct the agent. Needs no API key -- only running it does.

    Tool callbacks are attached unconditionally but only observe: with no
    AuditLog registered for the user they are no-ops (zero overhead).
    """
    return LlmAgent(
        name="support_agent",
        model=config.MODEL,
        description="E-commerce customer support with layered memory.",
        instruction=_instruction_provider,
        tools=tool_module.build_tools(),
        before_agent_callback=_recall_before_agent,
        before_tool_callback=_audit_before_tool,
        after_tool_callback=_audit_after_tool,
    )


def init_user_memory(app_name: str, user_id: str, memory_service, semantic_store=None) -> None:
    """Register the episodic+semantic banks for a user (idempotent).

    Must be called once per user before the agent serves them, so tools
    and the recall callback can find the banks. The semantic store
    defaults to make_semantic_store() (env-selected backend).
    """
    if semantic_store is None:
        semantic_store = make_semantic_store(app_name, user_id)
    register_banks(app_name, user_id, memory_service, semantic=semantic_store)


def make_runner(user_id: str | None = None):
    """Build (runner, session_service, memory_service, agent).

    Services come from the backend factories (env-selected; in-memory by
    default, so zero infra still works). If user_id is given, its memory
    banks are registered immediately.
    """
    session_service = make_session_service()
    memory_service = make_memory_service()
    agent = build_agent()
    if user_id is not None:
        init_user_memory(config.APP_NAME, user_id, memory_service)
    runner = Runner(
        app_name=config.APP_NAME,
        agent=agent,
        session_service=session_service,
        memory_service=memory_service,
    )
    return runner, session_service, memory_service, agent
