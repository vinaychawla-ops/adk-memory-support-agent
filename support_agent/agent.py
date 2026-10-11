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
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

from support_agent import config, tools as tool_module
from support_agent.audit import (
    log_memory_recall,
    log_tool_call,
    log_tool_result,
    log_turn,
)
from support_agent.memory_banks import get_banks, register_banks

SYSTEM_PROMPT_BASE = """\
You are a customer support agent for an online store. Be warm, concise, \
and precise. Never invent order details -- always use the tools.

MEMORY DISCIPLINE (this is how you remember):
- A MEMORY block may appear below with durable customer facts and relevant \
past interactions. Trust it; do not ask the customer to repeat what it states.
- When the customer states a durable preference (contact channel, shipping \
preference, etc.), store it with remember_fact using a short slot name like \
"contact_preference".
- The active order and any in-progress return are tracked in session state \
by the tools; rely on that, not on re-reading the chat.

PROCEDURAL RULES (always apply, no exceptions):
- Refunds or credits of $%.2f or less may be issued through request_refund.
- Anything above $%.2f: request_refund will DRAFT the request and escalate \
it. Present the draft to the customer and confirm a human supervisor will \
review it. NEVER claim a large refund was executed.
- Returns are only eligible within %d days of delivery; otherwise escalate.
- Escalate to a human when: the customer threatens legal/regulatory action, \
this is their 3rd+ contact about the same issue, or you cannot resolve it \
with your tools. Summarize the situation when you escalate.
- For late deliveries, offer the goodwill credit the rules compute; keep \
the tone apologetic but brief.
""" % (
    config.REFUND_APPROVAL_THRESHOLD,
    config.REFUND_APPROVAL_THRESHOLD,
    config.RETURN_WINDOW_DAYS,
)


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

    The recall query is the user's latest message *plus* stable domain
    terms ("customer preferences past orders"). Keyword search (Phase 0)
    only matches on word overlap, so without the domain terms a message
    like "I want a refund" would never surface the customer's contact
    preference. The extra terms bias recall toward what a support agent
    should always surface; Phase 2 replaces this with vector search.
    """
    app_name, user_id, session_id = _session_ids(ctx)
    user_text = _latest_user_text(ctx)
    log_turn(app_name, user_id, session_id, user_text)
    query = f"{user_text} customer preferences past orders".strip()
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


def init_user_memory(app_name: str, user_id: str, memory_service) -> None:
    """Register the episodic+semantic banks for a user (idempotent).

    Must be called once per user before the agent serves them, so tools
    and the recall callback can find the banks.
    """
    register_banks(app_name, user_id, memory_service)


def make_runner(user_id: str | None = None):
    """Build (runner, session_service, memory_service, agent).

    If user_id is given, its memory banks are registered immediately.
    """
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()
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
