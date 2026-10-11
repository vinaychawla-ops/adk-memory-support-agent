"""Customer support agent with layered memory (Google ADK).

Phase 0 of the roadmap in ../ROADMAP.md: a single LlmAgent on Gemini that
embodies the four memory layers from the "AI Agent Memory Architecture"
video (Applied with AI):

- session state + conversation history -> ADK InMemorySessionService
- episodic + semantic memory            -> ADK InMemoryMemoryService (+ a
                                          dedicated semantic fact store)
- procedural memory                     -> system instruction + rules.py
- tools                                 -> mock Order Management System
"""

from support_agent.agent import build_agent, init_user_memory, make_runner
from support_agent.memory_banks import MemoryBanks, get_banks, register_banks

__all__ = [
    "build_agent",
    "init_user_memory",
    "make_runner",
    "MemoryBanks",
    "get_banks",
    "register_banks",
]
