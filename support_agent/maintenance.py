"""Phase 4: the forgetting layer.

The video's closing line is the spec: knowing what to remember, where to
remember it, *and when to forget*. This module gives every memory layer a
lifecycle, all driven by rules/policy.yaml (tunable, git-versioned):

- **Sessions** -- compacted when idle longer than SESSION_ABANDON_DAYS.
  Their episodic record was already captured per-turn, so nothing is
  truly lost.
- **Episodic** -- episodes older than EPISODIC_SUMMARY_DAYS are summarized
  into one rolling semantic fact, then the raw episodes are dropped.
  Summarization needs the LLM (env-driven, GOOGLE_API_KEY); a deterministic
  fake covers offline tests.
- **Semantic** -- staleness review: expired facts are purged, long-dead
  superseded facts are purged, low-confidence / stale / conflicted facts
  are *flagged* as drafts for human review (verdicts, not execution).
- **Procedural** -- deprecated rules (deprecated_<name>: true in policy)
  stop firing; maintenance verifies they don't.
- **Metrics** -- hit rate per layer, retrieval precision on scripted
  probes, context tokens per turn per layer, estimated cost per ticket --
  computed from the audit log, surfaced by the `report` command.

Every action runs through the audit log (kind "maintenance"). The
forgetting drill (tests/test_maintenance.py) is the acceptance test:
aged-out data is provably gone from every store, and the eval set still
passes.

CLI:
  python -m support_agent.maintenance report --audit PATH [--precision]
  python -m support_agent.maintenance run --user USER [--app APP]
      [--layer all|session|episodic|semantic|procedural] [--execute]

`run` is dry-run by default; pass --execute to perform. Stdlib + the
project's own modules only (summarizers import their SDKs lazily).
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from support_agent import config
from support_agent.audit import CARD_RE, log_memory_write, scrub_pii
from support_agent.backends import git_rules
from support_agent.memory_banks import _days_between, _purgeable_expired, _today_iso

SUMMARY_SLOT = "episodic_summary"
SUMMARY_CONFIDENCE = 0.8
CHARS_PER_TOKEN = 4.0  # rough estimate, always labeled as such


# --- Sessions ---------------------------------------------------------------


async def _list_sessions(session_service, app_name: str, user_id: str):
    resp = await session_service.list_sessions(app_name=app_name, user_id=user_id)
    return list(resp.sessions or [])


def find_abandoned(sessions, abandon_days: int, now: float | None = None):
    """Split sessions into (abandoned, unknown_age).

    A session is abandoned when its last_update_time is older than
    abandon_days. Sessions with no timestamp are reported as unknown
    rather than deleted -- never purge what you cannot age.
    """
    now = time.time() if now is None else now
    cutoff = now - abandon_days * 86400
    abandoned, unknown = [], []
    for s in sessions:
        lut = getattr(s, "last_update_time", None)
        if lut is None:
            unknown.append(s.id)
        elif lut < cutoff:
            abandoned.append(s)
    return abandoned, unknown


async def compact_sessions(
    session_service,
    app_name: str,
    user_id: str,
    abandon_days: int | None = None,
    dry_run: bool = True,
) -> dict:
    """Delete sessions idle longer than abandon_days. Dry-run by default."""
    abandon_days = (
        config.SESSION_ABANDON_DAYS if abandon_days is None else abandon_days
    )
    sessions = await _list_sessions(session_service, app_name, user_id)
    abandoned, unknown = find_abandoned(sessions, abandon_days)
    report = {
        "layer": "session",
        "dry_run": dry_run,
        "abandon_days": abandon_days,
        "total_sessions": len(sessions),
        "abandoned": [s.id for s in abandoned],
        "unknown_age": unknown,
        "deleted": 0,
    }
    if dry_run:
        return report
    for s in abandoned:
        await session_service.delete_session(
            app_name=app_name, user_id=user_id, session_id=s.id
        )
    report["deleted"] = len(abandoned)
    log_memory_write(
        app_name,
        user_id,
        "maintenance",
        {
            "action": "compact_sessions",
            "deleted": len(abandoned),
            "abandon_days": abandon_days,
        },
    )
    return report


# --- Summarizers ------------------------------------------------------------


class Summarizer(abc.ABC):
    """Turn old episode texts into one durable factual summary."""

    @abc.abstractmethod
    def summarize(
        self, texts: list[str], prior_summary: str | None = None
    ) -> str: ...


_STOPWORDS = frozenset(
    "the a an and or of to in on for with is was are were be been it its "
    "this that these those they them he she we you i my me our your his her "
    "at as by from have has had do does did will would can could should there "
    "their what when where who how not no yes if then than so but because about "
    "into over after before".split()
)


class FakeSummarizer(Summarizer):
    """Deterministic offline summarizer (no key, no network).

    Produces "N past interactions. Recurring topics: w1, w2, w3." from the
    most common significant tokens. Tests plumbing and PII-safety, not
    summary quality.
    """

    def summarize(
        self, texts: list[str], prior_summary: str | None = None
    ) -> str:
        tokens = []
        for text in texts:
            tokens.extend(
                w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text or "")
            )
        top = [
            w for w, _ in Counter(
                w for w in tokens if w not in _STOPWORDS
            ).most_common(5)
        ]
        out = f"{len(texts)} past customer support interactions"
        if top:
            out += f". Recurring topics: {', '.join(top)}"
        if prior_summary:
            out += f" Earlier summary: {prior_summary}"
        return out + "."


class GeminiSummarizer(Summarizer):
    """Real summarization via google.genai. The client reads GOOGLE_API_KEY
    from the environment when summarize() is first called -- never at import."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or config.MODEL

    def summarize(
        self, texts: list[str], prior_summary: str | None = None
    ) -> str:
        from google import genai

        client = genai.Client()
        prompt = (
            "Summarize these customer support interaction transcripts into "
            "2-3 durable factual sentences about the customer's history "
            "(recurring issues, resolutions, preferences). Plain facts only: "
            "no PII, no advice, no greeting, no markdown."
        )
        if prior_summary:
            prompt += (
                "\nAn earlier summary exists; fold it in if still relevant:\n"
                + prior_summary
            )
        prompt += "\n\nTranscripts:\n" + "\n---\n".join(texts)
        response = client.models.generate_content(
            model=self._model, contents=prompt
        )
        return (response.text or "").strip()


def make_summarizer() -> Summarizer:
    """Summarizer per SUMMARIZER=fake|gemini (default fake: offline-safe)."""
    if os.environ.get("SUMMARIZER", "fake").lower() == "gemini":
        return GeminiSummarizer()
    return FakeSummarizer()


# --- Episodic summarization -------------------------------------------------


def episodic_lifecycle_supported(memory_service) -> tuple[bool, str]:
    """Whether the episodic backend supports list/delete for maintenance.

    ADK's BaseMemoryService has no lifecycle API; only the pgvector store
    (real or fake) exposes it. The in-memory backend reports unsupported
    rather than failing -- its forgetting story is "restart the process".
    """
    store = getattr(memory_service, "episode_store", None)
    if store is not None and hasattr(store, "list_episodes_older_than"):
        return True, type(store).__name__
    return (
        False,
        f"{type(memory_service).__name__} has no episode lifecycle API; "
        "use EPISODIC_BACKEND=pgvector for the forgetting layer.",
    )


async def summarize_episodes(
    episode_store,
    semantic_store,
    summarizer: Summarizer,
    app_name: str,
    user_id: str,
    summary_days: int | None = None,
    dry_run: bool = True,
) -> dict:
    """Summarize episodes older than summary_days, then drop the raw rows.

    The summary is PII-scrubbed before the LLM ever sees it, and the
    produced text is refused if it still matches a card pattern -- raw
    episodes must never leak into semantic facts unscrubbed. The summary
    upserts into the `episodic_summary` slot, superseding the previous
    rollup (history keeps the old ones).
    """
    summary_days = (
        config.EPISODIC_SUMMARY_DAYS if summary_days is None else summary_days
    )
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=summary_days)
    ).isoformat()
    episodes = await asyncio.to_thread(
        episode_store.list_episodes_older_than,
        app_name=app_name,
        user_id=user_id,
        cutoff_iso=cutoff,
    )
    report = {
        "layer": "episodic",
        "dry_run": dry_run,
        "summary_days": summary_days,
        "candidates": len(episodes),
        "summarized": 0,
        "dropped": 0,
        "summary": None,
    }
    if not episodes:
        return report
    # Belt and braces: episodes are scrubbed at storage time, but scrub
    # again before the summarizer (LLM or otherwise) sees the text.
    texts = [scrub_pii(e.get("text") or "").strip() for e in episodes]
    texts = [t for t in texts if t]
    if not texts:
        return report
    prior = None
    for fact in semantic_store.active_facts():
        if fact.slot == SUMMARY_SLOT:
            prior = fact.text
            break
    summary = summarizer.summarize(texts, prior_summary=prior)
    if not summary or CARD_RE.search(summary):
        raise ValueError(
            "summarizer produced empty or card-number-shaped text; refusing"
        )
    report["summary"] = summary
    report["summarized"] = len(episodes)
    if dry_run:
        report["dropped"] = len(episodes)
        return report
    res = semantic_store.upsert(SUMMARY_SLOT, summary, confidence=SUMMARY_CONFIDENCE)
    if res.get("error"):
        raise RuntimeError(f"summary upsert failed: {res['error']}")
    for e in episodes:
        await asyncio.to_thread(episode_store.delete_episode, episode_id=e["id"])
    report["dropped"] = len(episodes)
    log_memory_write(
        app_name,
        user_id,
        "maintenance",
        {
            "action": "summarize_episodes",
            "episodes": len(episodes),
            "summary_days": summary_days,
        },
    )
    return report


# --- Semantic review --------------------------------------------------------


@dataclass
class ReviewItem:
    slot: str
    text: str
    action: str  # purge_expired | purge_superseded | flag_*
    reason: str


def _fact_age_days(fact, today: str) -> int | None:
    return _days_between((fact.confirmed_at or "")[:10], today)


def review_semantic(store, today: str | None = None) -> list[ReviewItem]:
    """Classify every fact: purge (unambiguous) or flag (needs a human).

    Purges: expired facts, superseded facts older than retention.
    Flags (drafts for human review -- never auto-applied): low confidence,
    stale (unconfirmed too long), conflicted (slot with >1 active fact).
    """
    today = today or _today_iso()
    min_conf = config.FACT_MIN_CONFIDENCE
    reconfirm_days = config.FACT_RECONFIRM_DAYS
    retention_days = config.SUPERSEDED_RETENTION_DAYS
    items: list[ReviewItem] = []
    facts = store.history()

    for f in facts:
        if _purgeable_expired(f):
            items.append(
                ReviewItem(f.slot, f.text, "purge_expired",
                           f"expired {f.expires}")
            )
        elif f.superseded_by is not None:
            age = _days_between((f.updated_at or "")[:10], today)
            if age is not None and age > retention_days:
                items.append(
                    ReviewItem(f.slot, f.text, "purge_superseded",
                               f"superseded {age}d ago (retention {retention_days}d)")
                )

    active = [f for f in facts if f.active]
    by_slot: dict[str, list] = {}
    for f in active:
        by_slot.setdefault(f.slot, []).append(f)
    for slot, group in by_slot.items():
        if len(group) > 1:
            for f in group:
                items.append(
                    ReviewItem(slot, f.text, "flag_conflicted",
                               f"{len(group)} active facts share this slot")
                )

    for f in active:
        if f.confidence < min_conf:
            items.append(
                ReviewItem(f.slot, f.text, "flag_low_confidence",
                           f"confidence {f.confidence} < {min_conf}")
            )
        age = _fact_age_days(f, today)
        if age is not None and age > reconfirm_days:
            items.append(
                ReviewItem(f.slot, f.text, "flag_stale",
                           f"unconfirmed for {age}d (limit {reconfirm_days}d)")
            )
    return items


def apply_semantic_review(store, items: list[ReviewItem], dry_run: bool = True) -> dict:
    """Execute the purge half of a review; flags stay as drafts.

    Purges are unambiguous (expired / long-dead superseded). Flags are
    returned for human review -- the re-confirm path is store.reconfirm().
    """
    flags = [i for i in items if i.action.startswith("flag_")]
    report = {
        "layer": "semantic",
        "dry_run": dry_run,
        "purged_expired": 0,
        "purged_superseded": 0,
        "flags": [
            {"slot": i.slot, "action": i.action, "reason": i.reason}
            for i in flags
        ],
    }
    if dry_run:
        report["would_purge_expired"] = sum(
            1 for i in items if i.action == "purge_expired"
        )
        report["would_purge_superseded"] = sum(
            1 for i in items if i.action == "purge_superseded"
        )
        return report
    report["purged_expired"] = store.purge_expired()
    report["purged_superseded"] = store.purge_superseded_older_than(
        config.SUPERSEDED_RETENTION_DAYS
    )
    log_memory_write(
        getattr(store, "_app_name", config.APP_NAME),
        getattr(store, "_user_id", "?"),
        "maintenance",
        {
            "action": "apply_semantic_review",
            "purged_expired": report["purged_expired"],
            "purged_superseded": report["purged_superseded"],
            "flagged": len(flags),
        },
    )
    return report


# --- Procedural -------------------------------------------------------------


def procedural_status(policy: dict | None = None) -> dict:
    """Which rules are deprecated (stop firing) vs active."""
    policy = policy if policy is not None else config.POLICY
    deprecated = sorted(git_rules.deprecated_rules(policy))
    return {
        "layer": "procedural",
        "deprecated": deprecated,
        "active": [r for r in git_rules.KNOWN_RULES if r not in deprecated],
    }


def check_deprecated_rules() -> dict:
    """Verify every deprecated rule returns a not-fired verdict.

    Calls each deprecated rule with representative args; a deprecated
    rule must short-circuit before touching them.
    """
    from support_agent import rules as rules_mod

    calls = {
        "refund_approval": (rules_mod.evaluate_refund_request, (10.0,)),
        "return_eligibility": (
            rules_mod.check_return_eligibility,
            ({"order_id": "ORD-1", "status": "delivered"},),
        ),
        "escalation": (rules_mod.should_escalate, ({"legal_threat": True},)),
        "late_delivery_credit": (
            rules_mod.late_delivery_credit,
            ({"order_id": "ORD-1"},),
        ),
    }
    results = {}
    for name in sorted(config.DEPRECATED_RULES):
        fn, args = calls[name]
        try:
            out = fn(*args)
            results[name] = {
                "not_fired": bool(out.get("deprecated"))
                and out.get("decision") == "not_fired",
                "verdict": out.get("decision"),
            }
        except Exception as exc:  # never break maintenance on a rule bug
            results[name] = {"not_fired": False, "error": str(exc)}
    return {"layer": "procedural", "deprecated_checks": results}


# --- Metrics ----------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def audit_metrics(events: list[dict]) -> dict:
    """Turn audit events into memory-layer metrics.

    Token figures are chars/4 estimates, always labeled as such; cost uses
    the placeholder rates from policy.yaml (input side only -- model
    output tokens are not logged).
    """
    turns = [e for e in events if e.get("event") == "turn"]
    recalls = [e for e in events if e.get("event") == "memory_recall"]
    n_recalls = len(recalls)
    sessions = sorted({e.get("session_id") for e in turns if e.get("session_id")})

    def _rate(key: str) -> float:
        if not n_recalls:
            return 0.0
        if key == "injected":
            return sum(1 for r in recalls if r.get("injected")) / n_recalls
        return sum(1 for r in recalls if r.get(key, 0) > 0) / n_recalls

    avg_episodic_chars = _mean([r.get("episodic_chars", 0) for r in recalls])
    avg_semantic_chars = _mean([r.get("semantic_chars", 0) for r in recalls])
    avg_prompt_chars = _mean([t.get("prompt_chars", 0) for t in turns])
    rate = config.LLM_COST_PER_MTOK_INPUT_USD
    avg_input_tokens = (avg_prompt_chars + avg_episodic_chars + avg_semantic_chars) / CHARS_PER_TOKEN

    per_ticket_chars: dict[str, float] = {}
    for t in turns:
        sid = t.get("session_id") or "unknown"
        per_ticket_chars[sid] = per_ticket_chars.get(sid, 0.0) + t.get("prompt_chars", 0)
    for r in recalls:
        sid = r.get("session_id") or "unknown"
        per_ticket_chars[sid] = per_ticket_chars.get(sid, 0.0) + r.get(
            "episodic_chars", 0
        ) + r.get("semantic_chars", 0)
    ticket_costs = [c / CHARS_PER_TOKEN / 1e6 * rate for c in per_ticket_chars.values()]

    return {
        "turns": len(turns),
        "tickets": len(sessions),
        "recall": {
            "overall_hit_rate": round(_rate("injected"), 3),
            "episodic_hit_rate": round(_rate("episodic_hits"), 3),
            "semantic_hit_rate": round(_rate("semantic_hits"), 3),
            "avg_episodic_chars": round(avg_episodic_chars, 1),
            "avg_semantic_chars": round(avg_semantic_chars, 1),
            "avg_episodic_tokens_est": round(avg_episodic_chars / CHARS_PER_TOKEN, 1),
            "avg_semantic_tokens_est": round(avg_semantic_chars / CHARS_PER_TOKEN, 1),
        },
        "avg_prompt_chars": round(avg_prompt_chars, 1),
        "est_input_cost_usd_per_turn": round(avg_input_tokens / 1e6 * rate, 6),
        "est_input_cost_usd_per_ticket": round(_mean(ticket_costs), 6),
        "cost_note": (
            "input-side estimate only (model output tokens are not logged); "
            "rates are placeholders from rules/policy.yaml"
        ),
    }


# Scripted retrieval probes: (query, layer, expected substring). Derived
# from the eval scenarios; they pin "does recall return the right thing".
_PROBES = [
    ("how do I get shipping updates", "semantic", "email"),
    ("my late order", "episodic", "ORD-1001"),
    ("refund for the damaged item", "episodic", "ORD-1002"),
]


async def run_retrieval_probes(k: int = 5) -> dict:
    """Seed scratch banks and measure precision@k. Fully offline.

    Uses FakeEmbedder + FakeEpisodeStore (deterministic) and an in-memory
    SemanticBank, so this runs anywhere with no key and no Docker.
    """
    from google.adk.sessions import Session
    from google.adk.events import Event
    from google.genai import types

    from support_agent.backends.pgvector_memory import (
        FakeEmbedder,
        FakeEpisodeStore,
        PgVectorMemoryService,
    )
    from support_agent.memory_banks import EpisodicBank, SemanticBank

    app, user = "probe_app", "probe_user"
    semantic = SemanticBank()
    semantic.upsert("contact_preference", "prefers email updates")
    semantic.upsert("shipping_preference", "express shipping")

    def _ev(text: str, i: int) -> Event:
        return Event(
            id=f"p{i}",
            author="user",
            content=types.Content(role="user", parts=[types.Part(text=text)]),
        )

    session = Session(
        app_name=app,
        user_id=user,
        id="probe-s1",
        state={},
        events=[
            _ev("My order ORD-1001 arrived 5 days late, please help", 1),
            _ev("ORD-1002 arrived with a damaged item, I want a refund", 2),
            _ev("Where is my package ORD-1003", 3),
        ],
    )
    service = PgVectorMemoryService(FakeEpisodeStore(), FakeEmbedder(dim=32))
    await service.add_session_to_memory(session)
    episodic = EpisodicBank(service, app, user)

    detail = []
    hits = 0
    for query, layer, expected in _PROBES:
        if layer == "semantic":
            candidates = [f.text for f in semantic.search(query, limit=k)]
        else:
            candidates = await episodic.recall(query, limit=k)
        ok = any(expected.lower() in c.lower() for c in candidates)
        hits += ok
        detail.append({"query": query, "layer": layer, "expected": expected, "hit": ok})
    return {
        "probes": len(_PROBES),
        "hits": hits,
        "precision_at_k": round(hits / len(_PROBES), 3),
        "k": k,
        "detail": detail,
    }


def report(audit_path: str | None = None, precision: bool = False) -> dict:
    """Build the maintenance report dict (metrics from an audit JSONL)."""
    events: list[dict] = []
    if audit_path and os.path.exists(audit_path):
        with open(audit_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    out = {
        "audit_path": audit_path,
        "audit_events": len(events),
        "metrics": audit_metrics(events),
    }
    if precision:
        out["retrieval_precision"] = asyncio.run(run_retrieval_probes())
    return out


def render_report(rep: dict) -> str:
    """Human-readable maintenance report."""
    m = rep["metrics"]
    r = m["recall"]
    lines = [
        "Memory maintenance report",
        f"  audit: {rep['audit_path']} ({rep['audit_events']} events)",
        f"  turns: {m['turns']}   tickets(sessions): {m['tickets']}",
        "  recall hit rate: overall "
        f"{r['overall_hit_rate']:.0%}  "
        f"(episodic {r['episodic_hit_rate']:.0%} / "
        f"semantic {r['semantic_hit_rate']:.0%})",
        "  avg memory context per turn: "
        f"{r['avg_episodic_chars']:.0f} episodic chars "
        f"(~{r['avg_episodic_tokens_est']:.0f} tok) + "
        f"{r['avg_semantic_chars']:.0f} semantic chars "
        f"(~{r['avg_semantic_tokens_est']:.0f} tok)",
        f"  est. input cost: ${m['est_input_cost_usd_per_turn']:.6f}/turn, "
        f"${m['est_input_cost_usd_per_ticket']:.6f}/ticket",
        f"  ({m['cost_note']})",
    ]
    if "retrieval_precision" in rep:
        p = rep["retrieval_precision"]
        lines.append(
            f"  retrieval precision@{p['k']}: {p['precision_at_k']:.0%} "
            f"({p['hits']}/{p['probes']} probes)"
        )
    return "\n".join(lines)


# --- Maintenance pass -------------------------------------------------------


async def _run_async(
    app_name: str, user_id: str, layers: tuple[str, ...], dry_run: bool
) -> dict:
    from support_agent.backends import (
        make_memory_service,
        make_semantic_store,
        make_session_service,
    )

    results: dict = {"dry_run": dry_run, "app": app_name, "user": user_id, "layers": {}}
    want = set(layers)

    if "all" in want or "session" in want:
        results["layers"]["session"] = await compact_sessions(
            make_session_service(), app_name, user_id, dry_run=dry_run
        )

    if "all" in want or "episodic" in want:
        memory_service = make_memory_service()
        ok, why = episodic_lifecycle_supported(memory_service)
        if ok:
            results["layers"]["episodic"] = await summarize_episodes(
                memory_service.episode_store,
                make_semantic_store(app_name, user_id),
                make_summarizer(),
                app_name,
                user_id,
                dry_run=dry_run,
            )
        else:
            results["layers"]["episodic"] = {
                "layer": "episodic",
                "supported": False,
                "reason": why,
            }

    if "all" in want or "semantic" in want:
        store = make_semantic_store(app_name, user_id)
        items = review_semantic(store)
        results["layers"]["semantic"] = apply_semantic_review(
            store, items, dry_run=dry_run
        )

    if "all" in want or "procedural" in want:
        status = procedural_status()
        status["checks"] = check_deprecated_rules()
        results["layers"]["procedural"] = status

    return results


def run(
    app_name: str | None = None,
    user_id: str | None = None,
    layers: tuple[str, ...] = ("all",),
    dry_run: bool = True,
) -> dict:
    """Run the maintenance pass. Dry-run by default; --user is required."""
    if not user_id:
        raise ValueError("--user is required: maintenance operates per user")
    return asyncio.run(
        _run_async(app_name or config.APP_NAME, user_id, layers, dry_run)
    )


# --- CLI --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m support_agent.maintenance",
        description="Phase 4: the forgetting layer. Dry-run is the safe default.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser(
        "report",
        help="memory metrics computed from an audit JSONL file",
    )
    rp.add_argument("--audit", required=True, help="path to the audit JSONL")
    rp.add_argument(
        "--precision",
        action="store_true",
        help="also run the offline retrieval-precision probes",
    )

    rn = sub.add_parser("run", help="maintenance pass (dry-run unless --execute)")
    rn.add_argument("--user", required=True, help="user_id to maintain")
    rn.add_argument("--app", default=None, help="app_name (default from config)")
    rn.add_argument(
        "--layer",
        default="all",
        choices=["all", "session", "episodic", "semantic", "procedural"],
    )
    rn.add_argument(
        "--execute",
        action="store_true",
        help="perform the actions (default is dry-run)",
    )

    args = parser.parse_args(argv)
    if args.cmd == "report":
        print(render_report(report(args.audit, precision=args.precision)))
    elif args.cmd == "run":
        print(
            json.dumps(
                run(
                    app_name=args.app,
                    user_id=args.user,
                    layers=(args.layer,),
                    dry_run=not args.execute,
                ),
                indent=2,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
