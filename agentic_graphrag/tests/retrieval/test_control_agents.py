"""A13 Feedback, A14 Retry Controller, A15 Cache Manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from construction.models import entity_node_id
from retrieval.agents.cache_manager import CacheManagerAgent, CacheRequest, SemanticCache
from retrieval.agents.feedback import FeedbackAgent, FeedbackInput, bounded_step
from retrieval.agents.retry_controller import RetryControllerAgent, RetryInput
from retrieval.agents.self_critique import CritiqueInput, SelfCritiqueAgent
from retrieval.config import RetrievalConfig
from retrieval.context import RetrievalAgentContext
from retrieval.graph import default_params
from retrieval.models import QueryType, RetrievalParams, ValidationStatus
from tests.retrieval.conftest import Stores, make_rctx
from tests.retrieval.test_answer_agents import BAD, GOOD, QUESTION, validate, valve_context

# =========================================================================== A13


def run_feedback(ctx: RetrievalAgentContext, answer: str, *, critique: bool = True):  # type: ignore[no-untyped-def]
    grounded, vs = validate(ctx, answer)
    va = (
        SelfCritiqueAgent(ctx)
        .run(
            CritiqueInput(
                question=QUESTION, grounded=grounded, status=vs, context=valve_context(ctx)
            )
        )
        .unwrap()
        if critique
        else None
    )
    out = FeedbackAgent(ctx).run(
        FeedbackInput(query=QUESTION, grounded=grounded, status=vs, critique=va)
    )
    return out.unwrap(), valve_context(ctx)


def test_bounded_step_never_crosses_floor_or_ceiling() -> None:
    assert bounded_step(0.5, -0.1, 0.05, 0.99) == 0.4
    assert bounded_step(0.1, -0.1, 0.05, 0.99) == 0.05
    assert bounded_step(0.02, -0.1, 0.05, 0.99) == 0.02  # already below the floor: unchanged
    assert bounded_step(0.98, 0.02, 0.05, 0.99) == 0.99
    assert bounded_step(1.0, 0.02, 0.05, 0.99) == 1.0  # above the ceiling: never lowered


def test_failed_answer_penalises_implicated_knowledge(rctx: RetrievalAgentContext) -> None:
    before = valve_context(rctx)
    edge_id = before.blocks[2].edge_ids[0]
    conf = rctx.graph.get_edge(edge_id).confidence  # type: ignore[union-attr]
    fb, _ = run_feedback(rctx, BAD)
    assert fb.mode == "penalize"
    edge = rctx.graph.get_edge(edge_id)
    assert edge is not None and edge.confidence == round(conf - 0.1, 4)
    assert edge.properties["feedback"]["penalties"] == 1
    valve = rctx.graph.get_node(entity_node_id("Valve"))
    assert valve is not None and valve.properties["retrieval_confidence"] < valve.confidence
    assert edge_id in fb.changed_ids and fb.logged
    record = json.loads(Path(rctx.settings.paths.feedback_log or "").read_text().splitlines()[-1])
    assert record["mode"] == "penalize" and record["updates"]
    untouched = rctx.graph.get_edge(before.blocks[0].edge_ids[0])
    assert untouched is not None and "feedback" not in untouched.properties


def test_repeated_or_low_confidence_failures_are_flagged(rctx: RetrievalAgentContext) -> None:
    fb1, _ = run_feedback(rctx, BAD)
    assert fb1.flagged == []
    fb2, _ = run_feedback(rctx, BAD)
    assert fb2.flagged and all(i in fb2.changed_ids for i in fb2.flagged)
    items = [i for i in rctx.review_queue.items if i.agent == "feedback"]
    assert items and items[0].details["penalties"] == 2


def test_verified_answer_reinforces_supporting_knowledge(rctx: RetrievalAgentContext) -> None:
    edge_id = valve_context(rctx).blocks[0].edge_ids[0]
    fb, _ = run_feedback(rctx, GOOD, critique=False)
    assert fb.mode == "reinforce" and fb.changed_ids == [] and fb.flagged == []
    assert rctx.graph.get_edge(edge_id).confidence == 0.97  # type: ignore[union-attr]
    assert {u.direction for u in fb.updates} == {"reinforce"}


def test_reinforcement_can_be_disabled_and_state_persisted(
    rconfig: RetrievalConfig, stores: Stores
) -> None:
    rconfig.feedback.reinforce_verified = False
    fb, _ = run_feedback(make_rctx(rconfig, stores), GOOD, critique=False)
    assert fb.mode == "none" and fb.updates == []
    rconfig.feedback.persist_state = True
    run_feedback(make_rctx(rconfig, stores), BAD)
    assert (Path(rconfig.paths.state_dir) / "graph.json").exists()


# =========================================================================== A14


def status(valid: bool, score: float = 0.3) -> ValidationStatus:
    return ValidationStatus(valid=valid, score=score, grounding=score, completeness=1.0, tau=0.7)


def decide(ctx: RetrievalAgentContext, attempt: int, p: RetrievalParams, valid: bool = False):  # type: ignore[no-untyped-def]
    return (
        RetryControllerAgent(ctx)
        .run(RetryInput(status=status(valid), attempt=attempt, params=p))
        .unwrap()
    )


def test_valid_answer_is_accepted(rctx: RetrievalAgentContext) -> None:
    d = decide(rctx, 1, default_params(rctx.settings, QueryType.LOCAL), valid=True)
    assert d.action == "accept"


def test_escalation_sequence_per_attempt(rctx: RetrievalAgentContext) -> None:
    p = default_params(rctx.settings, QueryType.LOCAL)
    steps = []
    for attempt in (1, 2, 3, 4):
        d = decide(rctx, attempt, p)
        if d.action != "retry":
            break
        assert d.attempt == attempt + 1
        steps.append(d.escalation)
        p = d.params
    assert steps == ["increase_k", "increase_hops", "switch_to_hybrid"]
    assert (p.top_k, p.entity_top_k, p.hops, p.query_type) == (10, 10, 3, QueryType.HYBRID)
    assert p.community_weight == rctx.settings.community.weights["Hybrid"]
    exhausted = decide(rctx, 4, p)
    assert exhausted.action == "exhausted" and "retry limit 3" in exhausted.reason


def test_widen_communities_and_capped_steps_are_skipped(
    rconfig: RetrievalConfig, stores: Stores
) -> None:
    rconfig.retry.max_retries = 10
    ctx = make_rctx(rconfig, stores)
    p = default_params(rconfig, QueryType.LOCAL).model_copy(update={"top_k": rconfig.retry.max_k})
    d = decide(ctx, 1, p)  # increase_k is capped (entity_top_k still grows)
    assert d.escalation == "increase_k"
    capped = p.model_copy(update={"entity_top_k": rconfig.retry.max_k})
    d = decide(ctx, 1, capped)
    assert d.escalation == "increase_hops"
    d = decide(ctx, 4, capped)
    assert d.escalation == "widen_communities"
    assert d.params.community_top_n == capped.community_top_n + 2
    assert d.params.community_weight == rconfig.retry.min_community_weight


def test_no_possible_escalation_ends_the_run(rconfig: RetrievalConfig, stores: Stores) -> None:
    r = rconfig.retry
    maxed = RetrievalParams(
        query_type=QueryType.HYBRID,
        top_k=r.max_k,
        entity_top_k=r.max_k,
        hops=r.max_hops,
        community_weight=1.0,
        community_top_n=r.max_communities,
    )
    d = decide(make_rctx(rconfig, stores), 1, maxed)
    assert d.action == "exhausted" and "limit" in d.reason


# =========================================================================== A15


def store(
    ctx: RetrievalAgentContext,
    query: str,
    answer: str = GOOD,
    verified: bool = True,
    history: str = "",
):  # type: ignore[no-untyped-def]
    grounded, _ = validate(ctx, answer)
    return (
        CacheManagerAgent(ctx)
        .run(
            CacheRequest(
                op="store",
                query=query,
                verified=verified,
                grounded=grounded,
                history_hash=history,
                score=0.9,
            )
        )
        .unwrap()
    )


def lookup(ctx: RetrievalAgentContext, query: str, history: str = ""):  # type: ignore[no-untyped-def]
    return (
        CacheManagerAgent(ctx)
        .run(CacheRequest(op="lookup", query=query, history_hash=history))
        .unwrap()
    )


def test_only_verified_answers_are_stored(rctx: RetrievalAgentContext) -> None:
    assert not store(rctx, QUESTION, verified=False).stored and len(rctx.cache) == 0
    out = store(rctx, QUESTION)
    assert out.stored and len(rctx.cache) == 1
    entry = rctx.cache.get(out.key or "")
    assert entry is not None and entry.fingerprints and all(entry.fingerprints.values())


def test_cache_hit_similar_hit_and_misses(rconfig: RetrievalConfig, stores: Stores) -> None:
    rconfig.cache.similarity_threshold = 0.9
    ctx = make_rctx(rconfig, stores)
    store(ctx, QUESTION)
    hit = lookup(ctx, QUESTION)
    assert hit.hit and hit.entry is not None and hit.entry.hits == 1 and hit.similarity == 1.0
    assert lookup(ctx, "What is the status and the interval of the Valve?").hit  # same terms
    assert not lookup(ctx, "Which standard do technicians follow?").hit
    assert not lookup(ctx, QUESTION, history="other-conversation").hit


def test_invalidation_by_element_and_on_graph_change(rctx: RetrievalAgentContext) -> None:
    key = store(rctx, QUESTION).key
    edge_id = valve_context(rctx).blocks[0].edge_ids[0]
    out = CacheManagerAgent(rctx).run(CacheRequest(op="invalidate", element_ids=[edge_id])).unwrap()
    assert out.invalidated == [key] and len(rctx.cache) == 0

    store(rctx, QUESTION)
    edge = rctx.graph.get_edge(edge_id)
    assert edge is not None
    edge.confidence = 0.5  # feedback-style change: not part of the fingerprint
    rctx.graph.upsert_edge(edge)
    assert lookup(rctx, QUESTION).hit
    edge.provenance[0] = edge.provenance[0].model_copy(update={"confidence": 0.4})  # Stage 2 change
    rctx.graph.upsert_edge(edge)
    stale = lookup(rctx, QUESTION)
    assert not stale.hit and stale.invalidated and stale.reason.startswith("stale")
    assert len(rctx.cache) == 0


def test_purge_stale_after_construction(rctx: RetrievalAgentContext) -> None:
    store(rctx, QUESTION)
    store(rctx, "Which standard do technicians follow?")
    rctx.graph.delete_node(entity_node_id("Valve"))
    out = CacheManagerAgent(rctx).run(CacheRequest(op="purge_stale")).unwrap()
    assert len(out.invalidated) == 2 and len(rctx.cache) == 0


def test_cache_persists_and_can_be_disabled(rconfig: RetrievalConfig, stores: Stores) -> None:
    ctx = make_rctx(rconfig, stores)
    store(ctx, QUESTION)
    reloaded = SemanticCache(rconfig.paths.cache_file)
    assert len(reloaded) == 1 and reloaded.entries[0].query == QUESTION
    rconfig.cache.enabled = False
    off = make_rctx(rconfig, stores)
    assert not lookup(off, QUESTION).hit and not store(off, QUESTION).stored


@pytest.mark.parametrize("max_entries", [1])
def test_cache_evicts_oldest(tmp_path: Path, max_entries: int, rctx: RetrievalAgentContext) -> None:
    rctx.cache.max_entries = max_entries
    store(rctx, QUESTION)
    store(rctx, "Which standard do technicians follow?")
    assert [e.query for e in rctx.cache.entries] == ["Which standard do technicians follow?"]
