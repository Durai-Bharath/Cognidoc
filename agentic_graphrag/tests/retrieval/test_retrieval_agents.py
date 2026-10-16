"""A4 Vector Retrieval, A5 Graph Traversal, A6 Community Retrieval, A7 Evidence
Aggregation, A8 Context Construction."""

from __future__ import annotations

from common.text import count_tokens
from construction.models import STRUCTURAL_PREDICATES, entity_node_id
from retrieval.agents.community_retrieval import CommunityRetrievalAgent, CommunityRetrievalInput
from retrieval.agents.context_construction import ContextConstructionAgent, ContextInput
from retrieval.agents.evidence_aggregation import (
    AggregationInput,
    EvidenceAggregationAgent,
    merge_items,
)
from retrieval.agents.graph_traversal import GraphTraversalAgent, GraphTraversalInput
from retrieval.agents.vector_retrieval import VectorRetrievalAgent, VectorRetrievalInput
from retrieval.config import RetrievalConfig
from retrieval.context import RetrievalAgentContext
from retrieval.graph import default_params
from retrieval.models import (
    AggregatedEvidence,
    CommunityEvidence,
    EvidenceItem,
    GraphEvidence,
    QueryType,
    RetrievalParams,
    VectorEvidence,
)
from tests.retrieval.conftest import Stores, make_rctx


def params(
    ctx: RetrievalAgentContext, qtype: QueryType = QueryType.LOCAL, **kw: object
) -> RetrievalParams:
    return default_params(ctx.settings, qtype).model_copy(update=kw)


def vector(
    ctx: RetrievalAgentContext, text: str, p: RetrievalParams | None = None
) -> VectorEvidence:
    return (
        VectorRetrievalAgent(ctx)
        .run(VectorRetrievalInput(subquery_id="q1", text=text, params=p or params(ctx)))
        .unwrap()
    )


def traverse(
    ctx: RetrievalAgentContext, text: str, p: RetrievalParams | None = None
) -> GraphEvidence:
    p = p or params(ctx)
    return (
        GraphTraversalAgent(ctx)
        .run(GraphTraversalInput(vector=vector(ctx, text, p), params=p))
        .unwrap()
    )


def communities(ctx: RetrievalAgentContext, text: str, p: RetrievalParams) -> CommunityEvidence:
    v = vector(ctx, text, p)
    g = GraphTraversalAgent(ctx).run(GraphTraversalInput(vector=v, params=p)).unwrap()
    return (
        CommunityRetrievalAgent(ctx)
        .run(CommunityRetrievalInput(vector=v, graph=g, params=p))
        .unwrap()
    )


def aggregate(ctx: RetrievalAgentContext, text: str, p: RetrievalParams) -> AggregatedEvidence:
    v = vector(ctx, text, p)
    g = GraphTraversalAgent(ctx).run(GraphTraversalInput(vector=v, params=p)).unwrap()
    c = (
        CommunityRetrievalAgent(ctx)
        .run(CommunityRetrievalInput(vector=v, graph=g, params=p))
        .unwrap()
    )
    return (
        EvidenceAggregationAgent(ctx)
        .run(AggregationInput(vector=v, graph=g, communities=c))
        .unwrap()
    )


# =========================================================================== A4


def test_vector_retrieval_ranks_hydrated_chunks_and_entities(rctx: RetrievalAgentContext) -> None:
    out = vector(rctx, "Which component has status Replaced?")
    assert out.chunks and out.entities
    scores = [c.score for c in out.chunks]
    assert scores == sorted(scores, reverse=True)
    top = out.chunks[0]
    node = rctx.graph.get_node(top.chunk_id)
    assert node is not None and node.label == "Chunk"
    assert top.text == node.properties["text"] and top.page == node.properties["page_start"]
    assert top.doc_id.startswith("doc-")
    assert all(rctx.graph.get_node(e.node_id).label == "Entity" for e in out.entities)  # type: ignore[union-attr]


def test_vector_hits_without_a_graph_node_are_dropped(rctx: RetrievalAgentContext) -> None:
    first = vector(rctx, "Which component has status Replaced?").chunks[0].chunk_id
    rctx.graph.delete_node(first)
    out = vector(rctx, "Which component has status Replaced?")
    assert first in out.stale_hits and first not in [c.chunk_id for c in out.chunks]


def test_min_score_and_entity_k_limit_results(rconfig: RetrievalConfig, stores: Stores) -> None:
    rconfig.vector.min_score = 0.999
    ctx = make_rctx(rconfig, stores)
    assert vector(ctx, "Which component has status Replaced?").chunks == []
    out = vector(
        make_rctx(RetrievalConfig(), stores), "Valve", params(ctx, entity_top_k=0, top_k=1)
    )
    assert out.entities == [] and len(out.chunks) == 1


# =========================================================================== A5


def test_traversal_seeds_and_collects_cited_facts(rctx: RetrievalAgentContext) -> None:
    g = traverse(rctx, "What is the status of the Valve?")
    valve = entity_node_id("Valve")
    assert valve in g.seeds
    fact = next(f for f in g.facts if f.source == "Valve" and f.predicate == "has_status")
    assert fact.target == "Replaced" and fact.hop == 1 and fact.confidence == 0.95
    assert fact.chunk_ids and all(rctx.graph.get_node(c) for c in fact.chunk_ids)
    assert fact.pages and fact.doc_ids
    assert not {f.predicate for f in g.facts} & STRUCTURAL_PREDICATES


def test_zero_hops_returns_seeds_only(rctx: RetrievalAgentContext) -> None:
    g = traverse(rctx, "What is the status of the Valve?", params(rctx, hops=0))
    assert g.seeds and g.facts == [] and g.paths == []


def test_multi_hop_paths_are_scored_with_decay(rconfig: RetrievalConfig, stores: Stores) -> None:
    ctx = make_rctx(rconfig, stores)
    only_named = VectorEvidence(subquery_id="q1", query_text="Hydraulic Control Unit")
    g = (
        GraphTraversalAgent(ctx)
        .run(GraphTraversalInput(vector=only_named, params=params(ctx, hops=2)))
        .unwrap()
    )
    assert g.seeds == [entity_node_id("Hydraulic Control Unit")]
    path = next(p for p in g.paths if len(p.edge_ids) == 2 and "Equation 1" in p.text)
    assert path.text == "Hydraulic Control Unit -[regulates]-> pressure -[computed_by]-> Equation 1"
    assert path.score == round(1.0 * 0.9 * 0.9 * rconfig.traversal.path_decay, 4)
    assert next(f for f in g.facts if f.predicate == "computed_by").hop == 2


def test_traversal_respects_max_nodes(rconfig: RetrievalConfig, stores: Stores) -> None:
    rconfig.traversal.max_nodes = 2
    ctx = make_rctx(rconfig, stores)
    g = traverse(ctx, "Hydraulic Control Unit", params(ctx, hops=3))
    assert len(g.visited) <= max(2, len(g.seeds))


# =========================================================================== A6


def test_community_retrieval_is_skipped_at_weight_zero(rctx: RetrievalAgentContext) -> None:
    out = communities(rctx, "Valve", params(rctx, QueryType.SIMPLE_LOOKUP))
    assert out.weight == 0 and out.items == []


def test_global_communities_ranked_by_relevance_times_weight(rctx: RetrievalAgentContext) -> None:
    out = communities(rctx, "What are the main themes?", params(rctx, QueryType.GLOBAL))
    assert out.weight == 1.0 and 0 < len(out.items) <= rctx.settings.community.top_n
    assert [h.score for h in out.items] == sorted((h.score for h in out.items), reverse=True)
    assert all(h.summary and h.members for h in out.items)
    assert all(h.score == round(h.relevance * 1.0, 4) for h in out.items)


def test_membership_and_query_type_weighting(rctx: RetrievalAgentContext) -> None:
    local = communities(rctx, "What is the status of the Valve?", params(rctx, QueryType.LOCAL))
    glob = communities(rctx, "What is the status of the Valve?", params(rctx, QueryType.GLOBAL))
    valve_local = next(h for h in local.items if "Valve" in h.members)
    valve_global = next(h for h in glob.items if "Valve" in h.members)
    assert valve_local.overlap > 0
    assert valve_local.score < valve_global.score
    assert valve_local.score == round(valve_local.relevance * 0.2, 4)


# =========================================================================== A7


def test_aggregation_builds_ranked_unified_evidence(rctx: RetrievalAgentContext) -> None:
    ae = aggregate(rctx, "What is the status of the Valve?", params(rctx, QueryType.HYBRID))
    kinds = {i.kind for i in ae.items}
    assert {"chunk", "fact", "community"} <= kinds
    assert [i.score for i in ae.items] == sorted((i.score for i in ae.items), reverse=True)
    for item in ae.items:
        if item.kind != "community":
            assert abs(item.score - item.relevance * item.confidence) < 1e-3
    assert len({i.evidence_id for i in ae.items}) == len(ae.items)


def test_feedback_confidence_lowers_fact_ranking(rctx: RetrievalAgentContext) -> None:
    p = params(rctx)
    before = aggregate(rctx, "What is the status of the Valve?", p)
    node = rctx.graph.get_node(entity_node_id("Valve"))
    assert node is not None
    node.properties["retrieval_confidence"] = 0.5
    rctx.graph.upsert_node(node)
    after = aggregate(rctx, "What is the status of the Valve?", p)
    eid = next(
        i.evidence_id for i in before.items if i.kind == "fact" and "Valve has status" in i.text
    )
    b = next(i for i in before.items if i.evidence_id == eid)
    a = next(i for i in after.items if i.evidence_id == eid)
    assert a.confidence == round(b.confidence * 0.5, 4) and a.score < b.score


def test_merge_items_deduplicates_and_unions_sources() -> None:
    one = EvidenceItem(
        evidence_id="chunk:c1",
        kind="chunk",
        text="t",
        relevance=0.4,
        score=0.4,
        sources=["vector"],
        subquery_ids=["q1"],
    )
    two = one.model_copy(
        update={"relevance": 0.6, "score": 0.6, "sources": ["graph"], "subquery_ids": ["q2"]}
    )
    merged, removed = merge_items([one, two])
    assert removed == 1 and len(merged) == 1
    assert merged[0].score == 0.6 and merged[0].sources == ["vector", "graph"]
    assert merged[0].subquery_ids == ["q1", "q2"]


def test_aggregation_caps_items(rconfig: RetrievalConfig, stores: Stores) -> None:
    rconfig.aggregation.max_items = 3
    ctx = make_rctx(rconfig, stores)
    ae = aggregate(ctx, "What is the status of the Valve?", params(ctx, QueryType.HYBRID))
    assert len(ae.items) == 3 and ae.duplicates_removed >= 0


# =========================================================================== A8


def test_context_blocks_carry_citation_ids_and_provenance(rctx: RetrievalAgentContext) -> None:
    ae = aggregate(rctx, "What is the status of the Valve?", params(rctx, QueryType.HYBRID))
    ctx = (
        ContextConstructionAgent(rctx)
        .run(
            ContextInput(
                query="q",
                query_type=QueryType.HYBRID,
                evidence=[ae, ae],
                sub_answers={"q1": "Valve"},
            )
        )
        .unwrap()
    )
    assert [b.citation_id for b in ctx.blocks] == [f"E{i}" for i in range(1, len(ctx.blocks) + 1)]
    order = {"community": 0, "path": 1, "fact": 2, "chunk": 3}
    assert [order[b.kind] for b in ctx.blocks] == sorted(order[b.kind] for b in ctx.blocks)
    assert len({b.evidence_id for b in ctx.blocks}) == len(ctx.blocks)  # merged across subqueries
    chunk = next(b for b in ctx.blocks if b.kind == "chunk")
    assert chunk.doc_title == "Hydraulic Pump Maintenance Report" and chunk.page
    assert chunk.render().startswith(f"[{chunk.citation_id}] (type=chunk, doc=Hydraulic")
    assert ctx.sub_answers == {"q1": "Valve"} and ctx.token_count <= ctx.token_budget


def test_token_budget_truncates_and_drops(rctx: RetrievalAgentContext) -> None:
    ae = aggregate(
        rctx, "Which component has status Replaced?", params(rctx, top_k=1, entity_top_k=0, hops=0)
    )
    chunk = next(i for i in ae.items if i.kind == "chunk")
    budget = count_tokens(chunk.text) // 2 + 12
    ctx = (
        ContextConstructionAgent(rctx)
        .run(
            ContextInput(query="q", query_type=QueryType.LOCAL, evidence=[ae], token_budget=budget)
        )
        .unwrap()
    )
    assert ctx.token_count <= budget
    assert len(ctx.blocks) == 1 and ctx.blocks[0].truncated and ctx.blocks[0].text.endswith("…")
    tiny = (
        ContextConstructionAgent(rctx)
        .run(ContextInput(query="q", query_type=QueryType.LOCAL, evidence=[ae], token_budget=8))
        .unwrap()
    )
    assert tiny.blocks == [] and tiny.dropped == len(ae.items)
