"""A14 Construction Validation, A15 Graph Health Audit and the CommunityBuilder post-step."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from common.llm import LLMError, MockLLM
from construction.agents.construction_validation import (
    ConstructionValidationAgent,
    ConstructionValidationReport,
    ValidationRequest,
)
from construction.agents.embedding import EmbeddingAgent, EmbeddingInput
from construction.agents.health_audit import GraphHealthAuditAgent
from construction.agents.provenance import ProvenanceAgent, ProvenanceInput
from construction.agents.vector_index import VectorIndexAgent
from construction.community import CommunityBuilder, CommunityInput, CommunityReport
from construction.config import ConstructionConfig
from construction.context import ConstructionContext
from construction.models import MEMBER_OF, GraphEdge, GraphNode, edge_id, entity_node_id
from construction.stores import COMMUNITIES, ENTITIES, VectorRecord
from tests.construction.builders import DOC, add_edge, add_entity, chunk, make_ctx, prov

FACTS = [
    ("Pump", "uses", "oil"),
    ("Pump", "requires", "seal"),
    ("seal", "part_of", "Pump"),
    ("Valve", "regulates", "pressure"),
    ("Valve", "uses", "spring"),
    ("spring", "part_of", "Valve"),
]


def build_clean_graph(ctx: ConstructionContext, facts=FACTS) -> None:  # type: ignore[no-untyped-def]
    """Document + one chunk + facts, run through A11-A13 like the pipeline does."""
    ctx.graph.upsert_node(
        GraphNode(id=DOC, label="Document", name="Doc", provenance=[prov(chunk_id="document")])
    )
    edges = [add_edge(ctx.graph, h, r, t) for h, r, t in facts]  # type: ignore[arg-type]
    nodes = sorted({n for e in edges for n in (e.source, e.target)})
    c = chunk(" ".join(f"{h} {r} {t}." for h, r, t in facts))
    ProvenanceAgent(ctx).run(
        ProvenanceInput(chunk=c, touched_nodes=nodes, touched_edges=[e.id for e in edges])
    ).unwrap()
    batch = EmbeddingAgent(ctx).run(EmbeddingInput(node_ids=[c.chunk_id, *nodes])).unwrap()
    VectorIndexAgent(ctx).run(batch).unwrap()


def _validate(ctx: ConstructionContext) -> ConstructionValidationReport:
    return ConstructionValidationAgent(ctx).run(ValidationRequest()).unwrap()


def _failed(report: ConstructionValidationReport, check: str) -> list[str]:
    c = report.validation.get(check)
    assert c is not None
    return c.reasons


# ----------------------------------------------------------------------------- A14


def test_clean_graph_passes_validation(cctx: ConstructionContext) -> None:
    build_clean_graph(cctx)
    rep = _validate(cctx)
    assert rep.passed, rep.validation.failure_reasons()
    assert rep.stats["ontology_compliance_ratio"] == 1.0 and rep.stats["entity_components"] == 2
    assert _failed(rep, "single_entity_component") == [
        "2 disconnected entity components"
    ]  # warning only


def test_ontology_violations_are_detected(cctx: ConstructionContext) -> None:
    build_clean_graph(cctx)
    node = cctx.graph.get_node(entity_node_id("oil"))
    assert node is not None
    cctx.graph.upsert_node(node.model_copy(update={"type": "Alien"}))
    bad = add_edge(cctx.graph, "Report", "authored_by", "42")  # type: ignore[arg-type]
    for n, t in (("Report", "Document"), ("42", "Quantity")):
        cctx.graph.upsert_node(
            cctx.graph.get_node(entity_node_id(n)).model_copy(update={"type": t})
        )  # type: ignore[union-attr]
    cctx.graph.upsert_edge(
        GraphEdge(
            id=edge_id(DOC, MEMBER_OF, f"{DOC}:c0"),
            source=DOC,
            target=f"{DOC}:c0",
            predicate=MEMBER_OF,
            provenance=[prov()],
        )
    )
    rep = _validate(cctx)
    reasons = " ".join(_failed(rep, "ontology_compliance"))
    assert (
        "unknown type 'Alien'" in reasons
        and "range" in reasons
        and "MEMBER_OF must link" in reasons
    )
    assert entity_node_id("oil") in rep.invalid_entities and bad.id in rep.invalid_relations
    assert not rep.passed and rep.stats["ontology_compliance_ratio"] < 1.0


def test_consistency_provenance_and_connectivity_failures(cctx: ConstructionContext) -> None:
    build_clean_graph(cctx)
    e = cctx.graph.edges("uses")[0]
    cctx.graph.upsert_edge(e.model_copy(update={"id": "rel-bogus", "provenance": []}))
    add_entity(cctx.graph, "Floating")  # type: ignore[arg-type]  # never MENTIONED_IN a chunk
    cctx.graph.upsert_node(
        GraphNode(id="lonely-chunk", label="Chunk", name="c", provenance=[prov()])
    )
    rep = _validate(cctx)
    assert any("id does not match" in r for r in _failed(rep, "node_edge_consistency"))
    assert any("rel-bogus: no provenance" in r for r in _failed(rep, "provenance_coverage"))
    conn = " ".join(_failed(rep, "graph_connectivity"))
    assert (
        "lonely-chunk is not linked to a document" in conn and "is not linked to any chunk" in conn
    )


def test_vector_mapping_integrity(cctx: ConstructionContext) -> None:
    build_clean_graph(cctx)
    cctx.vectors.delete(ENTITIES, [entity_node_id("oil")])
    cctx.vectors.upsert(
        ENTITIES,
        [VectorRecord(id="ghost", vector=[0.0] * cctx.embedder.dim, payload={"node_id": "ghost"})],
    )
    rec = cctx.vectors.get(ENTITIES, [entity_node_id("seal")])[0]
    cctx.vectors.upsert(ENTITIES, [rec.model_copy(update={"payload": {"node_id": "x"}})])
    node = cctx.graph.get_node(entity_node_id("spring"))
    cctx.graph.upsert_node(node.model_copy(update={"vector_id": None}))  # type: ignore[union-attr]
    reasons = " ".join(_failed(_validate(cctx), "vector_mapping_integrity"))
    assert "vector missing" in reasons and "dangling vector" in reasons
    assert "maps to x" in reasons and "has no vector_id" in reasons


def test_invalid_entity_names_are_warnings(cctx: ConstructionContext) -> None:
    build_clean_graph(cctx, [("it", "uses", "oil"), ("Pump", "uses", "x" * 200)])
    rep = _validate(cctx)
    assert len(_failed(rep, "valid_entities_and_relations")) == 2 and rep.passed


# ----------------------------------------------------------------------------- A15


def _audit(ctx: ConstructionContext):  # type: ignore[no-untyped-def]
    return GraphHealthAuditAgent(ctx).run(_validate(ctx)).unwrap()


def test_audit_metrics_on_clean_graph(cctx: ConstructionContext) -> None:
    build_clean_graph(cctx)
    a = _audit(cctx)
    assert (
        a.orphans == []
        and a.duplicates == []
        and a.stale_edges == []
        and a.low_confidence_edges == []
    )
    m = a.metrics
    assert (
        m["entities"] == 6 and m["semantic_edges"] == 6 and m["chunks"] == 1 and m["documents"] == 1
    )
    assert m["provenance_coverage"] == 1.0 and m["vector_coverage"] == 1.0
    assert a.components["count"] == 2 and a.components["sizes"] == [3, 3]
    assert 0 < a.health_score <= 1 and m["health_score"] == a.health_score


def test_audit_detects_orphans_duplicates_stale_and_low_confidence(
    cctx: ConstructionContext,
) -> None:
    build_clean_graph(cctx)
    cctx.graph.upsert_node(
        GraphNode(id="orphan", label="Entity", name="Orphan", provenance=[prov()])
    )
    dup = add_entity(cctx.graph, "Pumps")  # type: ignore[arg-type]  # same mention key as "Pump"
    twin = add_entity(cctx.graph, "oil pump valve")  # type: ignore[arg-type]
    cctx.graph.set_embedding(twin.id, cctx.embedder.embed_one("valve oil pump"))
    cctx.graph.set_embedding(dup.id, cctx.embedder.embed_one("pump valve oil"))
    old = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    stale = add_edge(
        cctx.graph, "Valve", "measures", "flow", p=prov().model_copy(update={"timestamp": old})
    )  # type: ignore[arg-type]
    gone = add_edge(
        cctx.graph, "Valve", "produces", "noise", 0.3, p=prov(doc="doc-deleted", chunk_id="x")
    )  # type: ignore[arg-type]
    a = _audit(cctx)
    assert "orphan" in a.orphans and dup.id in a.orphans
    reasons = {(d.reason, frozenset((d.a_name, d.b_name))) for d in a.duplicates}
    assert ("same mention key", frozenset({"Pump", "Pumps"})) in reasons
    assert ("embedding similarity", frozenset({"Pumps", "oil pump valve"})) in reasons
    assert set(a.stale_edges) == {stale.id, gone.id} and a.low_confidence_edges == [gone.id]
    assert a.metrics["stale_ratio"] > 0 and a.metrics["orphan_ratio"] > 0
    assert cctx.review_queue.items[-1].reason == "graph health issues"


# ----------------------------------------------------------------------------- CommunityBuilder


def _communities(ctx: ConstructionContext) -> CommunityReport:
    return CommunityBuilder(ctx).run(CommunityInput(document_id=DOC)).unwrap()


def test_communities_are_detected_summarised_linked_and_indexed(
    cconfig: ConstructionConfig,
) -> None:
    ctx = make_ctx(cconfig)
    build_clean_graph(ctx)
    rep = _communities(ctx)
    assert rep.total == 2 and rep.created == 2 and rep.llm_summaries == 2 and rep.modularity
    comms = ctx.graph.nodes("Community")
    assert sorted(c.properties["size"] for c in comms) == [3, 3]
    for c in comms:
        members = {e.source for e in ctx.graph.find_edges(target=c.id, predicate=MEMBER_OF)}
        assert members == set(c.properties["members"]) and c.properties["summary"]
        assert c.vector_id == c.id and ctx.vectors.get(COMMUNITIES, [c.id])
        assert c.provenance[0].doc_id == DOC
    assert _validate(ctx).passed


def test_rerun_skips_unchanged_and_rebuilds_changed_communities(
    cconfig: ConstructionConfig,
) -> None:
    llm = MockLLM({"community_summary": {"title": "T", "summary": "S", "key_entities": []}})
    ctx = make_ctx(cconfig, llm)
    build_clean_graph(ctx)
    _communities(ctx)
    calls = len(llm.calls)
    again = _communities(ctx)
    assert again.unchanged == 2 and again.created == again.updated == 0 and len(llm.calls) == calls
    # Same members, new internal edge → signature changes → updated in place.
    add_edge(ctx.graph, "oil", "part_of", "seal")  # type: ignore[arg-type]
    changed = _communities(ctx)
    assert changed.updated == 1 and changed.unchanged == 1 and len(llm.calls) == calls + 1
    # A member joins → new community id; the old node and its vector are removed.
    add_edge(ctx.graph, "spring", "part_of", "lever")  # type: ignore[arg-type]
    moved = _communities(ctx)
    assert moved.created == 1 and moved.removed == 1
    ids = {c.id for c in ctx.graph.nodes("Community")}
    assert set(ctx.vectors.ids(COMMUNITIES)) == ids and len(ids) == 2


def test_community_summary_falls_back_and_respects_settings(cconfig: ConstructionConfig) -> None:
    ctx = make_ctx(cconfig, MockLLM({"community_summary": LLMError("down")}))
    build_clean_graph(ctx)
    rep = _communities(ctx)
    assert rep.llm_summaries == 0 and all(
        c.properties["summary"] for c in ctx.graph.nodes("Community")
    )
    cconfig.community.min_size = 10
    assert _communities(make_ctx(cconfig)).total == 0
    empty = make_ctx(cconfig)
    assert _communities(empty).total == 0 and _communities(empty).modularity is None
    cconfig.community.min_size = 2
    cconfig.community.algorithm = "leiden"  # falls back to louvain without leidenalg
    ctx = make_ctx(cconfig)
    build_clean_graph(ctx)
    assert _communities(ctx).total == 2
