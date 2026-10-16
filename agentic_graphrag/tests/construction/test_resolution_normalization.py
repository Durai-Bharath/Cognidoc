"""A6 Entity Resolution and A7 Relation Normalization."""

from __future__ import annotations

from common.llm import LLMError, MockLLM
from construction.agents.entity_resolution import (
    EntityResolutionAgent,
    EntityResolutionInput,
    ResolvedTriples,
)
from construction.agents.relation_normalization import (
    NormalizationInput,
    NormalizedTriples,
    RelationNormalizationAgent,
)
from construction.config import ConstructionConfig
from construction.context import ConstructionContext
from construction.models import ResolvedTriple, entity_node_id
from tests.construction.builders import add_entity, make_ctx, ref, triple


def _resolve(ctx: ConstructionContext, triples) -> ResolvedTriples:  # type: ignore[no-untyped-def]
    return EntityResolutionAgent(ctx).run(EntityResolutionInput(triples=triples)).unwrap()


def _embed(ctx: ConstructionContext, name: str) -> None:
    ctx.graph.set_embedding(entity_node_id(name), ctx.embedder.embed_one(name))


# ----------------------------------------------------------------------------- A6


def test_exact_canonical_match_links_existing_node(cctx: ConstructionContext) -> None:
    add_entity(cctx.graph, "Hydraulic Pump", "Component")  # type: ignore[arg-type]
    out = _resolve(cctx, [triple("hydraulic pump", "uses", "oil", head_type="Component")])
    head = out.items[0].head
    assert not head.is_new and head.similarity == 1.0 and head.name == "Hydraulic Pump"
    assert out.items[0].tail.is_new and out.linked == 1 and out.new == 1
    assert cctx.graph.get_node(entity_node_id("oil")) is None  # nothing is written by A6


def test_similarity_at_or_above_delta_links(cctx: ConstructionContext) -> None:
    add_entity(cctx.graph, "pump seal assembly")  # type: ignore[arg-type]
    _embed(cctx, "pump seal assembly")
    cctx.settings.entity_resolution.delta = 0.8
    out = _resolve(cctx, [triple("seal assembly pump", "uses", "oil")])  # same words, other order
    head = out.items[0].head
    assert not head.is_new and head.node_id == entity_node_id("pump seal assembly")
    assert head.similarity >= 0.8 and head.mention == "seal assembly pump"


def test_similarity_below_delta_creates_new_entity(cctx: ConstructionContext) -> None:
    add_entity(cctx.graph, "hydraulic pump")  # type: ignore[arg-type]
    _embed(cctx, "hydraulic pump")
    out = _resolve(cctx, [triple("pump", "uses", "oil")])  # cosine ≈ 0.71 < δ = 0.85
    assert out.items[0].head.is_new and out.items[0].head.node_id == entity_node_id("pump")


def test_type_incompatibility_blocks_linking(cctx: ConstructionContext) -> None:
    add_entity(cctx.graph, "Mercury", "Material")  # type: ignore[arg-type]
    out = _resolve(cctx, [triple("Mercury", "located_in", "space", head_type="Location")])
    assert out.items[0].head.is_new
    cctx.settings.entity_resolution.require_type_compatibility = False
    out = _resolve(cctx, [triple("Mercury", "located_in", "space", head_type="Location")])
    assert not out.items[0].head.is_new
    # Concept nodes link with typed mentions and take the more specific type.
    add_entity(cctx.graph, "Valve", "Concept")  # type: ignore[arg-type]
    cctx.settings.entity_resolution.require_type_compatibility = True
    out = _resolve(cctx, [triple("valve", "uses", "oil", head_type="Component")])
    assert not out.items[0].head.is_new and out.items[0].head.type == "Component"


def test_near_duplicate_new_mentions_in_one_batch_share_a_node(cctx: ConstructionContext) -> None:
    cctx.settings.entity_resolution.delta = 0.8
    out = _resolve(
        cctx, [triple("pump seal kit", "uses", "oil"), triple("seal kit pump", "uses", "grease")]
    )
    a, b = out.items[0].head, out.items[1].head
    assert a.is_new and b.is_new and a.node_id == b.node_id and b.mention == "seal kit pump"


# ----------------------------------------------------------------------------- A7


def _normalize(ctx: ConstructionContext, *relations: str) -> NormalizedTriples:
    items = [
        ResolvedTriple(triple=triple("A", r, "B"), head=ref("A"), tail=ref("B")) for r in relations
    ]
    return RelationNormalizationAgent(ctx).run(NormalizationInput(items=items)).unwrap()


def test_relations_map_by_exact_synonym_inverse_and_stem(cctx: ConstructionContext) -> None:
    out = _normalize(cctx, "regulates", "is inspected every", "contains", "conforms to")
    got = [(i.predicate, i.method, i.inverted) for i in out.items]
    assert got == [
        ("regulates", "exact", False),
        ("has_interval", "synonym", False),
        ("part_of", "synonym", True),
        ("complies_with", "synonym", False),
    ]
    inv = out.items[2]
    assert inv.subject.name == "B" and inv.object.name == "A"
    stemmed = _normalize(cctx, "generated").items[0]  # "generates" synonym of produces
    assert stemmed.predicate == "produces" and stemmed.method == "stem"


def test_llm_fallback_maps_unmatched_relation(cconfig: ConstructionConfig) -> None:
    llm = MockLLM({"relation_normalization": {"predicate": "regulates", "inverted": True}})
    ctx = make_ctx(cconfig, llm)
    out = _normalize(ctx, "is governed through", "is governed through")
    assert all(i.predicate == "regulates" and i.method == "llm" and i.inverted for i in out.items)
    assert len(llm.calls) == 1  # cached per relation phrase


def test_unmatched_relation_is_rejected(cconfig: ConstructionConfig) -> None:
    for llm in (
        MockLLM({"relation_normalization": {"predicate": "NONE"}}),
        MockLLM({"relation_normalization": {"predicate": "made_up_predicate"}}),
        MockLLM({"relation_normalization": LLMError("down")}),
    ):
        out = _normalize(make_ctx(cconfig, llm), "teleports")
        assert out.items[0].predicate is None and "no ontology predicate" in (
            out.items[0].rejected_reason or ""
        )
        assert len(out.rejected) == 1
    cconfig.relations.llm_fallback = False
    llm = MockLLM({"relation_normalization": {"predicate": "uses"}})
    out = _normalize(make_ctx(cconfig, llm), "teleports")
    assert out.items[0].predicate is None and llm.calls == []
