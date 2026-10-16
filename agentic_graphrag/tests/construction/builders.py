"""Small Stage 2 builders shared by the construction tests."""

from __future__ import annotations

from typing import Any

from common.embedder import FakeEmbedder
from common.llm import MockLLM
from common.review_queue import ReviewQueue
from construction.config import ConstructionConfig
from construction.context import ConstructionContext
from construction.models import (
    EntityRef,
    GraphEdge,
    GraphNode,
    NormalizedTriple,
    Provenance,
    ResolvedTriple,
    SemanticChunk,
    Triple,
    edge_id,
    entity_node_id,
    stable_hash,
)
from construction.ontology import load_ontology
from construction.stores import InMemoryGraphStore, InMemoryVectorStore
from construction.testing import offline_construction_llm

DOC = "doc-test"


def make_ctx(config: ConstructionConfig, llm: MockLLM | None = None) -> ConstructionContext:
    return ConstructionContext(
        config=config.as_extraction_config(),
        llm=llm or offline_construction_llm(),
        embedder=FakeEmbedder(),
        review_queue=ReviewQueue(config.paths.review_queue),
        settings=config,
        graph=InMemoryGraphStore(),
        vectors=InMemoryVectorStore(),
        ontology=load_ontology(config.paths.ontology),
        document_id=DOC,
    )


def chunk(
    text: str, *, chunk_id: str = f"{DOC}:c0", context: str = "", page: int = 1, **kw: Any
) -> SemanticChunk:
    return SemanticChunk(
        chunk_id=chunk_id,
        doc_id=kw.pop("doc_id", DOC),
        order=kw.pop("order", 0),
        text=text,
        context=context,
        section_id=kw.pop("section_id", "sec-1"),
        section_title=kw.pop("section_title", "Intro"),
        section_path=kw.pop("section_path", ["Intro"]),
        page_start=page,
        page_end=page,
        **kw,
    )


def prov(
    doc: str = DOC, chunk_id: str = f"{DOC}:c0", conf: float = 0.9, page: int | None = 1
) -> Provenance:
    return Provenance(
        doc_id=doc, chunk_id=chunk_id, page=page, confidence=conf, source_ref="sec-1/c0"
    )


def triple(
    head: str,
    relation: str,
    tail: str,
    conf: float = 0.9,
    *,
    head_type: str = "Concept",
    tail_type: str = "Concept",
    provenance: Provenance | None = None,
) -> Triple:
    p = provenance or prov(conf=conf)
    return Triple(
        triple_id="tr-" + stable_hash(p.chunk_id, head, relation, tail),
        head=head,
        head_type=head_type,
        relation=relation,
        tail=tail,
        tail_type=tail_type,
        confidence=conf,
        llm_confidence=conf,
        provenance=p.model_copy(update={"confidence": conf}),
        mentions={"head": head, "relation": relation, "tail": tail},
    )


def ref(name: str, etype: str = "Concept", *, new: bool = True) -> EntityRef:
    return EntityRef(mention=name, name=name, type=etype, node_id=entity_node_id(name), is_new=new)


def normalized(
    head: str,
    predicate: str | None,
    tail: str,
    conf: float = 0.9,
    *,
    head_type: str = "Concept",
    tail_type: str = "Concept",
    provenance: Provenance | None = None,
    inverted: bool = False,
    rejected: str | None = None,
) -> NormalizedTriple:
    t = triple(
        head,
        predicate or "unknown",
        tail,
        conf,
        head_type=head_type,
        tail_type=tail_type,
        provenance=provenance,
    )
    return NormalizedTriple(
        resolved=ResolvedTriple(triple=t, head=ref(head, head_type), tail=ref(tail, tail_type)),
        predicate=predicate,
        method="exact" if predicate else "none",
        inverted=inverted,
        rejected_reason=rejected,
    )


def add_entity(
    graph: InMemoryGraphStore, name: str, etype: str = "Concept", p: Provenance | None = None
) -> GraphNode:
    node = GraphNode(
        id=entity_node_id(name), label="Entity", name=name, type=etype, provenance=[p or prov()]
    )
    graph.upsert_node(node)
    return node


def add_edge(
    graph: InMemoryGraphStore,
    head: str,
    predicate: str,
    tail: str,
    conf: float = 0.9,
    p: Provenance | None = None,
) -> GraphEdge:
    for n in (head, tail):
        if not graph.has_node(entity_node_id(n)):
            add_entity(graph, n)
    s, t = entity_node_id(head), entity_node_id(tail)
    edge = GraphEdge(
        id=edge_id(s, predicate, t),
        source=s,
        target=t,
        predicate=predicate,
        confidence=conf,
        provenance=[(p or prov()).model_copy(update={"confidence": conf})],
    )
    graph.upsert_edge(edge)
    return edge
