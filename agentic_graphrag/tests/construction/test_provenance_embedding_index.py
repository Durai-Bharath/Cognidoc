"""A11 Provenance, A12 Embedding, A13 Vector Index."""

from __future__ import annotations

import pytest

from construction.agents.embedding import (
    EmbeddingAgent,
    EmbeddingBatch,
    EmbeddingInput,
    embedding_text,
)
from construction.agents.provenance import ProvenanceAgent, ProvenanceInput
from construction.agents.vector_index import VectorIndexAgent
from construction.context import ConstructionContext
from construction.models import IN_DOCUMENT, MENTIONED_IN, NEXT_CHUNK, GraphNode, edge_id
from construction.stores import CHUNKS, ENTITIES, InMemoryVectorStore, VectorRecord
from tests.construction.builders import DOC, add_edge, chunk, prov


def _doc(ctx: ConstructionContext) -> None:
    ctx.graph.upsert_node(
        GraphNode(id=DOC, label="Document", name="Doc", provenance=[prov(chunk_id="document")])
    )


def _provenance(ctx: ConstructionContext, c, nodes, edges):  # type: ignore[no-untyped-def]
    return ProvenanceAgent(ctx).run(
        ProvenanceInput(chunk=c, touched_nodes=nodes, touched_edges=edges)
    )


# ----------------------------------------------------------------------------- A11


def test_chunk_node_structural_links_and_mentions(cctx: ConstructionContext) -> None:
    _doc(cctx)
    e = add_edge(cctx.graph, "Pump", "uses", "oil")  # type: ignore[arg-type]
    c0 = chunk("The pump uses oil.", chunk_id=f"{DOC}:c0")
    rep = _provenance(cctx, c0, [e.source, e.target], [e.id]).unwrap()
    assert rep.validation.passed and rep.checked_nodes == 2 and rep.checked_edges == 1
    node = cctx.graph.get_node(c0.chunk_id)
    assert node is not None and node.label == "Chunk" and node.properties["text"] == c0.text
    assert node.properties["section_path"] == ["Intro"] and node.provenance[0].page == 1
    assert cctx.graph.get_edge(edge_id(c0.chunk_id, IN_DOCUMENT, DOC)) is not None
    for nid in (e.source, e.target):
        mention = cctx.graph.get_edge(edge_id(nid, MENTIONED_IN, c0.chunk_id))
        assert mention is not None and mention.provenance[0].chunk_id == c0.chunk_id
    c1 = chunk("Next.", chunk_id=f"{DOC}:c1", previous_chunk=c0.chunk_id)
    _provenance(cctx, c1, [], []).unwrap()
    assert cctx.graph.get_edge(edge_id(c0.chunk_id, NEXT_CHUNK, c1.chunk_id)) is not None


def test_missing_provenance_fields_are_repaired_from_the_chunk(cctx: ConstructionContext) -> None:
    _doc(cctx)
    p = prov(page=None).model_copy(update={"source_ref": ""})
    e = add_edge(cctx.graph, "Pump", "uses", "oil", p=p)  # type: ignore[arg-type]
    rep = _provenance(cctx, chunk("x y z", page=4), [], [e.id]).unwrap()
    fixed = cctx.graph.get_edge(e.id).provenance[0]  # type: ignore[union-attr]
    assert rep.validation.passed and e.id in rep.repaired
    assert fixed.page == 4 and fixed.source_ref == "sec-1/doc-test:c0" and fixed.timestamp


def test_incomplete_provenance_fails_and_is_flagged(cctx: ConstructionContext) -> None:
    _doc(cctx)
    e = add_edge(cctx.graph, "Pump", "uses", "oil", p=prov(doc="other", chunk_id="other:c9"))  # type: ignore[arg-type]
    rep = _provenance(cctx, chunk("x y z"), ["missing-node"], [e.id, "missing-edge"]).unwrap()
    reasons = rep.validation.failure_reasons()
    assert not rep.validation.passed
    assert any("no provenance record" in r for r in reasons) and any(
        "missing-node" in r for r in reasons
    )
    assert any("missing-edge" in r for r in reasons)
    assert cctx.review_queue.items[-1].reason == "incomplete provenance"


# ----------------------------------------------------------------------------- A12


def test_entities_and_chunks_are_embedded_and_attached(cctx: ConstructionContext) -> None:
    _doc(cctx)
    e = add_edge(cctx.graph, "Pump", "uses", "oil")  # type: ignore[arg-type]
    c0 = chunk("The pump uses oil.")
    _provenance(cctx, c0, [e.source], [e.id]).unwrap()
    batch: EmbeddingBatch = (
        EmbeddingAgent(cctx)
        .run(EmbeddingInput(node_ids=[c0.chunk_id, e.source, e.source, "missing"]))
        .unwrap()
    )
    assert [i.collection for i in batch.items] == [
        CHUNKS,
        ENTITIES,
    ] and batch.dim == cctx.embedder.dim
    assert batch.items[1].text == "Pump" and batch.items[1].payload["doc_ids"] == [DOC]
    assert (
        batch.items[0].text.startswith("Intro\nThe pump")
        and batch.items[0].payload["doc_id"] == DOC
    )
    assert cctx.graph.get_embedding(e.source) is not None
    hits = cctx.graph.similar_nodes(cctx.embedder.embed_one("Pump"), top_k=1)
    assert hits[0][0].id == e.source and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    community = GraphNode(id="c", label="Community", name="T", properties={"summary": "S"})
    assert embedding_text(community) == "T\nS"


# ----------------------------------------------------------------------------- A13


def _batch(ctx: ConstructionContext) -> EmbeddingBatch:
    e = add_edge(ctx.graph, "Pump", "uses", "oil")  # type: ignore[arg-type]
    return EmbeddingAgent(ctx).run(EmbeddingInput(node_ids=[e.source, e.target])).unwrap()


def test_vectors_are_upserted_mapped_and_verified(cctx: ConstructionContext) -> None:
    batch = _batch(cctx)
    rep = VectorIndexAgent(cctx).run(batch).unwrap()
    assert rep.upserted == {ENTITIES: 2} and rep.verified == 2
    for item in batch.items:
        (rec,) = cctx.vectors.get(ENTITIES, [item.node_id])
        assert rec.payload["node_id"] == item.node_id
        assert cctx.graph.get_node(item.node_id).vector_id == item.node_id  # type: ignore[union-attr]
    VectorIndexAgent(cctx).run(batch).unwrap()  # idempotent upsert
    assert cctx.vectors.count(ENTITIES) == 2


class _LossyStore(InMemoryVectorStore):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        if self.mode == "drop":
            return
        if self.mode == "remap":
            records = [r.model_copy(update={"payload": {"node_id": "wrong"}}) for r in records]
        if self.mode == "corrupt":
            records = [r.model_copy(update={"vector": [-x for x in r.vector]}) for r in records]
        super().upsert(collection, records)


@pytest.mark.parametrize(
    ("mode", "message"), [("drop", "not found"), ("remap", "wrong node"), ("corrupt", "differs")]
)
def test_failed_verification_is_a_typed_error(
    cctx: ConstructionContext, mode: str, message: str
) -> None:
    cctx.vectors = _LossyStore(mode)
    result = VectorIndexAgent(cctx).run(_batch(cctx))
    assert not result.ok and result.error.error_type == "VectorIndexError"  # type: ignore[union-attr]
    assert message in result.error.message  # type: ignore[union-attr]
    assert cctx.review_queue.items[-1].reason == "vector indexing verification failed"
