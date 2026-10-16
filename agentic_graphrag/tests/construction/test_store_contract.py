"""One GraphStore/VectorStore contract run against every backend.

The in-memory backends always run. Neo4j and Qdrant run only with
``GRAPHRAG_LIVE_STORES=1`` and the servers from ``docker-compose.yml``, so ``make test``
stays offline.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from common.embedder import FakeEmbedder
from construction.config import ConstructionConfig
from construction.graph import AgenticGraphConstructionFramework
from construction.models import GraphEdge, GraphNode, Provenance, edge_id
from construction.stores import (
    ENTITIES,
    GraphStore,
    GraphStoreError,
    InMemoryGraphStore,
    InMemoryVectorStore,
    VectorRecord,
    VectorStore,
)
from construction.testing import make_document, offline_construction_llm

LIVE = os.environ.get("GRAPHRAG_LIVE_STORES") == "1"
live = pytest.mark.skipif(not LIVE, reason="set GRAPHRAG_LIVE_STORES=1 with docker compose up")


def _neo4j() -> GraphStore:
    from construction.stores.neo4j_store import Neo4jGraphStore

    store = Neo4jGraphStore(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "graphrag-dev"),
    )
    store._run("MATCH (n:KGNode) DETACH DELETE n")
    return store


def _qdrant() -> VectorStore:
    from construction.stores.qdrant_store import QdrantVectorStore

    return QdrantVectorStore(
        os.environ.get("QDRANT_URL", "http://localhost:6333"),
        prefix=f"test_{uuid.uuid4().hex[:8]}_",
    )


@pytest.fixture(params=["memory", pytest.param("neo4j", marks=live)])
def graph_store(request: pytest.FixtureRequest) -> Iterator[GraphStore]:
    store = InMemoryGraphStore() if request.param == "memory" else _neo4j()
    yield store
    store.close()


@pytest.fixture(params=["memory", pytest.param("qdrant", marks=live)])
def vector_store(request: pytest.FixtureRequest) -> Iterator[VectorStore]:
    store = InMemoryVectorStore() if request.param == "memory" else _qdrant()
    yield store
    if request.param == "qdrant":
        _drop_collections(store)
    store.close()


def _drop_collections(store: VectorStore) -> None:
    client, prefix = store.client, store.prefix  # type: ignore[attr-defined]
    for c in client.get_collections().collections:
        if c.name.startswith(prefix):
            client.delete_collection(c.name)


def _node(nid: str, name: str, label: str = "Entity") -> GraphNode:
    p = Provenance(doc_id="d", chunk_id="c", page=1, confidence=0.9, source_ref="s")
    return GraphNode(
        id=nid,
        label=label,
        name=name,
        type="Concept",
        provenance=[p],  # type: ignore[arg-type]
        properties={"nested": {"a": [1, 2]}},
    )


def test_graph_store_contract(graph_store: GraphStore) -> None:
    g = graph_store
    stored = {
        nid: _node(nid, name) for nid, name in (("a", "alpha"), ("b", "beta"), ("c", "gamma"))
    }
    for node in stored.values():
        g.upsert_node(node)
    assert g.get_node("a") == stored["a"]  # nested properties and provenance round-trip
    assert g.get_node("missing") is None
    e1 = GraphEdge(
        id=edge_id("a", "uses", "b"),
        source="a",
        target="b",
        predicate="uses",
        confidence=0.7,
        provenance=_node("x", "x").provenance,
    )
    e2 = GraphEdge(id=edge_id("c", "uses", "b"), source="c", target="b", predicate="uses")
    g.upsert_edge(e1)
    g.upsert_edge(e2)
    g.upsert_edge(e1.model_copy(update={"confidence": 0.8}))  # replace, not duplicate
    assert g.get_edge(e1.id).confidence == 0.8 and len(g.edges()) == 2  # type: ignore[union-attr]
    assert g.get_edge(e1.id).provenance == e1.provenance  # type: ignore[union-attr]
    assert {e.id for e in g.find_edges(target="b")} == {e1.id, e2.id}
    assert [e.id for e in g.find_edges(source="a", predicate="uses")] == [e1.id]
    assert {n.id for n in g.neighbors("b", direction="in")} == {"a", "c"}
    assert [n.id for n in g.neighbors("a", direction="out")] == ["b"] and g.degree("b") == 2
    with pytest.raises(GraphStoreError):
        g.upsert_edge(GraphEdge(id="bad", source="a", target="missing", predicate="uses"))
    for nid, vec in (("a", [1, 0, 0]), ("b", [0, 1, 0]), ("c", [0.9, 0.1, 0])):
        g.set_embedding(nid, np.asarray(vec, np.float32))
    assert np.allclose(g.get_embedding("a"), [1, 0, 0])  # type: ignore[arg-type]
    hits = g.similar_nodes(np.asarray([1, 0, 0], np.float32), top_k=2)
    assert [n.id for n, _ in hits] == ["a", "c"] and hits[0][1] == pytest.approx(1.0, abs=1e-4)
    g.delete_edge(e2.id)
    assert g.get_edge(e2.id) is None
    g.delete_node("b")
    assert (
        g.get_node("b") is None
        and g.edges() == []
        and {n.id for n in g.nodes("Entity")} == {"a", "c"}
    )


def test_vector_store_contract(vector_store: VectorStore) -> None:
    v = vector_store
    v.ensure_collection(ENTITIES, 3)
    v.ensure_collection(ENTITIES, 3)  # idempotent
    v.upsert(
        ENTITIES,
        [
            VectorRecord(id="ent-a", vector=[1, 0, 0], payload={"node_id": "ent-a", "k": "x"}),
            VectorRecord(id="ent-b", vector=[0, 1, 0], payload={"node_id": "ent-b", "k": "y"}),
        ],
    )
    v.upsert(
        ENTITIES,
        [VectorRecord(id="ent-a", vector=[1, 0, 0], payload={"node_id": "ent-a", "k": "x"})],
    )
    assert sorted(v.ids(ENTITIES)) == ["ent-a", "ent-b"] and v.count(ENTITIES) == 2
    (rec,) = v.get(ENTITIES, ["ent-a"])
    assert rec.id == "ent-a" and rec.payload == {"node_id": "ent-a", "k": "x"}
    assert np.allclose(rec.vector, [1, 0, 0], atol=1e-5)
    hits = v.search(ENTITIES, np.asarray([0.1, 1, 0], np.float32), top_k=1)
    assert hits[0].id == "ent-b" and hits[0].payload["node_id"] == "ent-b"
    assert [
        h.id for h in v.search(ENTITIES, np.asarray([0, 1, 0], np.float32), where={"k": "x"})
    ] == ["ent-a"]
    v.delete(ENTITIES, ["ent-a"])
    assert v.ids(ENTITIES) == ["ent-b"] and v.get("never-created", ["x"]) == []


@live
def test_pipeline_runs_on_neo4j_and_qdrant(cconfig: ConstructionConfig, tmp_path: Path) -> None:
    graph, vectors = _neo4j(), _qdrant()
    try:
        fw = AgenticGraphConstructionFramework(
            cconfig,
            llm=offline_construction_llm(),
            embedder=FakeEmbedder(),
            graph_store=graph,
            vector_store=vectors,
        )
        table = ("Table 1", ["Component", "Interval (h)"], [["Seal", "500"]])
        j, m = make_document(
            [("A", "The pump uses oil. The HCU regulates pressure.")],
            abbreviations={"HCU": "Hydraulic Control Unit"},
            tables={0: [table]},
        )
        first = fw.run(j, m, tmp_path / "1")
        assert first.ok and first.report["validation"]["passed"], first.report["validation"]
        assert first.report["backends"] == {"graph": "neo4j", "vectors": "qdrant"}
        second = fw.run(j, m, tmp_path / "2")
        assert (
            second.report["operations"]["Create"] == 0 and second.report["operations"]["Update"] > 0
        )
        assert second.report["graph"] == first.report["graph"]
        assert second.report["communities"]["unchanged"] == second.report["communities"]["total"]
    finally:
        graph.close()
        _drop_collections(vectors)
        vectors.close()
