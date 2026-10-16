"""GraphStore / VectorStore in-memory backends, ontology loading and config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from construction.config import ConstructionConfig, load_construction_config
from construction.models import GraphEdge, GraphNode, edge_id
from construction.ontology import OntologyError, load_ontology
from construction.stores import (
    ENTITIES,
    GraphStoreError,
    InMemoryGraphStore,
    InMemoryVectorStore,
    VectorRecord,
    VectorStoreError,
    build_graph_store,
    build_vector_store,
    save_state,
)
from tests.construction.builders import add_edge, add_entity, prov

ROOT = Path(__file__).resolve().parents[2]


def test_graph_store_crud_neighbors_and_deletes() -> None:
    g = InMemoryGraphStore()
    e = add_edge(g, "Pump", "part_of", "System")
    add_edge(g, "Valve", "part_of", "System")
    assert g.get_edge(e.id) == e
    assert {n.name for n in g.neighbors(e.target, direction="in")} == {"Pump", "Valve"}
    assert [n.name for n in g.neighbors(e.source, direction="out", predicate="part_of")] == [
        "System"
    ]
    assert len(g.find_edges(target=e.target)) == 2 and g.find_edges(source="missing") == []
    # upsert replaces in place
    g.upsert_edge(e.model_copy(update={"confidence": 0.5}))
    assert g.get_edge(e.id).confidence == 0.5 and len(g.edges()) == 2  # type: ignore[union-attr]
    g.delete_edge(e.id)
    assert g.get_edge(e.id) is None and len(g.edges()) == 1
    g.delete_node(e.target)  # detaches remaining edges
    assert g.edges() == [] and g.get_node(e.target) is None
    with pytest.raises(GraphStoreError):
        g.upsert_edge(GraphEdge(id="x", source="a", target="b", predicate="p"))


def test_graph_store_similarity_and_persistence(tmp_path: Path) -> None:
    g = InMemoryGraphStore()
    a, b = add_entity(g, "alpha"), add_entity(g, "beta")
    g.upsert_node(GraphNode(id="c1", label="Chunk", name="chunk"))
    g.set_embedding(a.id, np.array([1, 0, 0], np.float32))
    g.set_embedding(b.id, np.array([0, 1, 0], np.float32))
    g.set_embedding("c1", np.array([1, 0, 0], np.float32))
    hits = g.similar_nodes(np.array([0.9, 0.1, 0], np.float32), label="Entity", top_k=5)
    assert [n.name for n, _ in hits] == ["alpha", "beta"] and hits[0][1] > 0.99
    with pytest.raises(GraphStoreError):
        g.set_embedding("nope", np.zeros(3, np.float32))
    path = g.save(tmp_path / "g.json")
    g2 = InMemoryGraphStore.load(path)
    assert {n.id for n in g2.nodes()} == {n.id for n in g.nodes()}
    assert np.allclose(g2.get_embedding(a.id), g.get_embedding(a.id))  # type: ignore[arg-type]
    assert InMemoryGraphStore.load(tmp_path / "missing.json").nodes() == []
    exported = Path(g.export(tmp_path / "kg.json")).read_text()
    assert '"backend": "memory"' in exported


def test_vector_store_upsert_get_search_delete(tmp_path: Path) -> None:
    v = InMemoryVectorStore()
    v.ensure_collection(ENTITIES, 2)
    v.upsert(
        ENTITIES,
        [
            VectorRecord(id="a", vector=[1, 0], payload={"node_id": "a", "k": 1}),
            VectorRecord(id="b", vector=[0, 1], payload={"node_id": "b", "k": 2}),
        ],
    )
    assert [h.id for h in v.search(ENTITIES, np.array([1, 0.1], np.float32), top_k=1)] == ["a"]
    assert [h.id for h in v.search(ENTITIES, np.array([1, 0], np.float32), where={"k": 2})] == ["b"]
    assert v.count(ENTITIES) == 2 and [r.id for r in v.get(ENTITIES, ["b", "zz"])] == ["b"]
    with pytest.raises(VectorStoreError):
        v.upsert(ENTITIES, [VectorRecord(id="c", vector=[1, 2, 3])])
    with pytest.raises(VectorStoreError):
        v.ensure_collection(ENTITIES, 3)
    with pytest.raises(VectorStoreError):
        v.upsert("unknown", [])
    v2 = InMemoryVectorStore.load(v.save(tmp_path / "v.json"))
    assert v2.ids(ENTITIES) == ["a", "b"]
    v.delete(ENTITIES, ["a"])
    assert v.ids(ENTITIES) == ["b"]


def test_build_stores_and_save_state(cconfig: ConstructionConfig) -> None:
    g, v = build_graph_store(cconfig), build_vector_store(cconfig)
    add_entity(g, "Pump")  # type: ignore[arg-type]
    v.ensure_collection(ENTITIES, 2)
    paths = save_state(g, v, cconfig.paths.state_dir)
    assert set(paths) == {"graph_state", "vector_state"}
    assert [n.name for n in build_graph_store(cconfig).nodes()] == ["Pump"]
    assert build_graph_store(cconfig, load_state=False).nodes() == []
    assert build_vector_store(cconfig, load_state=False).ids(ENTITIES) == []


def test_default_ontology_matches_relations() -> None:
    o = load_ontology(ROOT / "ontology.yaml")
    assert o.match_relation("regulates").method == "exact"  # type: ignore[union-attr]
    m = o.match_relation("is inspected every")
    assert m and m.predicate == "has_interval" and not m.inverted
    m = o.match_relation("contains")
    assert m and m.predicate == "part_of" and m.inverted
    m = o.match_relation("utilising")  # stem of a synonym
    assert m is None or m.method == "stem"
    assert o.match_relation("teleports") is None
    assert (
        o.canonical_type("Software/Framework") == "Software"
        and o.canonical_type("weird") == "Concept"
    )
    assert o.is_functional("has_interval") and not o.is_functional("uses")
    assert o.exclusive_with("requires") == {"prohibits"}
    assert o.check_domain_range("authored_by", "Document", "Person") is None
    assert "range" in (o.check_domain_range("authored_by", "Document", "Quantity") or "")
    assert (
        o.check_domain_range("authored_by", "Concept", "Concept") is None
    )  # Concept is a wildcard
    assert "not in the ontology" in (o.check_domain_range("nope", "Concept", "Concept") or "")


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"predicates": {"BadName": {}}}, "snake_case"),
        ({"predicates": {"p": {"domain": ["Alien"]}}}, "unknown entity type"),
        ({"predicates": {"p": {"synonyms": ["uses"]}}}, "maps to both"),
        ({"exclusive": [["uses", "ghost"]]}, "unknown predicate"),
        ({"entity_types": {"Thing": {}}}, "fallback"),
    ],
)
def test_invalid_ontology_is_rejected(
    tmp_path: Path, patch: dict[str, object], message: str
) -> None:
    data = yaml.safe_load((ROOT / "ontology.yaml").read_text())
    for key, value in patch.items():
        if key == "entity_types":
            data[key] = value
        elif isinstance(value, dict):
            data[key].update(value)
        else:
            data[key] = value
    path = tmp_path / "o.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(OntologyError, match=message):
        load_ontology(path)
    with pytest.raises(OntologyError):
        load_ontology(tmp_path / "missing.yaml")


def test_construction_config_loading(tmp_path: Path) -> None:
    cfg = load_construction_config(ROOT / "construction.yaml")
    assert Path(cfg.paths.ontology).is_absolute() and Path(cfg.paths.ontology).exists()
    assert load_construction_config(None) == ConstructionConfig()
    bad = tmp_path / "c.yaml"
    bad.write_text("chunking: {max_tokens: 50, overlap_tokens: 60}\n")
    with pytest.raises(ValueError, match="overlap"):
        load_construction_config(bad)
    bad.write_text("- 1\n")
    with pytest.raises(ValueError, match="mapping"):
        load_construction_config(bad)
    ext = ConstructionConfig().as_extraction_config()
    assert ext.llm == ConstructionConfig().llm


def test_edge_ids_are_deterministic() -> None:
    assert edge_id("a", "p", "b") == edge_id("a", "p", "b") != edge_id("b", "p", "a")
    assert prov().missing_fields() == [] and prov(page=None).missing_fields() == ["page"]
