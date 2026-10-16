"""A14 / Algorithm 15 - Construction Validation Agent (runs once, after all chunks).

Checks the whole KG and VDB:

* ``ontology_compliance``: entity types and predicates are in O, domain/range hold, and
  structural edges connect the right labels;
* ``node_edge_consistency``: endpoints exist, edge ids match their (source, predicate,
  target), confidences lie in [0, 1], no semantic self-loops;
* ``provenance_coverage``: every node and edge has at least one complete record;
* ``graph_connectivity``: every Chunk is ``IN_DOCUMENT`` and every Entity is
  ``MENTIONED_IN`` a chunk. Several disconnected entity components only warn;
* ``valid_entities_and_relations``: no empty, pronoun-only or oversized names (warning);
* ``vector_mapping_integrity``: every Entity/Chunk/Community node has its vector with the
  right ``node_id``, and no vector is left dangling.
"""

from __future__ import annotations

from typing import Any, ClassVar

import networkx as nx
from pydantic import BaseModel, Field

from common.validation import CheckResult, Severity, ValidationOutcome
from construction.agents.coreference import PRONOUNS
from construction.agents.embedding import collection_for
from construction.context import ConstructionAgent
from construction.models import (
    IN_DOCUMENT,
    MEMBER_OF,
    MENTIONED_IN,
    NEXT_CHUNK,
    GraphEdge,
    GraphNode,
    edge_id,
)
from construction.stores.vector_store import COLLECTIONS

STRUCTURAL_SHAPES: dict[str, tuple[str, str]] = {
    IN_DOCUMENT: ("Chunk", "Document"),
    NEXT_CHUNK: ("Chunk", "Chunk"),
    MENTIONED_IN: ("Entity", "Chunk"),
    MEMBER_OF: ("Entity", "Community"),
}
EMBEDDED_LABELS = ("Entity", "Chunk", "Community")


class ValidationRequest(BaseModel):
    document_id: str | None = None


class ConstructionValidationReport(BaseModel):
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    invalid_entities: list[str] = Field(default_factory=list)
    invalid_relations: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.validation.passed


def entity_components(nodes: list[GraphNode], edges: list[GraphEdge]) -> list[set[str]]:
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(n.id for n in nodes if n.label == "Entity")
    g.add_edges_from(
        (e.source, e.target) for e in edges if not e.structural and e.source in g and e.target in g
    )
    return [set(c) for c in nx.connected_components(g)]


class ConstructionValidationAgent(
    ConstructionAgent[ValidationRequest, ConstructionValidationReport]
):
    name: ClassVar[str] = "construction_validation"
    algorithm: ClassVar[int] = 15
    agent_id: ClassVar[str] = "A14"

    def _execute(self, data: ValidationRequest) -> ConstructionValidationReport:
        graph, onto = self.ctx.graph, self.ctx.ontology
        nodes = graph.nodes()
        edges = graph.edges()
        by_id = {n.id: n for n in nodes}
        out = ConstructionValidationReport()

        # Ontology compliance
        onto_fail: list[str] = []
        for n in nodes:
            if n.label == "Entity" and n.type not in onto.entity_types:
                onto_fail.append(f"entity {n.id} has unknown type {n.type!r}")
                out.invalid_entities.append(n.id)
        semantic = [e for e in edges if not e.structural]
        for e in edges:
            s, t = by_id.get(e.source), by_id.get(e.target)
            if s is None or t is None:
                continue
            if e.structural:
                shape = STRUCTURAL_SHAPES[e.predicate]
                if (s.label, t.label) != shape:
                    onto_fail.append(
                        f"{e.id}: {e.predicate} must link {shape}, got {(s.label, t.label)}"
                    )
                    out.invalid_relations.append(e.id)
                continue
            if (s.label, t.label) != ("Entity", "Entity"):
                onto_fail.append(f"{e.id}: semantic edge between {s.label} and {t.label}")
                out.invalid_relations.append(e.id)
                continue
            violation = onto.check_domain_range(e.predicate, s.type, t.type)
            if violation:
                onto_fail.append(f"{e.id}: {violation}")
                out.invalid_relations.append(e.id)
        out.validation.add(CheckResult.from_failures("ontology_compliance", onto_fail))

        # Node / relationship consistency
        cons: list[str] = []
        for e in edges:
            for end in (e.source, e.target):
                if end not in by_id:
                    cons.append(f"{e.id}: dangling endpoint {end}")
            if e.id != edge_id(e.source, e.predicate, e.target):
                cons.append(f"{e.id}: id does not match (source, predicate, target)")
            if not 0.0 <= e.confidence <= 1.0:
                cons.append(f"{e.id}: confidence {e.confidence} out of range")
            if not e.structural and e.source == e.target:
                cons.append(f"{e.id}: self-loop")
        out.validation.add(CheckResult.from_failures("node_edge_consistency", cons))

        # Provenance coverage
        prov_fail: list[str] = []
        for ref, records in [(n.id, n.provenance) for n in nodes] + [
            (e.id, e.provenance) for e in edges
        ]:
            if not records:
                prov_fail.append(f"{ref}: no provenance")
            elif all(p.missing_fields() for p in records):
                prov_fail.append(f"{ref}: incomplete provenance {records[0].missing_fields()}")
        out.validation.add(CheckResult.from_failures("provenance_coverage", prov_fail))

        # Connectivity
        conn: list[str] = []
        in_doc = {e.source for e in edges if e.predicate == IN_DOCUMENT}
        mentioned = {e.source for e in edges if e.predicate == MENTIONED_IN}
        for n in nodes:
            if n.label == "Chunk" and n.id not in in_doc:
                conn.append(f"chunk {n.id} is not linked to a document")
            if n.label == "Entity" and n.id not in mentioned:
                conn.append(f"entity {n.id} is not linked to any chunk")
        out.validation.add(CheckResult.from_failures("graph_connectivity", conn))
        components = entity_components(nodes, edges)
        out.validation.add(
            CheckResult.from_failures(
                "single_entity_component",
                [f"{len(components)} disconnected entity components"]
                if len(components) > 1
                else [],
                Severity.WARNING,
            )
        )

        # Invalid entities / relations
        max_len = self.settings.validation.max_entity_name_chars
        bad: list[str] = []
        for n in nodes:
            if n.label != "Entity":
                continue
            name = n.name.strip()
            if not name or name.lower() in PRONOUNS or len(name) > max_len:
                bad.append(f"entity {n.id} has an invalid name {n.name[:40]!r}")
                out.invalid_entities.append(n.id)
        for e in semantic:
            if e.predicate not in onto.predicates:
                bad.append(f"relation {e.id} uses unknown predicate {e.predicate!r}")
        out.validation.add(
            CheckResult.from_failures("valid_entities_and_relations", bad, Severity.WARNING)
        )

        # Vector mappings
        vec_fail: list[str] = []
        store = self.ctx.vectors
        expected: dict[str, set[str]] = {c: set() for c in COLLECTIONS}
        for n in nodes:
            if n.label not in EMBEDDED_LABELS:
                continue
            coll = collection_for(n)
            expected[coll].add(n.id)
            if n.vector_id is None:
                vec_fail.append(f"{n.label} {n.id} has no vector_id")
        for coll, ids in expected.items():
            stored = {r.id: r for r in store.get(coll, sorted(ids))} if ids else {}
            for nid in ids:
                rec = stored.get(nid)
                if rec is None:
                    vec_fail.append(f"{coll}/{nid}: vector missing")
                elif rec.payload.get("node_id") != nid:
                    vec_fail.append(f"{coll}/{nid}: vector maps to {rec.payload.get('node_id')}")
            for vid in set(store.ids(coll)) - ids:
                vec_fail.append(f"{coll}/{vid}: dangling vector (no graph node)")
        out.validation.add(CheckResult.from_failures("vector_mapping_integrity", vec_fail))

        out.invalid_entities = sorted(set(out.invalid_entities))
        out.invalid_relations = sorted(set(out.invalid_relations))
        compliant = len(semantic) - len(
            {r for r in out.invalid_relations if r in {e.id for e in semantic}}
        )
        out.stats = {
            "nodes": len(nodes),
            "edges": len(edges),
            "semantic_edges": len(semantic),
            "entity_components": len(components),
            "ontology_compliance_ratio": round(compliant / len(semantic), 4) if semantic else 1.0,
            "vectors": {c: store.count(c) for c in COLLECTIONS},
        }
        self.flag_if_failed(out.validation, "construction validation failed")
        return out
