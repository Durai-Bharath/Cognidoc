"""A8 Conflict Detection, A9 Decision, A10 Graph CRUD."""

from __future__ import annotations

from construction.agents.conflict_detection import (
    ConflictDetectionAgent,
    ConflictInput,
    ConflictReport,
)
from construction.agents.decision import DecisionAgent
from construction.agents.graph_crud import CRUDInput, GraphCRUDAgent
from construction.context import ConstructionContext
from construction.models import (
    ConflictStatus,
    GraphOperation,
    Operation,
    StatusedTriple,
    edge_id,
    entity_node_id,
)
from tests.construction.builders import DOC, add_edge, add_entity, normalized, prov


def _detect(ctx: ConstructionContext, *items) -> ConflictReport:  # type: ignore[no-untyped-def]
    return ConflictDetectionAgent(ctx).run(ConflictInput(items=list(items))).unwrap()


def _status(ctx: ConstructionContext, item) -> StatusedTriple:  # type: ignore[no-untyped-def]
    return _detect(ctx, item).items[0]


def _apply(ctx: ConstructionContext, *ops: GraphOperation):  # type: ignore[no-untyped-def]
    return GraphCRUDAgent(ctx).run(CRUDInput(operations=list(ops))).unwrap()


# ----------------------------------------------------------------------------- A8


def test_new_when_relationship_absent(cctx: ConstructionContext) -> None:
    s = _status(cctx, normalized("Pump", "uses", "oil"))
    assert s.status is ConflictStatus.NEW and s.existing_edge_id is None


def test_update_when_same_source_reasserts(cctx: ConstructionContext) -> None:
    e = add_edge(cctx.graph, "Pump", "uses", "oil", 0.7)  # type: ignore[arg-type]
    s = _status(cctx, normalized("Pump", "uses", "oil", 0.9))
    assert (
        s.status is ConflictStatus.UPDATE and s.existing_edge_id == e.id and "0.7 → 0.9" in s.reason
    )


def test_merge_when_new_source_corroborates(cctx: ConstructionContext) -> None:
    add_edge(cctx.graph, "Pump", "uses", "oil")  # type: ignore[arg-type]
    other = prov(doc="doc-other", chunk_id="doc-other:c0")
    s = _status(cctx, normalized("Pump", "uses", "oil", provenance=other))
    assert s.status is ConflictStatus.MERGE


def test_conflict_on_functional_predicate_with_different_value(cctx: ConstructionContext) -> None:
    e = add_edge(cctx.graph, "Seal", "has_interval", "500 h")  # type: ignore[arg-type]
    other = prov(doc="doc-b", chunk_id="doc-b:c0")
    s = _status(cctx, normalized("Seal", "has_interval", "1000 h", provenance=other))
    assert s.status is ConflictStatus.CONFLICT and s.conflicting_edge_ids == [e.id]
    assert "functional" in s.reason
    # Non-functional predicates can have many objects.
    assert _status(cctx, normalized("Seal", "uses", "grease")).status is ConflictStatus.NEW


def test_conflict_on_exclusive_predicates_in_kg_and_batch(cctx: ConstructionContext) -> None:
    add_edge(cctx.graph, "Pump", "requires", "oil")  # type: ignore[arg-type]
    s = _status(cctx, normalized("Pump", "prohibits", "oil"))
    assert s.status is ConflictStatus.CONFLICT and "contradicts existing requires" in s.reason
    out = _detect(
        cctx,
        normalized("Valve", "causes", "leak"),
        normalized("Valve", "prevents", "leak"),
        normalized("Filter", "has_status", "OK"),
        normalized("Filter", "has_status", "Replaced"),
    )
    assert [i.status for i in out.items] == [
        ConflictStatus.NEW,
        ConflictStatus.CONFLICT,
        ConflictStatus.NEW,
        ConflictStatus.CONFLICT,
    ]
    assert out.count(ConflictStatus.CONFLICT) == 2


def test_reject_cases(cctx: ConstructionContext) -> None:
    no_pred = _status(
        cctx, normalized("A", None, "B", rejected="relation 'x' has no ontology predicate")
    )
    assert no_pred.status is ConflictStatus.REJECT and "no ontology predicate" in no_pred.reason
    dr = _status(
        cctx, normalized("Report", "authored_by", "42", head_type="Document", tail_type="Quantity")
    )
    assert dr.status is ConflictStatus.REJECT and "range" in dr.reason
    loop = normalized("Pump", "uses", "oil")
    loop.resolved.tail = loop.resolved.head
    assert _status(cctx, loop).status is ConflictStatus.REJECT


# ----------------------------------------------------------------------------- A9


def test_decisions_map_statuses_and_log_conflict_records(cctx: ConstructionContext) -> None:
    e = add_edge(cctx.graph, "Seal", "has_interval", "500 h")  # type: ignore[arg-type]
    add_edge(cctx.graph, "Pump", "uses", "oil")  # type: ignore[arg-type]
    other = prov(doc="doc-b", chunk_id="doc-b:c0")
    report = _detect(
        cctx,
        normalized("Valve", "uses", "water"),  # New
        normalized("Pump", "uses", "oil", provenance=other),  # Merge
        normalized("Pump", "uses", "oil"),  # Update
        normalized("Seal", "has_interval", "1000 h", provenance=other),  # Conflict
        normalized("A", None, "B", rejected="no predicate"),  # Reject
    )
    d = DecisionAgent(cctx).run(report).unwrap()
    assert [o.op for o in d.operations] == [
        Operation.CREATE,
        Operation.MERGE,
        Operation.UPDATE,
        Operation.REJECT,
        Operation.REJECT,
    ]
    assert [c.status for c in d.conflicts] == [ConflictStatus.CONFLICT, ConflictStatus.REJECT]
    conflict = d.conflicts[0]
    assert (
        conflict.incoming["tail"] == "1000 h"
        and conflict.incoming["provenance"]["doc_id"] == "doc-b"
    )
    assert conflict.existing[0]["edge_id"] == e.id and conflict.existing[0]["tail"] == "500 h"
    assert conflict.existing[0]["provenance"][0]["doc_id"] == DOC
    assert cctx.conflicts == d.conflicts
    flagged = [i for i in cctx.review_queue.items if i.agent == "decision"]
    assert len(flagged) == 2 and flagged[0].details["conflict_id"] == conflict.conflict_id


# ----------------------------------------------------------------------------- A10


def _op(ctx: ConstructionContext, item, op: Operation) -> GraphOperation:  # type: ignore[no-untyped-def]
    return GraphOperation(op=op, item=_status(ctx, item))


def test_create_inserts_nodes_and_relationship(cctx: ConstructionContext) -> None:
    out = _apply(
        cctx, _op(cctx, normalized("Pump", "uses", "oil", head_type="Component"), Operation.CREATE)
    )
    eid = edge_id(entity_node_id("Pump"), "uses", entity_node_id("oil"))
    assert out.count(Operation.CREATE) == 1 and out.touched_edges == [eid]
    assert set(out.applied[0].created_nodes) == {entity_node_id("Pump"), entity_node_id("oil")}
    edge = cctx.graph.get_edge(eid)
    assert (
        edge is not None
        and edge.provenance[0].doc_id == DOC
        and edge.properties["raw_relations"] == ["uses"]
    )
    assert cctx.graph.get_node(entity_node_id("Pump")).type == "Component"  # type: ignore[union-attr]


def test_create_respects_inversion_and_existing_nodes(cctx: ConstructionContext) -> None:
    add_entity(cctx.graph, "System", "Concept")  # type: ignore[arg-type]
    item = normalized("System", "part_of", "Pump", inverted=True, head_type="System")
    item.resolved.head = item.resolved.head.model_copy(
        update={"mention": "the system", "is_new": False}
    )
    out = _apply(cctx, _op(cctx, item, Operation.CREATE))
    edge = cctx.graph.get_edge(out.touched_edges[0])
    assert edge.source == entity_node_id("Pump") and edge.target == entity_node_id("System")  # type: ignore[union-attr]
    node = cctx.graph.get_node(entity_node_id("System"))
    assert node.type == "System" and "the system" in node.aliases  # type: ignore[union-attr]
    assert out.applied[0].created_nodes == [entity_node_id("Pump")]


def test_merge_adds_provenance_and_takes_max_confidence(cctx: ConstructionContext) -> None:
    e = add_edge(cctx.graph, "Pump", "uses", "oil", 0.6)  # type: ignore[arg-type]
    other = prov(doc="doc-b", chunk_id="doc-b:c0")
    item = normalized("Pump", "uses", "oil", 0.8, provenance=other)
    item.resolved.triple.relation = "utilises"
    _apply(cctx, _op(cctx, item, Operation.MERGE))
    edge = cctx.graph.get_edge(e.id)
    assert edge is not None and edge.support == 2 and edge.confidence == 0.8
    assert edge.properties["raw_relations"] == ["utilises"]
    assert {p.doc_id for p in cctx.graph.get_node(e.source).provenance} == {DOC, "doc-b"}  # type: ignore[union-attr]


def test_update_changes_only_confidence_and_provenance(cctx: ConstructionContext) -> None:
    e = add_edge(cctx.graph, "Pump", "uses", "oil", 0.6)  # type: ignore[arg-type]
    before = cctx.graph.get_edge(e.id)
    _apply(cctx, _op(cctx, normalized("Pump", "uses", "oil", 0.9), Operation.UPDATE))
    after = cctx.graph.get_edge(e.id)
    assert after is not None and before is not None
    assert after.confidence == 0.9 and after.support == 1 and after.provenance[0].confidence == 0.9
    assert (after.source, after.target, after.predicate, after.properties) == (
        before.source,
        before.target,
        before.predicate,
        before.properties,
    )


def test_reject_discards_and_failed_operations_are_isolated(cctx: ConstructionContext) -> None:
    reject = GraphOperation(
        op=Operation.REJECT, item=_status(cctx, normalized("A", None, "B", rejected="x"))
    )
    broken = _op(cctx, normalized("Ghost", "uses", "oil"), Operation.UPDATE)  # edge does not exist
    good = _op(cctx, normalized("Pump", "uses", "oil"), Operation.CREATE)
    out = _apply(cctx, reject, broken, good)
    assert out.count(Operation.REJECT) == 1 and out.count(Operation.CREATE) == 1
    assert len(out.errors) == 1 and "disappeared" in out.errors[0]
    assert cctx.graph.get_node(entity_node_id("A")) is None


def test_delete_retracts_source_then_removes_edge_and_orphans(cctx: ConstructionContext) -> None:
    e = add_edge(cctx.graph, "Pump", "uses", "oil")  # type: ignore[arg-type]
    other = prov(doc="doc-b", chunk_id="doc-b:c0")
    _apply(cctx, _op(cctx, normalized("Pump", "uses", "oil", provenance=other), Operation.MERGE))
    _apply(
        cctx,
        GraphOperation(
            op=Operation.DELETE, delete_edge_id=e.id, delete_source=("doc-b", "doc-b:c0")
        ),
    )
    assert cctx.graph.get_edge(e.id).support == 1  # type: ignore[union-attr]
    out = _apply(cctx, GraphOperation(op=Operation.DELETE, delete_edge_id=e.id))
    assert cctx.graph.get_edge(e.id) is None and set(out.applied[0].deleted) == {
        e.id,
        e.source,
        e.target,
    }
    out = _apply(
        cctx,
        GraphOperation(op=Operation.DELETE, delete_edge_id="missing"),
        GraphOperation(op=Operation.DELETE),
    )
    assert out.count(Operation.DELETE) == 1 and len(out.errors) == 1
