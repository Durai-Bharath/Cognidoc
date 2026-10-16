"""A10 / Algorithm 11 - Graph CRUD Agent.

=======  ==================================================================================
Create   insert missing endpoint nodes and the new relationship (with its provenance)
Merge    add the new source's provenance to an existing relationship and its endpoints
         (aliases are unioned, and a ``Concept`` type is refined); confidence = max over sources
Update   replace this source's provenance record (confidence and timestamp) and nothing else
Reject   discard; the conflict record was already written by the Decision Agent
Delete   retract one source from a relationship, or the whole relationship; endpoint
         entities left with no edges at all are removed
=======  ==================================================================================

Each operation is isolated: a failing operation is recorded in ``errors`` and the rest
still apply.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from construction.context import ConstructionAgent
from construction.models import (
    EntityRef,
    GraphEdge,
    GraphNode,
    GraphOperation,
    Operation,
    Provenance,
    StatusedTriple,
    merge_provenance,
    utc_now,
)
from construction.ontology import FALLBACK_TYPE


class CRUDInput(BaseModel):
    operations: list[GraphOperation]


class AppliedOperation(BaseModel):
    op: Operation
    edge_id: str | None = None
    created_nodes: list[str] = Field(default_factory=list)
    touched_nodes: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)


class CRUDResult(BaseModel):
    applied: list[AppliedOperation] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def touched_nodes(self) -> list[str]:
        return list(dict.fromkeys(n for a in self.applied for n in a.touched_nodes))

    @property
    def touched_edges(self) -> list[str]:
        return list(
            dict.fromkeys(a.edge_id for a in self.applied if a.edge_id and a.op != Operation.DELETE)
        )

    def count(self, op: Operation) -> int:
        return sum(a.op == op for a in self.applied)


class GraphCRUDAgent(ConstructionAgent[CRUDInput, CRUDResult]):
    name: ClassVar[str] = "graph_crud"
    algorithm: ClassVar[int] = 11
    agent_id: ClassVar[str] = "A10"

    def _execute(self, data: CRUDInput) -> CRUDResult:
        out = CRUDResult()
        for op in data.operations:
            try:
                applied = self._apply(op)
            except Exception as exc:  # per-operation isolation
                ref = op.item.triple.triple.triple_id if op.item else op.delete_edge_id
                out.errors.append(f"{op.op.value} {ref}: {type(exc).__name__}: {exc}")
                self.log.exception("crud_operation_failed", extra={"op": op.op.value, "ref": ref})
                continue
            out.applied.append(applied)
        return out

    # ------------------------------------------------------------------ dispatch
    def _apply(self, op: GraphOperation) -> AppliedOperation:
        if op.op is Operation.DELETE:
            return self._delete(op)
        if op.op is Operation.REJECT or op.item is None:
            return AppliedOperation(op=Operation.REJECT)
        if op.op is Operation.CREATE:
            return self._create(op.item)
        if op.op is Operation.MERGE:
            return self._merge(op.item)
        return self._update(op.item)

    # ------------------------------------------------------------------ nodes
    def _upsert_entity(self, ref: EntityRef, prov: Provenance) -> bool:
        """Create or enrich an entity node. Returns True when the node was created."""
        graph = self.ctx.graph
        node = graph.get_node(ref.node_id)
        now = utc_now()
        if node is None:
            aliases = [ref.mention] if ref.mention != ref.name else []
            graph.upsert_node(
                GraphNode(
                    id=ref.node_id,
                    label="Entity",
                    name=ref.name,
                    type=ref.type,
                    aliases=aliases,
                    provenance=[prov],
                    created_at=now,
                    updated_at=now,
                )
            )
            return True
        if ref.mention not in (node.name, *node.aliases):
            node.aliases.append(ref.mention)
        if node.type == FALLBACK_TYPE and ref.type != FALLBACK_TYPE:
            node.type = ref.type
        node.provenance = merge_provenance(node.provenance, prov)
        node.updated_at = now
        graph.upsert_node(node)
        return False

    # ------------------------------------------------------------------ operations
    def _create(self, item: StatusedTriple) -> AppliedOperation:
        t = item.triple
        prov = t.triple.provenance
        created = [ref.node_id for ref in (t.subject, t.object) if self._upsert_entity(ref, prov)]
        now = utc_now()
        edge = GraphEdge(
            id=t.edge_id,
            source=t.subject.node_id,
            target=t.object.node_id,
            predicate=t.predicate or "",
            confidence=t.triple.confidence,
            provenance=[prov],
            properties={"raw_relations": [t.triple.relation], "normalization": t.method},
            created_at=now,
            updated_at=now,
        )
        self.ctx.graph.upsert_edge(edge)
        return AppliedOperation(
            op=Operation.CREATE,
            edge_id=edge.id,
            created_nodes=created,
            touched_nodes=[t.subject.node_id, t.object.node_id],
        )

    def _existing(self, item: StatusedTriple) -> GraphEdge:
        eid = item.existing_edge_id or item.triple.edge_id
        edge = self.ctx.graph.get_edge(eid)
        if edge is None:
            raise LookupError(f"edge {eid} disappeared before {item.status.value}")
        return edge

    def _merge(self, item: StatusedTriple) -> AppliedOperation:
        t = item.triple
        prov = t.triple.provenance
        for ref in (t.subject, t.object):
            self._upsert_entity(ref, prov)
        edge = self._existing(item)
        edge.provenance = merge_provenance(edge.provenance, prov)
        raw = list(edge.properties.get("raw_relations", []))
        if t.triple.relation not in raw:
            raw.append(t.triple.relation)
        edge.properties["raw_relations"] = raw
        edge.recompute_confidence()
        edge.updated_at = utc_now()
        self.ctx.graph.upsert_edge(edge)
        return AppliedOperation(
            op=Operation.MERGE, edge_id=edge.id, touched_nodes=[t.subject.node_id, t.object.node_id]
        )

    def _update(self, item: StatusedTriple) -> AppliedOperation:
        t = item.triple
        prov = t.triple.provenance
        graph = self.ctx.graph
        for ref in (t.subject, t.object):
            node = graph.get_node(ref.node_id)
            if node is not None:  # provenance only, no other node changes
                node.provenance = merge_provenance(node.provenance, prov)
                node.updated_at = utc_now()
                graph.upsert_node(node)
        edge = self._existing(item)
        edge.provenance = merge_provenance(edge.provenance, prov)
        edge.recompute_confidence()
        edge.updated_at = utc_now()
        graph.upsert_edge(edge)
        return AppliedOperation(
            op=Operation.UPDATE,
            edge_id=edge.id,
            touched_nodes=[t.subject.node_id, t.object.node_id],
        )

    def _delete(self, op: GraphOperation) -> AppliedOperation:
        graph = self.ctx.graph
        if not op.delete_edge_id:
            raise ValueError("Delete requires delete_edge_id")
        edge = graph.get_edge(op.delete_edge_id)
        if edge is None:
            return AppliedOperation(op=Operation.DELETE, edge_id=op.delete_edge_id)
        deleted: list[str] = []
        if op.delete_source is not None:
            edge.provenance = [p for p in edge.provenance if p.key != tuple(op.delete_source)]
        if op.delete_source is None or not edge.provenance:
            graph.delete_edge(edge.id)
            deleted.append(edge.id)
            for end in (edge.source, edge.target):
                node = graph.get_node(end)
                if node is not None and node.label == "Entity" and graph.degree(end) == 0:
                    graph.delete_node(end)
                    deleted.append(end)
        else:
            edge.recompute_confidence()
            edge.updated_at = utc_now()
            graph.upsert_edge(edge)
        return AppliedOperation(op=Operation.DELETE, edge_id=edge.id, deleted=deleted)
