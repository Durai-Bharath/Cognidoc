"""A8 / Algorithm 9 - Conflict Detection Agent.

For every normalised triple, it compares entity, relation, confidence and provenance with
the KG. The first rule that matches decides the status:

=========  ================================================================================
Reject     no ontology predicate (A7), self-loop, or an ontology domain/range violation
Update     the edge exists and already has a provenance record for this (doc, chunk)
           source: re-ingestion refreshes confidence and provenance only
Merge      the edge exists and this is a new source: provenance is appended
Conflict   the edge is new, but the predicate is *functional* and the subject already has
           a different object, or an *exclusive* predicate links the same pair (in the
           KG, or earlier in this batch)
New        everything else
=========  ================================================================================
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from construction.context import ConstructionAgent
from construction.models import ConflictStatus, NormalizedTriple, StatusedTriple, edge_id


class ConflictInput(BaseModel):
    items: list[NormalizedTriple]


class ConflictReport(BaseModel):
    items: list[StatusedTriple] = Field(default_factory=list)

    def count(self, status: ConflictStatus) -> int:
        return sum(i.status == status for i in self.items)


class ConflictDetectionAgent(ConstructionAgent[ConflictInput, ConflictReport]):
    name: ClassVar[str] = "conflict_detection"
    algorithm: ClassVar[int] = 9
    agent_id: ClassVar[str] = "A8"

    def _execute(self, data: ConflictInput) -> ConflictReport:
        onto = self.ctx.ontology
        graph = self.ctx.graph
        out = ConflictReport()
        # (subject, predicate) → object staged by earlier New triples in this batch.
        staged: dict[tuple[str, str], str] = {}
        staged_pairs: dict[tuple[str, str], set[str]] = {}

        for item in data.items:

            def emit(
                status: ConflictStatus,
                reason: str = "",
                existing: str | None = None,
                conflicting: list[str] | None = None,
                item: NormalizedTriple = item,
            ) -> None:
                out.items.append(
                    StatusedTriple(
                        triple=item,
                        status=status,
                        reason=reason,
                        existing_edge_id=existing,
                        conflicting_edge_ids=conflicting or [],
                    )
                )

            if item.predicate is None:
                emit(ConflictStatus.REJECT, item.rejected_reason or "no predicate")
                continue
            pred = item.predicate
            subj, obj = item.subject, item.object
            if subj.node_id == obj.node_id:
                emit(ConflictStatus.REJECT, f"self-loop on {subj.name!r} after entity resolution")
                continue
            violation = onto.check_domain_range(pred, subj.type, obj.type)
            if violation:
                emit(ConflictStatus.REJECT, violation)
                continue

            eid = edge_id(subj.node_id, pred, obj.node_id)
            existing = graph.get_edge(eid)
            if existing is not None:
                incoming = item.triple.provenance.key
                if any(p.key == incoming for p in existing.provenance):
                    old = next(p for p in existing.provenance if p.key == incoming)
                    emit(
                        ConflictStatus.UPDATE,
                        f"same source re-asserted (confidence {old.confidence} → "
                        f"{item.triple.confidence})",
                        existing=eid,
                    )
                else:
                    emit(
                        ConflictStatus.MERGE,
                        f"corroborated by a new source ({existing.support} existing)",
                        existing=eid,
                    )
                continue

            conflicting: list[str] = []
            reasons: list[str] = []
            if onto.is_functional(pred):
                others = [
                    e
                    for e in graph.find_edges(source=subj.node_id, predicate=pred)
                    if e.target != obj.node_id
                ]
                if others:
                    conflicting += [e.id for e in others]
                    reasons.append(
                        f"{pred} is functional and {subj.name!r} already has a different value"
                    )
                staged_obj = staged.get((subj.node_id, pred))
                if staged_obj is not None and staged_obj != obj.node_id:
                    reasons.append(f"{pred} is functional; conflicting value earlier in this chunk")
            for other_pred in sorted(onto.exclusive_with(pred)):
                clash = graph.find_edges(
                    source=subj.node_id, target=obj.node_id, predicate=other_pred
                )
                if clash:
                    conflicting += [e.id for e in clash]
                    reasons.append(f"{pred} contradicts existing {other_pred}")
                if other_pred in staged_pairs.get((subj.node_id, obj.node_id), set()):
                    reasons.append(f"{pred} contradicts {other_pred} earlier in this chunk")
            if reasons:
                emit(ConflictStatus.CONFLICT, "; ".join(reasons), conflicting=conflicting)
                continue

            emit(ConflictStatus.NEW, "relationship not in the knowledge graph")
            staged.setdefault((subj.node_id, pred), obj.node_id)
            staged_pairs.setdefault((subj.node_id, obj.node_id), set()).add(pred)
        return out
