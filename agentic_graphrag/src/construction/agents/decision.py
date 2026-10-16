"""A9 / Algorithm 10 - Decision Agent.

New → Create, Merge → Merge, Update → Update, anything else → Reject. In Alg. 10,
Conflict and Reject both collapse to Reject. Neither is silently dropped: each one becomes
a :class:`ConflictRecord` (incoming triple, the existing edges it clashes with, both
provenances, the reason), kept on the context for the construction report and written
to the review queue.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from construction.agents.conflict_detection import ConflictReport
from construction.context import ConstructionAgent
from construction.models import (
    ConflictRecord,
    ConflictStatus,
    GraphOperation,
    Operation,
    edge_summary,
    stable_hash,
    triple_summary,
)

_DECISIONS = {
    ConflictStatus.NEW: Operation.CREATE,
    ConflictStatus.MERGE: Operation.MERGE,
    ConflictStatus.UPDATE: Operation.UPDATE,
}


class Decisions(BaseModel):
    operations: list[GraphOperation] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)

    def count(self, op: Operation) -> int:
        return sum(o.op == op for o in self.operations)


class DecisionAgent(ConstructionAgent[ConflictReport, Decisions]):
    name: ClassVar[str] = "decision"
    algorithm: ClassVar[int] = 10
    agent_id: ClassVar[str] = "A9"

    def _execute(self, data: ConflictReport) -> Decisions:
        out = Decisions()
        for item in data.items:
            op = _DECISIONS.get(item.status, Operation.REJECT)
            out.operations.append(GraphOperation(op=op, item=item, reason=item.reason))
            if op is not Operation.REJECT:
                continue
            existing = [
                e for eid in item.conflicting_edge_ids if (e := self.ctx.graph.get_edge(eid))
            ]
            names = {
                n.id: n.name
                for e in existing
                for n in (self.ctx.graph.get_node(e.source), self.ctx.graph.get_node(e.target))
                if n is not None
            }
            incoming = triple_summary(item.triple)
            record = ConflictRecord(
                conflict_id="cf-"
                + stable_hash(
                    item.status.value, item.triple.triple.triple_id, *item.conflicting_edge_ids
                ),
                status=item.status,
                reason=item.reason,
                incoming=incoming,
                existing=[edge_summary(e, names) for e in existing],
            )
            out.conflicts.append(record)
            self.ctx.conflicts.append(record)
            self.flag(
                f"{item.status.value.lower()}: {item.reason}",
                item_ref=item.triple.triple.triple_id,
                details=record.model_dump(),
            )
        return out
