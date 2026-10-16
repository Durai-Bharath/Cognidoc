"""A13 / Algorithm 14 - Vector Index Agent.

Upserts each embedding into its collection under the graph node's id. That stores the
mapping in both directions: ``payload.node_id`` on the vector, ``vector_id`` on the node.
It then reads every vector back to verify it was indexed: present, the right node, and
the same vector (cosine ≥ 0.999). A verification failure is a typed error.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from common.base_agent import AgentFailure
from common.embedder import cosine
from construction.agents.embedding import EmbeddingBatch
from construction.context import ConstructionAgent
from construction.stores.vector_store import VectorRecord

VERIFY_MIN_COSINE = 0.999


def _same_vector(stored: list[float], sent: list[float]) -> bool:
    """Identical, or the same direction (backends may re-normalise; zero vectors have none)."""
    a, b = np.asarray(stored, np.float32), np.asarray(sent, np.float32)
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, atol=1e-5)) or cosine(a, b) >= VERIFY_MIN_COSINE


class IndexReport(BaseModel):
    upserted: dict[str, int] = Field(default_factory=dict)
    verified: int = 0
    mapped_nodes: list[str] = Field(default_factory=list)


class VectorIndexAgent(ConstructionAgent[EmbeddingBatch, IndexReport]):
    name: ClassVar[str] = "vector_index"
    algorithm: ClassVar[int] = 14
    agent_id: ClassVar[str] = "A13"

    def _execute(self, data: EmbeddingBatch) -> IndexReport:
        store = self.ctx.vectors
        graph = self.ctx.graph
        out = IndexReport()
        by_collection: dict[str, list[VectorRecord]] = {}
        for item in data.items:
            by_collection.setdefault(item.collection, []).append(
                VectorRecord(id=item.node_id, vector=item.vector, payload=item.payload)
            )
        for collection, records in by_collection.items():
            store.ensure_collection(collection, data.dim)
            store.upsert(collection, records)
            out.upserted[collection] = len(records)

        failures: list[str] = []
        for collection, records in by_collection.items():
            stored = {r.id: r for r in store.get(collection, [r.id for r in records])}
            for rec in records:
                got = stored.get(rec.id)
                if got is None:
                    failures.append(f"{collection}/{rec.id}: not found after upsert")
                elif got.payload.get("node_id") != rec.id:
                    failures.append(f"{collection}/{rec.id}: wrong node mapping")
                elif not _same_vector(got.vector, rec.vector):
                    failures.append(f"{collection}/{rec.id}: stored vector differs")
                else:
                    out.verified += 1
                    node = graph.get_node(rec.id)
                    if node is not None and node.vector_id != rec.id:
                        node.vector_id = rec.id
                        graph.upsert_node(node)
                    out.mapped_nodes.append(rec.id)
        if failures:
            self.flag("vector indexing verification failed", details={"failures": failures})
            raise AgentFailure("; ".join(failures[:5]), error_type="VectorIndexError")
        return out
