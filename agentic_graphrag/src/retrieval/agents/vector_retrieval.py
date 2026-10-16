"""A4 / Algorithm 5 - Vector Retrieval Agent.

Embeds the subquery (line 2), searches the ``chunks`` collection for the top-k chunks and
the ``entities`` collection for the top entities (lines 3-4), and ranks both by cosine
similarity (line 5).

Every hit is hydrated from the graph: chunk text, document and page come from the Chunk
node, never from the vector payload. A vector whose node no longer exists is reported in
``stale_hits`` and dropped, so later citations can only point at real graph nodes.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from construction.stores.vector_store import CHUNKS, ENTITIES
from retrieval.context import RetrievalAgent
from retrieval.models import ChunkHit, EntityHit, RetrievalParams, VectorEvidence


class VectorRetrievalInput(BaseModel):
    subquery_id: str
    text: str
    params: RetrievalParams


class VectorRetrievalAgent(RetrievalAgent[VectorRetrievalInput, VectorEvidence]):
    name: ClassVar[str] = "vector_retrieval"
    algorithm: ClassVar[int] = 5
    agent_id: ClassVar[str] = "A4"

    def _execute(self, data: VectorRetrievalInput) -> VectorEvidence:
        graph, vectors = self.ctx.graph, self.ctx.vectors
        min_score = self.settings.vector.min_score
        vec = self.ctx.embedder.embed_one(data.text)
        out = VectorEvidence(subquery_id=data.subquery_id, query_text=data.text)

        for hit in vectors.search(CHUNKS, vec, top_k=data.params.top_k):
            if hit.score < min_score:
                continue
            node = graph.get_node(str(hit.payload.get("node_id", hit.id)))
            if node is None or node.label != "Chunk":
                out.stale_hits.append(hit.id)
                continue
            props = node.properties
            out.chunks.append(
                ChunkHit(
                    chunk_id=node.id,
                    doc_id=str(props.get("doc_id") or ""),
                    page=props.get("page_start"),
                    section=" > ".join(props.get("section_path") or []),
                    text=str(props.get("text", "")),
                    score=round(hit.score, 4),
                )
            )
        if data.params.entity_top_k:
            for hit in vectors.search(ENTITIES, vec, top_k=data.params.entity_top_k):
                if hit.score < min_score:
                    continue
                node = graph.get_node(str(hit.payload.get("node_id", hit.id)))
                if node is None or node.label != "Entity":
                    out.stale_hits.append(hit.id)
                    continue
                out.entities.append(
                    EntityHit(
                        node_id=node.id, name=node.name, type=node.type, score=round(hit.score, 4)
                    )
                )
        out.chunks.sort(key=lambda c: (-c.score, c.chunk_id))
        out.entities.sort(key=lambda e: (-e.score, e.node_id))
        return out

    def confidence(self, output: VectorEvidence) -> float | None:
        return output.chunks[0].score if output.chunks else 0.0
